from __future__ import annotations

"""Batched self-play for AlphaZero-style training.

The dominant cost of self-play is running the neural network once per MCTS
leaf, one position at a time.  This module removes that bottleneck by playing
many *independent* games concurrently in a single process and batching the
network evaluation of all their in-flight leaves into a **single** forward pass
per simulation step.

Because every game contributes at most one in-flight leaf per step and the
games share no search state, the batching is purely across independent trees:
there is no need for virtual loss and the search semantics are identical to the
single-game :func:`alpha_chess.self_play.play_game`.

Public symbols
--------------
Node
    Lightweight per-game search-tree node (per-child N/W/Q/P + child moves).
BatchedSelfPlay
    Driver that plays waves of concurrent games and emits training examples.
generate_selfplay_data
    Convenience wrapper around :class:`BatchedSelfPlay`.

Every training example produced here is identical in meaning and format to the
one produced by :func:`alpha_chess.self_play.play_game`::

    {"state": np.ndarray(19, 8, 8) float32,
     "policy": np.ndarray(POLICY_SIZE,) float32,
     "value": float}
"""

import contextlib
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import chess
import torch

from alpha_chess.encoding import (
    encode_board,
    move_to_index,
    POLICY_SIZE,
    NUM_PLANES,
)
from alpha_chess.network import get_device


# --------------------------------------------------------------------------- #
# Search-tree node
# --------------------------------------------------------------------------- #
class Node:
    """A node in a per-game PUCT search tree.

    Statistics are stored per child edge (keyed by the ``chess.Move`` leading to
    the child) and expressed from the perspective of the player to move at *this*
    node -- the same convention used by :mod:`alpha_chess.mcts`.

        child_N -- visit counts
        child_W -- summed action values (backups)
        child_Q -- mean action values (W / N)
        child_P -- prior probabilities from the network policy
    """

    __slots__ = ("prior", "is_expanded", "children",
                 "child_N", "child_W", "child_Q", "child_P")

    def __init__(self, prior: float = 0.0) -> None:
        self.prior: float = prior
        self.is_expanded: bool = False
        self.children: Dict[chess.Move, "Node"] = {}
        self.child_N: Dict[chess.Move, int] = {}
        self.child_W: Dict[chess.Move, float] = {}
        self.child_Q: Dict[chess.Move, float] = {}
        self.child_P: Dict[chess.Move, float] = {}

    def total_visits(self) -> int:
        """Sum of visit counts over all children of this node."""
        return sum(self.child_N.values())


class _Game:
    """Per-game state carried through a wave of concurrent self-play."""

    __slots__ = ("board", "root", "examples", "move_count", "done")

    def __init__(self) -> None:
        self.board: chess.Board = chess.Board()
        self.root: Optional[Node] = None
        self.examples: List[dict] = []
        self.move_count: int = 0
        self.done: bool = False


class BatchedSelfPlay:
    """Play many independent self-play games at once, batching NN evaluations.

    Parameters
    ----------
    network:
        An ``AlphaZeroNet`` (or compatible) returning ``(policy_logits, value)``
        for a ``(B, NUM_PLANES, 8, 8)`` input.  It is used in eval mode.
    device:
        Torch device for inference; defaults to :func:`get_device`.
    num_parallel_games:
        Maximum number of games run concurrently within a wave (the batch size
        of a full network call equals the number of games still active).
    num_simulations:
        PUCT simulations per move.
    c_puct:
        Exploration constant in the PUCT selection rule.
    dirichlet_alpha, dirichlet_epsilon:
        Root Dirichlet-noise parameters (applied per game, at the root only).
    temperature_moves:
        Number of opening plies played by sampling from the visit distribution
        (temperature 1); subsequent plies are played greedily (argmax).
    max_moves:
        Hard cap on plies per game; hitting it finalises the game as a draw.
    seed:
        Optional seed for the (numpy + python) RNGs used for Dirichlet noise and
        move sampling.
    """

    def __init__(
        self,
        network,
        device=None,
        num_parallel_games: int = 64,
        num_simulations: int = 200,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        temperature_moves: int = 30,
        max_moves: int = 400,
        seed: Optional[int] = None,
    ) -> None:
        self.network = network
        self.device = device if device is not None else get_device()
        self.num_parallel_games = int(num_parallel_games)
        self.num_simulations = int(num_simulations)
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_epsilon = float(dirichlet_epsilon)
        self.temperature_moves = int(temperature_moves)
        self.max_moves = int(max_moves)

        # Independent RNGs so results are reproducible for a given seed.
        self._np_rng = np.random.RandomState(seed)
        self._py_rng = random.Random(seed)

        # Keep the network on the right device for inference.
        self.network.to(self.device)

    # ------------------------------------------------------------------ #
    # Batched network evaluation
    # ------------------------------------------------------------------ #
    def _run_network(
        self, boards: List[chess.Board]
    ) -> List[Tuple[Dict[chess.Move, float], float]]:
        """Evaluate a batch of boards in a single forward pass.

        Returns one ``(priors, value)`` pair per input board where ``priors``
        maps each legal move to its softmax-over-legal-moves prior and ``value``
        is the network's evaluation from the perspective of the side to move.
        Input boards are never mutated.
        """
        states = np.stack(
            [encode_board(b) for b in boards]
        ).astype(np.float32, copy=False)  # (B, NUM_PLANES, 8, 8)
        tensor = torch.from_numpy(states).to(self.device)

        # Autocast (fp16) only on CUDA; a no-op elsewhere so CPU/MPS stay fp32.
        if self.device.type == "cuda":
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
        else:
            autocast_ctx = contextlib.nullcontext()

        with torch.no_grad(), autocast_ctx:
            policy_logits, value = self.network(tensor)

        # Cast back to fp32 on the host for the softmax bookkeeping.
        policy_logits = policy_logits.float().cpu().numpy()   # (B, POLICY_SIZE)
        values = value.float().cpu().numpy().reshape(-1)      # (B,)

        results: List[Tuple[Dict[chess.Move, float], float]] = []
        for i, board in enumerate(boards):
            legal_moves = list(board.legal_moves)
            priors: Dict[chess.Move, float] = {}
            if legal_moves:
                indices = np.array(
                    [move_to_index(m, board) for m in legal_moves],
                    dtype=np.int64,
                )
                legal_logits = policy_logits[i][indices]
                # Numerically stable softmax over the legal-move logits only.
                legal_logits = legal_logits - np.max(legal_logits)
                exps = np.exp(legal_logits)
                probs = exps / np.sum(exps)
                for move, p in zip(legal_moves, probs):
                    priors[move] = float(p)
            results.append((priors, float(values[i])))
        return results

    # ------------------------------------------------------------------ #
    # Expansion / selection / noise / backup
    # ------------------------------------------------------------------ #
    @staticmethod
    def _expand_node(node: Node, priors: Dict[chess.Move, float]) -> None:
        """Attach children to ``node`` from a move->prior mapping."""
        for move, prior in priors.items():
            node.children[move] = Node(prior=prior)
            node.child_N[move] = 0
            node.child_W[move] = 0.0
            node.child_Q[move] = 0.0
            node.child_P[move] = prior
        node.is_expanded = True

    def _select_child(self, node: Node) -> chess.Move:
        """Select the child maximising the PUCT score.

        score = Q + c_puct * P * sqrt(sum(N_parent)) / (1 + N_child)
        """
        sqrt_total = math.sqrt(node.total_visits())  # sqrt(0)=0 on first descent
        best_move: Optional[chess.Move] = None
        best_score = -float("inf")
        for move in node.children:
            q = node.child_Q[move]
            p = node.child_P[move]
            n = node.child_N[move]
            score = q + self.c_puct * p * sqrt_total / (1 + n)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _add_dirichlet_noise(self, node: Node) -> None:
        """Mix per-game Dirichlet noise into the root priors in place.

        P = (1 - eps) * P + eps * noise
        """
        moves = list(node.children.keys())
        if not moves:
            return
        noise = self._np_rng.dirichlet([self.dirichlet_alpha] * len(moves))
        eps = self.dirichlet_epsilon
        for move, n in zip(moves, noise):
            mixed = (1.0 - eps) * node.child_P[move] + eps * float(n)
            node.child_P[move] = mixed
            node.children[move].prior = mixed

    @staticmethod
    def _terminal_value(board: chess.Board) -> float:
        """Game-theoretic value of a terminal board, from side-to-move view.

        The side to move can never be the winner, so checkmate is ``-1.0`` and
        every other terminal outcome (stalemate, repetition, 50-move,
        insufficient material) is a draw ``0.0``.
        """
        if board.is_checkmate():
            return -1.0
        return 0.0

    @staticmethod
    def _backup(path: List[Tuple[Node, chess.Move]], value: float) -> None:
        """Back a leaf value up ``path``, negating the perspective per ply."""
        for parent, move in reversed(path):
            value = -value
            parent.child_N[move] += 1
            parent.child_W[move] += value
            parent.child_Q[move] = parent.child_W[move] / parent.child_N[move]

    def _descend(
        self, root: Node, game_board: chess.Board
    ) -> Tuple[Node, chess.Board, List[Tuple[Node, chess.Move]]]:
        """Descend from ``root`` by PUCT to a leaf.

        Returns ``(leaf_node, leaf_board, path)`` where ``leaf_board`` is a fresh
        copy (``game_board`` is never mutated) and ``path`` is the list of
        traversed ``(parent_node, move)`` edges.
        """
        node = root
        board = game_board.copy()
        path: List[Tuple[Node, chess.Move]] = []
        while node.is_expanded and not board.is_game_over(claim_draw=True):
            move = self._select_child(node)
            path.append((node, move))
            board.push(move)
            node = node.children[move]
        return node, board, path

    # ------------------------------------------------------------------ #
    # Move selection from root visit counts
    # ------------------------------------------------------------------ #
    def _root_distribution(
        self, root: Node, board: chess.Board
    ) -> Dict[chess.Move, float]:
        """Normalised root visit-count distribution (matches ``MCTS.run``)."""
        total = root.total_visits()
        distribution: Dict[chess.Move, float] = {}
        if total > 0:
            for move, n in root.child_N.items():
                distribution[move] = n / total
        else:
            # Degenerate fallback: uniform over legal moves.
            legal = list(board.legal_moves)
            if legal:
                uniform = 1.0 / len(legal)
                for move in legal:
                    distribution[move] = uniform
        return distribution

    def _sample_move(self, distribution: Dict[chess.Move, float]) -> chess.Move:
        """Sample a move proportional to its visit probability (temperature 1)."""
        moves = list(distribution.keys())
        probs = np.asarray([distribution[m] for m in moves], dtype=np.float64)
        total = probs.sum()
        if total <= 0:
            return moves[self._py_rng.randrange(len(moves))]
        probs = probs / total
        r = self._py_rng.random()
        cumulative = 0.0
        for move, p in zip(moves, probs):
            cumulative += p
            if r <= cumulative:
                return move
        return moves[-1]

    @staticmethod
    def _argmax_move(distribution: Dict[chess.Move, float]) -> chess.Move:
        return max(distribution.items(), key=lambda kv: kv[1])[0]

    # ------------------------------------------------------------------ #
    # One ply across all active games
    # ------------------------------------------------------------------ #
    def _play_one_ply(self, active: List[_Game]) -> None:
        """Run a full search and make one move for every active game.

        All active games are non-terminal.  For each of ``num_simulations``
        steps every active game descends to a leaf; the non-terminal leaves are
        evaluated together in a single network call.
        """
        # --- Fresh roots, expanded together, then per-game root noise. ---
        for g in active:
            g.root = Node()
        root_evals = self._run_network([g.board for g in active])
        for g, (priors, _value) in zip(active, root_evals):
            self._expand_node(g.root, priors)
            self._add_dirichlet_noise(g.root)

        # --- Simulations: one batched leaf evaluation per step. ---
        for _ in range(self.num_simulations):
            pending: List[Tuple[Node, List[Tuple[Node, chess.Move]]]] = []
            pending_boards: List[chess.Board] = []
            for g in active:
                leaf_node, leaf_board, path = self._descend(g.root, g.board)
                if leaf_board.is_game_over(claim_draw=True):
                    # Terminal leaf: exact value, no network call.
                    self._backup(path, self._terminal_value(leaf_board))
                else:
                    pending.append((leaf_node, path))
                    pending_boards.append(leaf_board)

            if pending_boards:
                evals = self._run_network(pending_boards)
                for (leaf_node, path), (priors, value) in zip(pending, evals):
                    self._expand_node(leaf_node, priors)
                    self._backup(path, value)

        # --- Record training targets, pick + play a move, maybe finalise. ---
        for g in active:
            distribution = self._root_distribution(g.root, g.board)

            target = np.zeros(POLICY_SIZE, dtype=np.float32)
            for move, prob in distribution.items():
                target[move_to_index(move, g.board)] = prob

            g.examples.append(
                {
                    "state": encode_board(g.board),
                    "policy": target,
                    "turn": g.board.turn,
                }
            )

            if g.move_count < self.temperature_moves:
                chosen = self._sample_move(distribution)
            else:
                chosen = self._argmax_move(distribution)

            g.board.push(chosen)
            g.move_count += 1
            g.root = None  # drop the tree; a fresh one is built next ply

            if g.board.is_game_over(claim_draw=True) or g.move_count >= self.max_moves:
                self._finalize(g)

    @staticmethod
    def _finalize(game: _Game) -> None:
        """Assign value targets from each mover's perspective; mark game done."""
        board = game.board
        if board.is_game_over(claim_draw=True):
            result_str = board.result(claim_draw=True)
            if result_str == "1-0":
                result = 1.0
            elif result_str == "0-1":
                result = -1.0
            else:
                result = 0.0
        else:
            # Max-move cutoff: treat as a draw.
            result = 0.0

        for example in game.examples:
            example["value"] = result if example["turn"] == chess.WHITE else -result
            del example["turn"]
        game.done = True

    # ------------------------------------------------------------------ #
    # Wave / public API
    # ------------------------------------------------------------------ #
    def _play_wave(self, wave_size: int) -> List[dict]:
        """Play ``wave_size`` games to completion and return their examples."""
        games = [_Game() for _ in range(wave_size)]
        while True:
            active = [g for g in games if not g.done]
            if not active:
                break
            self._play_one_ply(active)

        examples: List[dict] = []
        for g in games:
            examples.extend(g.examples)
        return examples

    def generate(self, num_games: int) -> List[dict]:
        """Play ``num_games`` self-play games and return all training examples.

        Games are played in waves of up to ``num_parallel_games`` at a time
        until at least ``num_games`` have completed.  The returned list is the
        flat concatenation of every game's examples, each a dict
        ``{"state", "policy", "value"}`` matching ``self_play.play_game``.
        """
        self.network.eval()  # BatchNorm/eval-mode inference for batched calls

        examples: List[dict] = []
        completed = 0
        while completed < num_games:
            wave_size = min(self.num_parallel_games, num_games - completed)
            examples.extend(self._play_wave(wave_size))
            completed += wave_size
        return examples


def generate_selfplay_data(
    network,
    device=None,
    num_games: int = 1,
    **kwargs,
) -> List[dict]:
    """Convenience wrapper: build a :class:`BatchedSelfPlay` and generate data.

    Extra keyword arguments (``num_parallel_games``, ``num_simulations``,
    ``c_puct``, ``dirichlet_alpha``, ``dirichlet_epsilon``, ``temperature_moves``,
    ``max_moves``, ``seed``) are forwarded to :class:`BatchedSelfPlay`.
    """
    player = BatchedSelfPlay(network, device=device, **kwargs)
    return player.generate(num_games)
