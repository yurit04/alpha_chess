"""Monte-Carlo Tree Search (PUCT) guided by an AlphaZero-style network.

This module implements a network-guided PUCT search as described in the
AlphaZero paper.  The search is driven by a policy/value network: the policy
provides prior probabilities over legal moves and the value provides a scalar
evaluation of a leaf position from the perspective of the side to move.

Public symbols:
    Node   -- internal search-tree node (exposed for testing/inspection).
    MCTS   -- the search driver (run / best_move).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import chess
import numpy as np
import torch

from alpha_chess.encoding import encode_board, move_to_index, POLICY_SIZE, NUM_PLANES
from alpha_chess.network import get_device


class Node:
    """A node in the PUCT search tree.

    A node corresponds to a board position.  It stores the aggregate search
    statistics for each of its children (indexed by the ``chess.Move`` that
    leads to the child):

        N -- visit count
        W -- total action value (summed backups)
        Q -- mean action value (W / N)
        P -- prior probability from the network policy

    The value of every stat is expressed from the perspective of the player to
    move at *this* node (the parent), which is the convention used by the PUCT
    formula and the backup step.
    """

    def __init__(self, prior: float = 0.0) -> None:
        self.prior: float = prior          # prior of the edge leading into this node
        self.is_expanded: bool = False
        self.children: Dict[chess.Move, "Node"] = {}
        # Per-child aggregate statistics.
        self.child_N: Dict[chess.Move, int] = {}
        self.child_W: Dict[chess.Move, float] = {}
        self.child_Q: Dict[chess.Move, float] = {}
        self.child_P: Dict[chess.Move, float] = {}

    def total_visits(self) -> int:
        """Sum of visit counts over all children of this node."""
        return sum(self.child_N.values())


class MCTS:
    """Network-guided PUCT Monte-Carlo Tree Search.

    Parameters
    ----------
    network:
        An ``AlphaZeroNet`` (or compatible) module returning
        ``(policy_logits, value)`` for a ``(B, NUM_PLANES, 8, 8)`` input.
    device:
        Torch device used for inference.  Defaults to ``get_device()``.
    c_puct:
        Exploration constant in the PUCT selection rule.
    dirichlet_alpha, dirichlet_epsilon:
        Root Dirichlet-noise parameters (used only when ``add_noise=True``).
    """

    def __init__(
        self,
        network,
        device=None,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ) -> None:
        self.network = network
        self.device = device if device is not None else get_device()
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        # Keep the network on the right device and in eval mode for inference.
        self.network.to(self.device)
        self.network.eval()

    # ------------------------------------------------------------------ #
    # Network evaluation
    # ------------------------------------------------------------------ #
    def _evaluate(self, board: chess.Board):
        """Run a single-position network evaluation.

        Returns a tuple ``(priors, value)`` where ``priors`` is a dict mapping
        each legal move to its (softmax-normalised over legal moves) prior
        probability, and ``value`` is a Python float in ``[-1, 1]`` giving the
        network's evaluation of ``board`` from the perspective of the side to
        move.  The passed-in board is never mutated.
        """
        state = encode_board(board)  # (NUM_PLANES, 8, 8) float32
        tensor = torch.from_numpy(np.asarray(state, dtype=np.float32))
        tensor = tensor.unsqueeze(0).to(self.device)  # (1, NUM_PLANES, 8, 8)

        with torch.no_grad():
            policy_logits, value = self.network(tensor)

        policy_logits = policy_logits.squeeze(0).detach().cpu().numpy()  # (POLICY_SIZE,)
        value_scalar = float(value.squeeze().item())

        legal_moves = list(board.legal_moves)
        priors: Dict[chess.Move, float] = {}
        if legal_moves:
            indices = np.array(
                [move_to_index(m, board) for m in legal_moves], dtype=np.int64
            )
            legal_logits = policy_logits[indices]
            # Numerically stable softmax over the legal-move logits only.
            legal_logits = legal_logits - np.max(legal_logits)
            exps = np.exp(legal_logits)
            probs = exps / np.sum(exps)
            for move, p in zip(legal_moves, probs):
                priors[move] = float(p)

        return priors, value_scalar

    # ------------------------------------------------------------------ #
    # Expansion
    # ------------------------------------------------------------------ #
    def _expand(self, node: Node, board: chess.Board) -> float:
        """Expand ``node`` for ``board`` and return the leaf value.

        The returned value is the network's evaluation of ``board`` from the
        perspective of the side to move at ``board`` (i.e. at ``node``).  The
        board is not mutated.
        """
        priors, value = self._evaluate(board)
        for move, prior in priors.items():
            child = Node(prior=prior)
            node.children[move] = child
            node.child_N[move] = 0
            node.child_W[move] = 0.0
            node.child_Q[move] = 0.0
            node.child_P[move] = prior
        node.is_expanded = True
        return value

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #
    def _select_child(self, node: Node) -> chess.Move:
        """Select the child maximising the PUCT score.

        score = Q + c_puct * P * sqrt(sum(N_parent)) / (1 + N_child)
        """
        total_n = node.total_visits()
        sqrt_total = math.sqrt(total_n)  # sqrt(0) == 0 on the first descent

        best_move: Optional[chess.Move] = None
        best_score = -float("inf")
        for move, child in node.children.items():
            q = node.child_Q[move]
            p = node.child_P[move]
            n = node.child_N[move]
            u = self.c_puct * p * sqrt_total / (1 + n)
            score = q + u
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    # ------------------------------------------------------------------ #
    # Root noise
    # ------------------------------------------------------------------ #
    def _add_dirichlet_noise(self, node: Node) -> None:
        """Mix Dirichlet noise into the root priors in place.

        P = (1 - eps) * P + eps * noise
        """
        moves = list(node.children.keys())
        if not moves:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))
        eps = self.dirichlet_epsilon
        for move, n in zip(moves, noise):
            mixed = (1.0 - eps) * node.child_P[move] + eps * float(n)
            node.child_P[move] = mixed
            node.children[move].prior = mixed

    # ------------------------------------------------------------------ #
    # Terminal handling
    # ------------------------------------------------------------------ #
    @staticmethod
    def _terminal_value(board: chess.Board) -> float:
        """Return the game-theoretic value of a terminal ``board``.

        Value is from the perspective of the side to move at ``board``.  In
        chess the side to move can never be the winner: checkmate means the
        side to move has been mated (value -1.0); every other terminal
        outcome (stalemate, insufficient material, repetition, 50-move) is a
        draw (value 0.0).
        """
        if board.is_checkmate():
            return -1.0
        return 0.0

    # ------------------------------------------------------------------ #
    # A single simulation
    # ------------------------------------------------------------------ #
    def _simulate(self, root: Node, board: chess.Board) -> None:
        """Run one selection/expansion/backup simulation from ``root``.

        ``board`` is treated as scratch: it is mutated with push/pop during the
        descent and fully restored (via pops) before returning, so callers see
        no net change.
        """
        node = root
        path: List[tuple] = []  # list of (parent_node, move) edges traversed
        pushed = 0

        # --- Selection: descend until we hit a leaf or a terminal node. ---
        while node.is_expanded and not board.is_game_over(claim_draw=True):
            move = self._select_child(node)
            path.append((node, move))
            board.push(move)
            pushed += 1
            node = node.children[move]

        # --- Evaluate the leaf. ---
        if board.is_game_over(claim_draw=True):
            # Terminal position: use the exact game-theoretic value.
            value = self._terminal_value(board)
        else:
            # Non-terminal leaf: expand and use the network evaluation.
            value = self._expand(node, board)

        # ``value`` is from the perspective of the player to move at the leaf.
        # --- Backup: walk back up the path, negating at each ply. ---
        for parent, move in reversed(path):
            # The stats at ``parent`` are stored from the perspective of the
            # player to move at ``parent``.  Moving from the leaf back up one
            # ply flips the perspective, so negate before applying.
            value = -value
            parent.child_N[move] += 1
            parent.child_W[move] += value
            parent.child_Q[move] = parent.child_W[move] / parent.child_N[move]

        # Restore the board to its original state.
        for _ in range(pushed):
            board.pop()

    # ------------------------------------------------------------------ #
    # Public search API
    # ------------------------------------------------------------------ #
    def run(
        self,
        board: chess.Board,
        num_simulations: int,
        add_noise: bool = False,
    ) -> Dict[chess.Move, float]:
        """Run ``num_simulations`` PUCT simulations from ``board``.

        Returns the normalised root visit-count distribution as a dict mapping
        each legal move to ``N_child / sum(N)``.  No temperature is applied
        here; the caller is responsible for any temperature scaling/sampling.
        The input ``board`` is never left mutated.
        """
        root = Node()

        # Expand the root once so it has children (and priors) before search.
        if not board.is_game_over(claim_draw=True):
            self._expand(root, board)
            if add_noise:
                self._add_dirichlet_noise(root)

        for _ in range(num_simulations):
            self._simulate(root, board)

        total = root.total_visits()
        distribution: Dict[chess.Move, float] = {}
        if total > 0:
            for move, n in root.child_N.items():
                distribution[move] = n / total
        else:
            # Degenerate case (e.g. zero simulations): fall back to a uniform
            # distribution over the legal moves so the result is still valid.
            legal = list(board.legal_moves)
            if legal:
                uniform = 1.0 / len(legal)
                for move in legal:
                    distribution[move] = uniform
        return distribution

    def best_move(self, board: chess.Board, num_simulations: int) -> chess.Move:
        """Return the most-visited move after running the search."""
        distribution = self.run(board, num_simulations, add_noise=False)
        return max(distribution, key=distribution.get)
