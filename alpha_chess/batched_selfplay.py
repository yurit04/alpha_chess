from __future__ import annotations

"""High-throughput self-play for AlphaZero-style training.

The tree search is pure Python and the network is on the GPU, so a naive
implementation leaves the GPU almost completely idle: one process descends a
tree on one core while the device waits.  This module removes that imbalance:

* **Many games in flight per process.**  Every in-flight game contributes at
  most one leaf per simulation step, so a step turns into a single batched
  forward pass.  Games share no search state, so no virtual loss is needed and
  the search semantics match a plain sequential PUCT search.
* **Continuous refill.**  A finished game is immediately replaced by a fresh
  one, so the batch stays at full width instead of decaying to a handful of
  stragglers at the end of a wave.
* **Worker processes behind one inference server.**  Tree search is CPU-bound
  and holds the GIL, so it is spread over worker *processes* that exchange
  positions with a single GPU-owning server through shared memory.  The server
  concatenates every worker's pending request into one large forward pass.
* **O(1) draw detection.**  ``board.is_game_over(claim_draw=True)`` costs ~159us
  because it replays the move stack looking for repetitions; it used to be
  called at every node of every descent.  Repetition counts and the halfmove
  clock are tracked incrementally along the descent instead, and terminal
  status is only ever resolved at a leaf, reusing the legal-move list that the
  leaf needs anyway.
* **Subtree reuse.**  The subtree under the played move is carried into the next
  ply instead of being thrown away, so its visits still count.

Public symbols
--------------
Node
    Lightweight per-game search-tree node.
SelfPlayEngine
    Single-process driver: runs a pool of games against an ``evaluate`` callable.
SelfPlayBatch
    Column-oriented result of a self-play run (states / sparse policy / values).
generate_selfplay_data
    Top-level entry point; runs the multi-process pipeline.
"""

import math
import os
import random
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import chess

from alpha_chess.encoding import (
    encode_board,
    move_to_index,
    pack_state,
    NUM_PLANES,
)

# Policy targets are stored sparsely: a dense (POLICY_SIZE,) float32 row costs
# 18.7 KB, versus 0.5 KB for the at-most-K moves that actually receive visits.
# Chess positions with more than this many *visited* moves are vanishingly rare
# (the legal-move count averages ~35); the tail is truncated and renormalized.
MAX_POLICY_TARGETS = 80


# --------------------------------------------------------------------------- #
# Search-tree node
# --------------------------------------------------------------------------- #
class Node:
    """A node in a per-game PUCT search tree.

    Child statistics live in parallel Python lists indexed by child slot rather
    than in dicts keyed by ``chess.Move``.  For the ~35 children of a typical
    chess node a tight list loop selects a child in ~1.9us, against ~15us for
    the dict form and ~3.0us for a NumPy formulation (whose per-call overhead
    dominates at this size).

    All statistics are from the perspective of the player to move at *this*
    node:

        N -- visit counts       W -- summed backups
        Q -- mean action value  P -- network priors
    """

    __slots__ = ("moves", "move_idx", "P", "N", "W", "Q", "children",
                 "n_total", "is_expanded", "terminal_value")

    def __init__(self) -> None:
        self.moves: List[chess.Move] = []
        # Policy indices of ``moves``, cached from expansion so building the
        # visit-count training target costs no extra move encoding.
        self.move_idx: List[int] = []
        self.P: List[float] = []
        self.N: List[int] = []
        self.W: List[float] = []
        self.Q: List[float] = []
        self.children: List[Optional["Node"]] = []
        self.n_total: int = 0
        self.is_expanded: bool = False
        # None while the node is a normal (non-terminal) position; otherwise the
        # exact game-theoretic value from the side-to-move's perspective.
        self.terminal_value: Optional[float] = None

    def expand(
        self,
        moves: List[chess.Move],
        move_idx: List[int],
        priors: List[float],
    ) -> None:
        """Attach children for ``moves`` with the given priors."""
        n = len(moves)
        self.moves = moves
        self.move_idx = move_idx
        self.P = priors
        self.N = [0] * n
        self.W = [0.0] * n
        self.Q = [0.0] * n
        self.children = [None] * n
        self.is_expanded = True


class _Game:
    """Per-game state carried through the self-play pool."""

    __slots__ = ("board", "root", "rep", "cur_rep", "noise_pending",
                 "states", "pol_idx", "pol_val", "turns", "move_count",
                 "sims_left", "allow_resign", "resign_streak",
                 "would_resign_at", "would_resign_side", "resigned")

    def __init__(self, allow_resign: bool) -> None:
        self.board = chess.Board()
        self.root = Node()
        # Repetition counts for the *actual* game line, keyed by transposition
        # key. The starting position counts as one occurrence.
        self.rep: Dict[int, int] = {self.board._transposition_key(): 1}
        # Prior occurrences of the position now on the board (0 = first time).
        self.cur_rep = 0
        # Root Dirichlet noise is applied once per ply, as soon as the root is
        # known to be expanded (which may be immediately, under subtree reuse).
        self.noise_pending = True
        self.states: List[np.ndarray] = []
        self.pol_idx: List[np.ndarray] = []
        self.pol_val: List[np.ndarray] = []
        self.turns: List[bool] = []
        self.move_count = 0
        self.sims_left = 0
        self.allow_resign = allow_resign
        self.resign_streak = 0
        # Ply at which this game *would* have resigned, for games that are
        # played out anyway to measure the resign false-positive rate.
        self.would_resign_at: Optional[int] = None
        self.would_resign_side: Optional[bool] = None
        self.resigned = False


class SelfPlayBatch:
    """Column-oriented self-play output, ready to append to a replay buffer.

    ``states`` is uint8-packed (see :func:`alpha_chess.encoding.pack_state`).
    ``pol_idx``/``pol_val`` hold the sparse policy target: row ``i`` lists the
    policy indices that received visits and their normalized visit shares,
    padded with zeros beyond ``pol_len[i]``.
    """

    __slots__ = ("states", "pol_idx", "pol_val", "pol_len", "values", "stats")

    def __init__(self, states, pol_idx, pol_val, pol_len, values, stats=None):
        self.states = states
        self.pol_idx = pol_idx
        self.pol_val = pol_val
        self.pol_len = pol_len
        self.values = values
        self.stats: Dict[str, float] = stats or {}

    def __len__(self) -> int:
        return int(self.states.shape[0])

    @staticmethod
    def empty() -> "SelfPlayBatch":
        return SelfPlayBatch(
            np.zeros((0, NUM_PLANES, 8, 8), dtype=np.uint8),
            np.zeros((0, MAX_POLICY_TARGETS), dtype=np.uint16),
            np.zeros((0, MAX_POLICY_TARGETS), dtype=np.float32),
            np.zeros((0,), dtype=np.int16),
            np.zeros((0,), dtype=np.float32),
        )

    @staticmethod
    def concat(parts: List["SelfPlayBatch"]) -> "SelfPlayBatch":
        parts = [p for p in parts if len(p) > 0]
        if not parts:
            return SelfPlayBatch.empty()
        stats: Dict[str, float] = {}
        for p in parts:
            for k, v in p.stats.items():
                stats[k] = stats.get(k, 0.0) + v
        return SelfPlayBatch(
            np.concatenate([p.states for p in parts]),
            np.concatenate([p.pol_idx for p in parts]),
            np.concatenate([p.pol_val for p in parts]),
            np.concatenate([p.pol_len for p in parts]),
            np.concatenate([p.values for p in parts]),
            stats,
        )


# --------------------------------------------------------------------------- #
# Evaluator adapters
# --------------------------------------------------------------------------- #
class _SyncEvaluator:
    """Adapts a plain ``evaluate(states, idx, counts)`` callable to submit/wait.

    The engine always drives evaluation asynchronously so it can overlap tree
    search with inference; a synchronous callable simply computes on submit.
    """

    __slots__ = ("evaluate", "_results")

    def __init__(self, fn) -> None:
        self.evaluate = fn
        self._results = {}

    def submit(self, slot, states, idx, counts) -> None:
        self._results[slot] = self.evaluate(states, idx, counts)

    def wait(self, slot):
        return self._results.pop(slot)


# --------------------------------------------------------------------------- #
# Single-process engine
# --------------------------------------------------------------------------- #
class SelfPlayEngine:
    """Run a pool of concurrent self-play games against a batched evaluator.

    ``evaluate`` is called as ``evaluate(states, idx, counts)`` where ``states``
    is ``(B, NUM_PLANES, 8, 8)`` float32, ``idx`` is ``(B, L)`` int32 holding
    each position's legal-move policy indices (padded) and ``counts`` is
    ``(B,)``.  It must return ``(priors, values)`` with ``priors`` of shape
    ``(B, L)`` -- a softmax already restricted to each row's legal moves -- and
    ``values`` of shape ``(B,)`` from the side-to-move's perspective.

    Parameters
    ----------
    games_in_flight:
        Number of games searched concurrently.  Together with the worker count
        this sets how many positions the network sees per forward pass, so it
        is the main GPU-utilization knob.
    pipeline_stages:
        How many sub-pools to split the games into (see
        ``DEFAULT_PIPELINE_STAGES``).
    num_simulations:
        PUCT simulations backing each played move.  Visits carried over from the
        previous ply via subtree reuse count towards this total.
    resign_threshold:
        Resign once the mover's best root value stays at or below this for
        ``resign_plies`` consecutive plies.  ``None`` disables resignation.
    resign_disable_fraction:
        Fraction of games played to the end with resignation suppressed, used to
        measure how often resignation would have thrown away a non-loss.
    """

    def __init__(
        self,
        evaluate: Callable,
        games_in_flight: int = 128,
        num_simulations: int = 200,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        temperature_moves: int = 30,
        max_moves: int = 400,
        pipeline_stages: Optional[int] = None,
        resign_threshold: Optional[float] = -0.90,
        resign_plies: int = 2,
        resign_disable_fraction: float = 0.10,
        seed: Optional[int] = None,
    ) -> None:
        self.evaluator = (
            evaluate
            if hasattr(evaluate, "submit") and hasattr(evaluate, "wait")
            else _SyncEvaluator(evaluate)
        )
        self.evaluate = getattr(self.evaluator, "evaluate", evaluate)
        self.games_in_flight = int(games_in_flight)
        # See DEFAULT_PIPELINE_STAGES: the pool is split this many ways so
        # several requests per worker are always in flight.
        self.stages = resolve_stages(games_in_flight, pipeline_stages)
        self.num_simulations = int(num_simulations)
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_epsilon = float(dirichlet_epsilon)
        self.temperature_moves = int(temperature_moves)
        self.max_moves = int(max_moves)
        self.resign_threshold = resign_threshold
        self.resign_plies = int(resign_plies)
        self.resign_disable_fraction = float(resign_disable_fraction)
        self._rng = np.random.RandomState(seed)
        self._py_rng = random.Random(seed)

    # ------------------------------------------------------------------ #
    # Descent
    # ------------------------------------------------------------------ #
    def _select(self, node: Node) -> int:
        """Index of the child maximizing ``Q + c*P*sqrt(N_parent)/(1+N)``."""
        coeff = self.c_puct * math.sqrt(node.n_total) if node.n_total else 0.0
        Q, P, N = node.Q, node.P, node.N
        best_i = 0
        best_score = -1e30
        if coeff:
            for i in range(len(P)):
                score = Q[i] + coeff * P[i] / (1 + N[i])
                if score > best_score:
                    best_score = score
                    best_i = i
        else:
            # First descent: every N is 0 and sqrt(0) zeroes the exploration
            # term, so the prior alone decides.
            for i in range(len(P)):
                if P[i] > best_score:
                    best_score = P[i]
                    best_i = i
        return best_i

    def _descend(self, game: _Game):
        """Walk from the root to a leaf.

        Returns ``(leaf, board, path, rep_count)``.  ``board`` is a stack-free
        copy (``copy(stack=False)`` costs 0.8us against 11.1us for a full copy;
        the move stack is only needed for python-chess's own repetition
        detection, which this module replaces).  ``rep_count`` is how many times
        the leaf position already occurred along the game line plus this
        descent.
        """
        node = game.root
        board = game.board.copy(stack=False)
        base_rep = game.rep
        local_rep: Dict[int, int] = {}
        path: List[Tuple[Node, int]] = []
        rep_count = base_rep.get(board._transposition_key(), 1) - 1

        while node.is_expanded and node.terminal_value is None:
            i = self._select(node)
            path.append((node, i))
            board.push(node.moves[i])
            key = board._transposition_key()
            seen = base_rep.get(key, 0) + local_rep.get(key, 0)
            local_rep[key] = local_rep.get(key, 0) + 1
            rep_count = seen
            child = node.children[i]
            if child is None:
                child = Node()
                node.children[i] = child
            node = child

        return node, board, path, rep_count

    @staticmethod
    def _backup(path: List[Tuple[Node, int]], value: float) -> None:
        """Back ``value`` up ``path``, flipping perspective at every ply."""
        for parent, i in reversed(path):
            value = -value
            n = parent.N[i] + 1
            w = parent.W[i] + value
            parent.N[i] = n
            parent.W[i] = w
            parent.Q[i] = w / n
            parent.n_total += 1

    @staticmethod
    def _resolve_terminal(
        board: chess.Board, legal: List[chess.Move], rep_count: int
    ) -> Optional[float]:
        """Terminal value from the side-to-move's view, or ``None`` if in play.

        ``legal`` is the caller's already-computed legal-move list, so this adds
        no move generation of its own.  The three claimable-draw conditions are
        resolved from incrementally tracked state rather than by replaying the
        move stack.
        """
        if not legal:
            # No legal move: checkmate for the opponent, else stalemate.
            return -1.0 if board.is_check() else 0.0
        if rep_count >= 2:
            return 0.0  # third occurrence of this position
        if board.halfmove_clock >= 100:
            return 0.0  # fifty-move rule
        if board.is_insufficient_material():
            return 0.0
        return None

    def _add_root_noise(self, node: Node) -> None:
        """Mix Dirichlet noise into the root priors: ``P = (1-e)P + e*noise``."""
        n = len(node.P)
        if n == 0:
            return
        noise = self._rng.dirichlet([self.dirichlet_alpha] * n)
        eps = self.dirichlet_epsilon
        P = node.P
        for i in range(n):
            P[i] = (1.0 - eps) * P[i] + eps * float(noise[i])

    # ------------------------------------------------------------------ #
    # One simulation across a pool: collect leaves, then apply results
    # ------------------------------------------------------------------ #
    def _collect(self, pool: List[_Game]):
        """Descend once per active game and build the network request.

        Terminal leaves are resolved exactly here and never reach the network.
        Returns ``None`` when every descent ended in a terminal position.
        """
        pending = []
        leaves = []
        for g in pool:
            if g.sims_left <= 0:
                continue
            node, board, path, rep = self._descend(g)

            if node.terminal_value is not None:
                self._backup(path, node.terminal_value)
                g.sims_left -= 1
                continue

            legal = list(board.legal_moves)
            terminal = self._resolve_terminal(board, legal, rep)
            if terminal is not None:
                node.terminal_value = terminal
                self._backup(path, terminal)
                g.sims_left -= 1
                continue

            pending.append((g, node, path, legal))
            leaves.append((board, rep))

        if not pending:
            return None

        n = len(pending)
        max_legal = max(len(p[3]) for p in pending)
        states = np.empty((n, NUM_PLANES, 8, 8), dtype=np.float32)
        idx = np.zeros((n, max_legal), dtype=np.int32)
        counts = np.empty(n, dtype=np.int32)
        move_indices: List[List[int]] = []

        for k, (board, rep) in enumerate(leaves):
            states[k] = encode_board(board, rep)
            ii = [move_to_index(m, board) for m in pending[k][3]]
            move_indices.append(ii)
            idx[k, : len(ii)] = ii
            counts[k] = len(ii)

        return pending, move_indices, states, idx, counts

    def _apply(self, pending, move_indices, priors, values) -> None:
        """Expand each evaluated leaf and back its value up to the root."""
        for k, (g, node, path, legal) in enumerate(pending):
            node.expand(legal, move_indices[k], priors[k, : len(legal)].tolist())
            if node is g.root and g.noise_pending:
                self._add_root_noise(node)
                g.noise_pending = False
            self._backup(path, float(values[k]))
            g.sims_left -= 1

    def _simulate(self, pool: List[_Game]) -> int:
        """Advance every game in ``pool`` by one simulation, synchronously."""
        collected = self._collect(pool)
        if collected is None:
            return 0
        pending, move_indices, states, idx, counts = collected
        priors, values = self.evaluator.evaluate(states, idx, counts)
        self._apply(pending, move_indices, priors, values)
        return len(pending)

    # ------------------------------------------------------------------ #
    # Playing a move
    # ------------------------------------------------------------------ #
    def _record_example(self, game: _Game) -> None:
        """Store the root visit distribution as this ply's policy target."""
        root = game.root
        pairs = [
            (root.move_idx[i], root.N[i])
            for i in range(len(root.N))
            if root.N[i] > 0
        ]
        if not pairs:
            # Degenerate (no simulation reached a child): fall back to uniform
            # over the legal moves the root already knows about.
            pairs = [(mi, 1) for mi in root.move_idx]
        if len(pairs) > MAX_POLICY_TARGETS:
            pairs.sort(key=lambda kv: kv[1], reverse=True)
            pairs = pairs[:MAX_POLICY_TARGETS]

        total = float(sum(n for _, n in pairs))
        k = len(pairs)
        out_idx = np.zeros(MAX_POLICY_TARGETS, dtype=np.uint16)
        out_val = np.zeros(MAX_POLICY_TARGETS, dtype=np.float32)
        for j, (mi, n) in enumerate(pairs):
            out_idx[j] = mi
            out_val[j] = n / total

        game.states.append(pack_state(encode_board(game.board, game.cur_rep)))
        game.pol_idx.append(out_idx)
        game.pol_val.append(out_val)
        game.turns.append(game.board.turn)

    def _choose_move(self, game: _Game) -> int:
        """Pick a child slot from the root visit counts."""
        root = game.root
        N = root.N
        if game.move_count < self.temperature_moves and root.n_total > 0:
            # Temperature 1: sample proportional to visit count.
            r = self._py_rng.random() * root.n_total
            acc = 0
            for i in range(len(N)):
                acc += N[i]
                if r <= acc:
                    return i
            return len(N) - 1
        best_i, best_n = 0, -1
        for i in range(len(N)):
            if N[i] > best_n:
                best_n, best_i = N[i], i
        return best_i

    def _should_resign(self, game: _Game) -> bool:
        """Update the resign streak; True once the mover is hopelessly lost."""
        if self.resign_threshold is None:
            return False
        root = game.root
        best_q = -1.0
        seen = False
        for i in range(len(root.N)):
            if root.N[i] > 0:
                seen = True
                if root.Q[i] > best_q:
                    best_q = root.Q[i]
        if not seen:
            game.resign_streak = 0
            return False

        if best_q <= self.resign_threshold:
            game.resign_streak += 1
        else:
            game.resign_streak = 0

        if game.resign_streak < self.resign_plies:
            return False
        if game.would_resign_at is None:
            game.would_resign_at = game.move_count
            game.would_resign_side = game.board.turn
        return game.allow_resign

    def _advance(self, game: _Game) -> Optional[float]:
        """Play one move in ``game``; return the White-perspective result if over.

        Records the training target for the position being left, then pushes the
        chosen move, carries the corresponding subtree into the next ply and
        re-checks for termination.
        """
        self._record_example(game)

        if self._should_resign(game):
            game.resigned = True
            # The side to move gives up, so the opponent wins.
            return -1.0 if game.board.turn == chess.WHITE else 1.0

        i = self._choose_move(game)
        move = game.root.moves[i]
        child = game.root.children[i]

        game.board.push(move)
        game.move_count += 1

        key = game.board._transposition_key()
        seen = game.rep.get(key, 0)
        game.rep[key] = seen + 1
        game.cur_rep = seen

        # Subtree reuse: the child of the played move already holds the visits
        # spent below it, so they are not thrown away.
        game.root = child if child is not None else Node()
        game.noise_pending = True

        legal = list(game.board.legal_moves)
        terminal = self._resolve_terminal(game.board, legal, game.cur_rep)
        if terminal is not None:
            if terminal == -1.0:  # side to move is checkmated
                return -1.0 if game.board.turn == chess.WHITE else 1.0
            return 0.0
        if game.move_count >= self.max_moves:
            return 0.0

        # Seed the next search, crediting any reused visits towards the budget.
        game.sims_left = max(1, self.num_simulations - game.root.n_total)
        if game.root.is_expanded and game.noise_pending:
            self._add_root_noise(game.root)
            game.noise_pending = False
        return None

    @staticmethod
    def _harvest(game: _Game, result: float, out: dict) -> None:
        """Append a finished game's examples, labelled with its result."""
        for state, pi, pv, turn in zip(
            game.states, game.pol_idx, game.pol_val, game.turns
        ):
            out["states"].append(state)
            out["pol_idx"].append(pi)
            out["pol_val"].append(pv)
            out["values"].append(result if turn == chess.WHITE else -result)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self, num_games: int) -> SelfPlayBatch:
        """Play ``num_games`` games and return their training examples.

        The pool is split across ``self.stages`` pipeline stages.  Each pass
        waits on one stage's outstanding request, applies it and issues the
        next, so the remaining stages' requests stay in flight the whole time
        this stage's tree search is running -- which is what keeps the cores and
        the GPU busy at the same time instead of alternating between them.
        """
        out = {"states": [], "pol_idx": [], "pol_val": [], "values": []}
        stats = {
            "games": 0.0, "plies": 0.0, "evals": 0.0,
            "resigned": 0.0, "resign_checked": 0.0, "resign_false_pos": 0.0,
        }

        self._started = 0
        per_stage = max(1, self.games_in_flight // self.stages)
        pools: List[List[_Game]] = []
        for _ in range(self.stages):
            pool: List[_Game] = []
            while len(pool) < per_stage and self._started < num_games:
                pool.append(self._new_game())
                self._started += 1
            pools.append(pool)

        pending: List[Optional[tuple]] = [None] * self.stages
        for k, pool in enumerate(pools):
            if pool:
                pending[k] = self._submit(k, pool)

        while any(pools):
            for k, pool in enumerate(pools):
                if pending[k] is not None:
                    collected = pending[k]
                    pending[k] = None
                    priors, values = self.evaluator.wait(k)
                    self._apply(collected[0], collected[1], priors, values)
                    stats["evals"] += len(collected[0])
                if pool:
                    self._advance_pool(pool, num_games, out, stats)
                if pool:
                    pending[k] = self._submit(k, pool)

        return self._to_batch(out, stats)

    def _submit(self, slot: int, pool: List[_Game]):
        """Descend ``pool`` once and hand its leaves to the evaluator."""
        collected = self._collect(pool)
        if collected is None:
            return None
        pending, move_indices, states, idx, counts = collected
        self.evaluator.submit(slot, states, idx, counts)
        return pending, move_indices

    def _advance_pool(self, pool, num_games, out, stats) -> None:
        """Play a move for every game that has used up its simulation budget."""
        finished = []
        for slot, g in enumerate(pool):
            if g.sims_left > 0:
                continue
            result = self._advance(g)
            if result is None:
                continue

            self._harvest(g, result, out)
            stats["games"] += 1
            # Count recorded positions rather than g.move_count: a resigning
            # game records the position it gives up in without pushing a move,
            # so move_count would be one short of the examples it produced.
            stats["plies"] += len(g.states)
            if g.resigned:
                stats["resigned"] += 1
            if not g.allow_resign and g.would_resign_at is not None:
                # Played out despite meeting the resign criterion: check whether
                # resigning would have thrown away a draw or a win.
                stats["resign_checked"] += 1
                loser = g.would_resign_side
                lost = (result < 0) if loser == chess.WHITE else (result > 0)
                if not lost:
                    stats["resign_false_pos"] += 1
            finished.append(slot)

        if not finished:
            return
        # Continuous refill: keep the batch at full width rather than letting it
        # decay to a few stragglers.
        for slot in finished:
            if self._started < num_games:
                pool[slot] = self._new_game()
                self._started += 1
            else:
                pool[slot] = None
        pool[:] = [g for g in pool if g is not None]

    def _new_game(self) -> _Game:
        allow_resign = (
            self.resign_threshold is not None
            and self._py_rng.random() >= self.resign_disable_fraction
        )
        game = _Game(allow_resign)
        game.sims_left = self.num_simulations
        return game

    @staticmethod
    def _to_batch(out: dict, stats: dict) -> SelfPlayBatch:
        if not out["states"]:
            batch = SelfPlayBatch.empty()
            batch.stats = stats
            return batch
        pol_idx = np.stack(out["pol_idx"])
        pol_val = np.stack(out["pol_val"])
        return SelfPlayBatch(
            np.stack(out["states"]),
            pol_idx,
            pol_val,
            (pol_val > 0).sum(axis=1).astype(np.int16),
            np.asarray(out["values"], dtype=np.float32),
            stats,
        )


# --------------------------------------------------------------------------- #
# GPU evaluator
# --------------------------------------------------------------------------- #
class TorchEvaluator:
    """Batched network evaluation returning priors over legal moves only.

    The softmax is restricted to each position's legal moves *on the device*:
    only the ``(B, max_legal)`` gathered rows come back to the host, instead of
    the full ``(B, 4672)`` logit matrix.  For a 3072-position batch that is
    ~0.7 MB rather than ~57 MB per call, and it also removes the per-position
    NumPy softmax loop from the caller.
    """

    def __init__(self, model, device, use_amp: bool = True) -> None:
        import torch

        self.torch = torch
        self.model = model
        self.device = device
        self.use_amp = bool(use_amp) and device.type == "cuda"
        self.channels_last = device.type == "cuda"
        model.eval()

    def __call__(self, states: np.ndarray, idx: np.ndarray, counts: np.ndarray):
        torch = self.torch
        n = states.shape[0]
        x = torch.from_numpy(states).to(self.device, non_blocking=True)
        padded = _bucket_batch(n, max(n, _BATCH_QUANTUM))
        if padded > n:
            # See _bucket_batch: keeps cuDNN from re-tuning on every new shape.
            x = torch.cat([x, x[-1:].expand(padded - n, *x.shape[1:])])
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        idx_t = torch.from_numpy(idx).to(self.device, non_blocking=True).long()
        cnt_t = torch.from_numpy(counts).to(self.device, non_blocking=True).long()

        with torch.no_grad():
            with torch.autocast("cuda", torch.float16, enabled=self.use_amp):
                logits, value = self.model(x)
            gathered = logits[:n].gather(1, idx_t).float()
            mask = (
                torch.arange(idx.shape[1], device=self.device).unsqueeze(0)
                >= cnt_t.unsqueeze(1)
            )
            gathered.masked_fill_(mask, float("-inf"))
            priors = torch.softmax(gathered, dim=1)
            values = value[:n].float().view(-1)
            return priors.cpu().numpy(), values.cpu().numpy()


# --------------------------------------------------------------------------- #
# Multi-process pipeline
# --------------------------------------------------------------------------- #
# The theoretical maximum number of legal moves in a chess position, used to
# size the fixed-width shared-memory request slots.
_MAX_LEGAL = 218

# Shared-memory buffer descriptors, built once per run.
_BUFFERS = ("states", "idx", "counts", "priors", "values")

# Also the granularity of CUDA-graph capture, so a phase only ever captures a
# handful of shapes.
#
# Inference batches vary in size from pass to pass, but ``cudnn.benchmark``
# (which the trainer wants on, for its fixed-size training batches) re-runs its
# algorithm search for every previously unseen input shape. Measured on an RTX
# 3090 with a 128x10 net: 7.4k positions/s at freely varying batch sizes against
# 80k when they are snapped to a small set. Rounding the batch up to a multiple
# of this leaves cuDNN a handful of shapes to cache, at the cost of a few
# percent of padded rows.
_BATCH_QUANTUM = 128


def _bucket_batch(n: int, cap: int) -> int:
    """Round ``n`` up to the next batch shape the network is warmed up for."""
    return min(cap, ((n + _BATCH_QUANTUM - 1) // _BATCH_QUANTUM) * _BATCH_QUANTUM)


# Pipeline stages per worker. A worker splits its games into this many
# sub-pools and keeps one request per sub-pool outstanding, so it can keep
# searching while earlier requests are in flight. Two stages only cover a round
# trip as long as one stage's tree search; the round trip is typically longer
# than that, which leaves workers blocked and starves the server of the large
# batches it needs (the forward pass is ~87% of a pass, and it runs at 56k
# positions/s at batch 512 against 80k at batch 1536).
DEFAULT_PIPELINE_STAGES = 4


def resolve_stages(games_in_flight: int, stages: Optional[int] = None) -> int:
    """Clamp the requested stage count to what ``games_in_flight`` supports."""
    requested = DEFAULT_PIPELINE_STAGES if stages is None else int(stages)
    return max(1, min(requested, int(games_in_flight)))


def _buffer_specs(num_slots: int, rows: int):
    """Shapes/dtypes of the shared-memory request+response slots.

    There is one slot per worker *per pipeline stage*, so a worker can have its
    next request in flight while it searches for the other stage.
    """
    return {
        "states": ((num_slots, rows, NUM_PLANES, 8, 8), np.float32),
        "idx": ((num_slots, rows, _MAX_LEGAL), np.int32),
        "counts": ((num_slots, rows), np.int32),
        "priors": ((num_slots, rows, _MAX_LEGAL), np.float32),
        "values": ((num_slots, rows), np.float32),
    }


def _attach_buffers(info):
    """Attach to shared-memory blocks described by ``info``; return handles."""
    from multiprocessing import shared_memory

    shms, arrays = [], {}
    for key, (name, shape, dtype) in info.items():
        shm = shared_memory.SharedMemory(name=name)
        shms.append(shm)
        arrays[key] = np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf)
    return shms, arrays


class _IpcEvaluator:
    """Worker-side evaluator: shared-memory slots plus a queue/pipe round trip.

    ``submit`` never blocks, so the engine can issue one stage's request and
    keep searching the other stage while the server works.
    """

    def __init__(self, base_slot, stages, arrays, req_q, resp_conn):
        self.base = base_slot
        self.stages = stages
        self.states = arrays["states"]
        self.idx = arrays["idx"]
        self.counts = arrays["counts"]
        self.priors = arrays["priors"]
        self.values = arrays["values"]
        self.req_q = req_q
        self.conn = resp_conn
        self._shape = [(0, 0)] * stages
        self._ready = set()
        # How long this worker sat blocked on the server, for diagnostics.
        self.wait_time = 0.0

    def submit(self, slot, states, idx, counts):
        s = self.base + slot
        n, width = idx.shape
        self.states[s, :n] = states
        self.idx[s, :n, :width] = idx
        self.counts[s, :n] = counts
        self._shape[slot] = (n, width)
        self.req_q.put((s, n, width))

    def wait(self, slot):
        s = self.base + slot
        blocked_at = time.time()
        # Stages can complete out of order, so keep any other slot's wakeup.
        while s not in self._ready:
            self._ready.add(self.conn.recv())
        self.wait_time += time.time() - blocked_at
        self._ready.discard(s)
        n, width = self._shape[slot]
        # Views, not copies: the server only rewrites this slot in response to
        # our next submit, and the caller consumes the rows immediately.
        return self.priors[s, :n, :width], self.values[s, :n]

    def evaluate(self, states, idx, counts):
        """Synchronous convenience path (used only by ``_simulate``)."""
        self.submit(0, states, idx, counts)
        return self.wait(0)


def _worker_main(wid, info, cfg, stages, num_games, seed, req_q, resp_conn,
                 out_path):
    """Self-play worker: pure-Python tree search, no GPU, no torch threads.

    Positions are handed to the parent's inference server through this worker's
    shared-memory slots; the round trip is one small queue message plus one pipe
    wakeup, a fraction of a percent of the millisecond-scale tree work that
    produced the batch.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:  # pragma: no cover - torch is optional in a worker
        pass

    shms, arrays = _attach_buffers(info)
    evaluator = _IpcEvaluator(wid * stages, stages, arrays, req_q, resp_conn)

    worker_start = time.time()
    try:
        engine = SelfPlayEngine(evaluator, seed=seed, **cfg)
        batch = engine.run(num_games)
    finally:
        req_q.put((-1, wid, 0))
    batch.stats["worker_seconds"] = time.time() - worker_start
    batch.stats["worker_blocked_seconds"] = evaluator.wait_time

    np.savez(
        out_path,
        states=batch.states,
        pol_idx=batch.pol_idx,
        pol_val=batch.pol_val,
        pol_len=batch.pol_len,
        values=batch.values,
        stats_keys=np.array(list(batch.stats.keys())),
        stats_vals=np.array(list(batch.stats.values()), dtype=np.float64),
    )
    for shm in shms:
        shm.close()


class _GraphedNetwork:
    """Replays a captured CUDA graph of the forward pass, one per batch shape.

    Self-play inference is *host-launch bound*, not GPU bound: a 128x10 net at
    batch 384 needs ~3.2ms of host time to issue ~3.8ms of device work, and the
    server thread is competing with the search workers for cores.  Capturing
    the forward pass into a CUDA graph drops the issue cost to ~0.03ms, so the
    server can keep the device fed.

    Graphs are captured fresh for each self-play phase and dropped at the end of
    it.  Weights change between phases, and a graph captured under autocast
    would otherwise be free to bake in a stale fp16 copy of them; re-capturing
    removes that class of bug entirely.  Capture is lazy, so only the shapes a
    phase actually sees are built, and they share one memory pool.
    """

    def __init__(self, torch, model, device, use_amp, channels_last):
        self.torch = torch
        self.model = model
        self.device = device
        self.use_amp = use_amp
        self.channels_last = channels_last
        self._graphs = {}
        self._pool = None
        self._disabled = device.type != "cuda"

    def __call__(self, x):
        """Run the network on ``x``; returns ``(logits, value)``.

        The returned tensors are the graph's static outputs, valid until the
        next replay.  Callers consume them on the same stream, so ordinary
        stream ordering keeps that safe.
        """
        torch = self.torch
        if not self._disabled:
            size = x.shape[0]
            entry = self._graphs.get(size)
            if entry is None and size not in self._graphs:
                entry = self._capture(size)
            if entry is not None:
                static_in, logits, value, graph = entry
                static_in.copy_(x)
                graph.replay()
                return logits, value
        with torch.autocast("cuda", torch.float16, enabled=self.use_amp):
            return self.model(x)

    def _capture(self, size):
        """Capture the forward pass at ``size``; ``None`` if capture fails."""
        torch = self.torch
        try:
            static_in = torch.zeros(
                (size, NUM_PLANES, 8, 8), dtype=torch.float32, device=self.device
            )
            if self.channels_last:
                static_in = static_in.contiguous(
                    memory_format=torch.channels_last
                )

            # Warm up on a side stream first; capture of an uninitialised
            # cuDNN/cuBLAS workspace is not legal.
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                with torch.no_grad():
                    with torch.autocast(
                        "cuda", torch.float16,
                        enabled=self.use_amp, cache_enabled=False,
                    ):
                        for _ in range(3):
                            self.model(static_in)
            torch.cuda.current_stream().wait_stream(side)

            graph = torch.cuda.CUDAGraph()
            with torch.no_grad():
                with torch.cuda.graph(graph, pool=self._pool):
                    # cache_enabled=False keeps autocast from baking a cached
                    # fp16 copy of the weights into the graph.
                    with torch.autocast(
                        "cuda", torch.float16,
                        enabled=self.use_amp, cache_enabled=False,
                    ):
                        logits, value = self.model(static_in)
            if self._pool is None:
                self._pool = graph.pool()
            entry = (static_in, logits, value, graph)
            self._graphs[size] = entry
            return entry
        except Exception:
            # Any capture failure (driver, memory, an uncapturable op) falls
            # back to eager for this shape and every later one.
            self._graphs[size] = None
            self._disabled = True
            return None

    def close(self) -> None:
        """Release captured graphs and their memory pool."""
        self._graphs.clear()
        self._pool = None
        try:
            self.torch.cuda.synchronize()
            self.torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - best effort
            pass


class _PassBuffers:
    """One slot of the server's double buffer: pinned host staging + an event."""

    __slots__ = ("stage", "stage_np", "idx", "idx_np", "cnt", "cnt_np",
                 "host", "host_np", "event")

    def __init__(self, torch, cap: int) -> None:
        # Zeroed, not empty: rows past the real batch are still fed to the
        # network as padding, so they must hold valid floats and in-range
        # gather indices.
        self.stage = torch.zeros(
            (cap, NUM_PLANES, 8, 8), dtype=torch.float32
        ).pin_memory()
        self.stage_np = self.stage.numpy()
        self.idx = torch.zeros((cap, _MAX_LEGAL), dtype=torch.int32).pin_memory()
        self.idx_np = self.idx.numpy()
        self.cnt = torch.zeros((cap,), dtype=torch.int32).pin_memory()
        self.cnt_np = self.cnt.numpy()
        self.host = torch.zeros(
            (cap, _MAX_LEGAL + 1), dtype=torch.float32
        ).pin_memory()
        self.host_np = self.host.numpy()
        self.event = torch.cuda.Event()


class _InferenceServer:
    """Owns the GPU and answers every worker's evaluation request.

    Each pass drains all requests currently queued, so batches grow with the
    number of workers waiting: the forward pass is ~87% of a pass's cost and
    runs far more efficiently wide (measured on an RTX 3090 with a 128x10 net:
    56k positions/s at batch 512, 73k at 1024, 80k at 1536).

    Work is double-buffered.  :meth:`launch` stages a batch and starts the
    forward pass without blocking; :meth:`complete` then finishes the
    *previous* batch.  So while the GPU runs pass N the server is staging pass
    N+1 and scattering pass N-1, instead of leaving the device idle for every
    host-side copy and wakeup -- which had been costing ~25% of GPU time.
    """

    def __init__(self, model, device, num_slots, rows, arrays, use_amp=True):
        import torch

        self.torch = torch
        self.model = model
        self.device = device
        self.arrays = arrays
        self.use_amp = bool(use_amp) and device.type == "cuda"
        self.channels_last = device.type == "cuda"
        self._cap = num_slots * rows
        self._bufs = [_PassBuffers(torch, self._cap) for _ in range(2)]
        self._cur = 0
        self._stream = torch.cuda.Stream() if device.type == "cuda" else None
        self._net = _GraphedNetwork(
            torch, model, device, self.use_amp, self.channels_last
        )
        # Reused across passes; allocating this per call showed up in profiles.
        self._arange = torch.arange(_MAX_LEGAL, device=device)
        self.batches = 0
        self.positions = 0
        self.gpu_time = 0.0
        # Where a pass's wall time goes, for the verbose progress line.
        self.stage_time = 0.0
        self.sync_time = 0.0
        self.scatter_time = 0.0
        self._launched_at = None

    def launch(self, requests):
        """Stage ``requests`` and start their forward pass; never blocks."""
        torch = self.torch
        launch_start = time.time()
        buf = self._bufs[self._cur]
        src_states = self.arrays["states"]
        src_idx = self.arrays["idx"]
        src_cnt = self.arrays["counts"]

        offset = 0
        width = 0
        for slot, n, w in requests:
            buf.stage_np[offset:offset + n] = src_states[slot, :n]
            buf.idx_np[offset:offset + n, :w] = src_idx[slot, :n, :w]
            buf.cnt_np[offset:offset + n] = src_cnt[slot, :n]
            offset += n
            width = max(width, w)

        # Requests are padded out to the widest legal-move count in this batch.
        # ``gather`` reads every column regardless of the mask applied after it,
        # so the padding must hold a valid index, not a stale one.
        pos = 0
        for _slot, n, w in requests:
            if w < width:
                buf.idx_np[pos:pos + n, w:width] = 0
            pos += n

        staged_at = time.time()
        self.stage_time += staged_at - launch_start
        self._launched_at = staged_at
        padded = _bucket_batch(offset, self._cap)
        with torch.cuda.stream(self._stream):
            x = buf.stage[:padded].to(self.device, non_blocking=True)
            if self.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            idx_t = buf.idx[:offset, :width].to(
                self.device, non_blocking=True
            ).long()
            cnt_t = buf.cnt[:offset].to(self.device, non_blocking=True).long()

            with torch.no_grad():
                logits, value = self._net(x)
                # Padding rows exist only to keep the shape cached; drop them
                # before the policy work. (Eval-mode BatchNorm uses running
                # statistics, so they cannot influence the real rows.)
                logits = logits[:offset]
                # Gather in the network's own dtype and only then widen:
                # upcasting the full (B, 4672) logit matrix first would allocate
                # and copy megabytes per pass to read ~60 entries per row.
                gathered = logits.gather(1, idx_t).float()
                mask = self._arange[:width].unsqueeze(0) >= cnt_t.unsqueeze(1)
                gathered.masked_fill_(mask, float("-inf"))
                merged = torch.cat(
                    [torch.softmax(gathered, dim=1),
                     value[:offset].float().view(-1, 1)],
                    dim=1,
                )
                buf.host[:offset, : width + 1].copy_(merged, non_blocking=True)
            buf.event.record(self._stream)

        self._cur ^= 1
        self.batches += 1
        self.positions += offset
        return (requests, buf, offset, width)

    def close(self) -> None:
        """Release the captured graphs once the self-play phase is done."""
        self._net.close()

    def complete(self, ticket, conns, stages) -> None:
        """Wait for a launched pass, write results back and wake its workers."""
        requests, buf, offset, width = ticket
        sync_start = time.time()
        buf.event.synchronize()
        synced_at = time.time()
        self.sync_time += synced_at - sync_start
        if self._launched_at is not None:
            self.gpu_time += synced_at - self._launched_at
            self._launched_at = None

        out = buf.host_np
        dst_p, dst_v = self.arrays["priors"], self.arrays["values"]
        pos = 0
        for slot, n, w in requests:
            dst_p[slot, :n, :w] = out[pos:pos + n, :w]
            dst_v[slot, :n] = out[pos:pos + n, width]
            pos += n
        for slot, _n, _w in requests:
            conns[slot // stages].send(slot)
        self.scatter_time += time.time() - synced_at


def default_worker_count() -> int:
    """Workers to use when none is given.

    Measured on an 8-core/16-thread i9 with an RTX 3090, self-play throughput
    peaks at ~12 workers: fewer leaves cores idle while workers wait on the
    inference round trip, more just adds blocking and starves the server thread
    (which needs a core of its own). Three quarters of the logical CPUs lands on
    that peak and degrades sensibly on other machines.
    """
    cpus = os.cpu_count() or 4
    return max(1, min(cpus - 2, int(cpus * 0.75), 24))


def _run_parallel(
    model,
    device,
    num_games: int,
    num_workers: int,
    games_in_flight: int,
    cfg: dict,
    seed: Optional[int],
    scratch_dir: Optional[str],
    verbose: bool,
    use_amp: bool = True,
) -> SelfPlayBatch:
    """Spawn workers behind one GPU inference server and collect their games."""
    import multiprocessing as mp
    import queue as queue_mod
    import shutil
    import tempfile
    from multiprocessing import shared_memory

    ctx = mp.get_context("spawn")
    stages = resolve_stages(games_in_flight, cfg.pop("pipeline_stages", None))
    rows = max(1, games_in_flight // stages)
    num_slots = num_workers * stages
    specs = _buffer_specs(num_slots, rows)

    shms = {}
    info = {}
    arrays = {}
    workdir = tempfile.mkdtemp(prefix="alphachess_sp_", dir=scratch_dir)
    procs, conns = [], []
    try:
        for key in _BUFFERS:
            shape, dtype = specs[key]
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            shm = shared_memory.SharedMemory(create=True, size=nbytes)
            shms[key] = shm
            arrays[key] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            info[key] = (shm.name, shape, np.dtype(dtype).str)

        req_q = ctx.Queue()
        # Split the game quota; the remainder goes to the lowest-numbered
        # workers so every worker gets at least one game.
        base, extra = divmod(num_games, num_workers)
        out_paths = []
        for wid in range(num_workers):
            share = base + (1 if wid < extra else 0)
            out_path = os.path.join(workdir, "w{0}.npz".format(wid))
            out_paths.append(out_path)
            # Pipe(duplex=False) yields (read_end, write_end): the worker
            # blocks on the read end, the server wakes it via the write end.
            child_conn, parent_conn = ctx.Pipe(duplex=False)
            conns.append(parent_conn)
            worker_cfg = dict(cfg)
            worker_cfg["games_in_flight"] = min(games_in_flight, max(share, 1))
            worker_cfg["pipeline_stages"] = stages
            proc = ctx.Process(
                target=_worker_main,
                args=(
                    wid, info, worker_cfg, stages, share,
                    None if seed is None else seed * 1000 + wid,
                    req_q, child_conn, out_path,
                ),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            procs.append(proc)

        server = _InferenceServer(
            model, device, num_slots, rows, arrays, use_amp=use_amp,
        )

        active = set(range(num_workers))
        started_at = time.time()
        next_report = started_at + 30.0
        in_flight = None  # a launched pass whose results are not yet scattered
        queue_time = 0.0

        while active or in_flight is not None:
            if verbose and time.time() >= next_report:
                now = time.time()
                elapsed = now - started_at
                print(
                    "  [self-play {e:.0f}s] {p:,.0f} positions/s | mean batch "
                    "{m:.0f} | GPU {g:.0f}% | stage {st:.0f}% sync {sy:.0f}% "
                    "scatter {sc:.0f}% queue {q:.0f}% | {w} workers".format(
                        e=elapsed, p=server.positions / elapsed,
                        m=server.positions / max(server.batches, 1),
                        g=server.gpu_time / elapsed * 100.0,
                        st=server.stage_time / elapsed * 100.0,
                        sy=server.sync_time / elapsed * 100.0,
                        sc=server.scatter_time / elapsed * 100.0,
                        q=queue_time / elapsed * 100.0,
                        w=len(active),
                    ),
                    flush=True,
                )
                next_report = now + 30.0

            messages = []
            queue_start = time.time()
            if in_flight is None:
                # Nothing on the GPU, so it is safe to wait for work.
                try:
                    messages.append(req_q.get(timeout=5.0))
                except queue_mod.Empty:
                    for wid in list(active):
                        proc = procs[wid]
                        if not proc.is_alive():
                            active.discard(wid)
                            if proc.exitcode not in (0, None):
                                raise RuntimeError(
                                    "self-play worker {w} died with exit code "
                                    "{c}".format(w=wid, c=proc.exitcode)
                                )
                    continue

            # Drain whatever else is queued so one forward pass serves as many
            # workers as possible.
            while True:
                try:
                    messages.append(req_q.get_nowait())
                except queue_mod.Empty:
                    break
            queue_time += time.time() - queue_start

            requests = []
            for slot, n, width in messages:
                if slot < 0:
                    active.discard(n)  # a completion notice carries the worker
                else:
                    requests.append((slot, n, width))

            # Start this batch before finishing the previous one, so the GPU is
            # busy while the host scatters results and wakes workers.
            launched = server.launch(requests) if requests else None
            if in_flight is not None:
                server.complete(in_flight, conns, stages)
            in_flight = launched

        for wid, proc in enumerate(procs):
            proc.join(timeout=120)
            if proc.is_alive():  # pragma: no cover - defensive
                proc.terminate()
                raise RuntimeError(
                    "self-play worker {w} did not exit".format(w=wid)
                )
            if proc.exitcode:
                raise RuntimeError(
                    "self-play worker {w} failed with exit code {c}".format(
                        w=wid, c=proc.exitcode
                    )
                )

        parts = []
        for path in out_paths:
            if not os.path.exists(path):
                continue
            with np.load(path) as data:
                stats = {
                    str(k): float(v)
                    for k, v in zip(data["stats_keys"], data["stats_vals"])
                }
                parts.append(
                    SelfPlayBatch(
                        data["states"], data["pol_idx"], data["pol_val"],
                        data["pol_len"], data["values"], stats,
                    )
                )
        server.close()
        batch = SelfPlayBatch.concat(parts)
        batch.stats["nn_batches"] = float(server.batches)
        batch.stats["nn_positions"] = float(server.positions)
        batch.stats["gpu_seconds"] = float(server.gpu_time)
        if verbose and server.batches:
            print(
                "  inference: {b:,} passes, mean batch {m:.0f} positions, "
                "{g:.1f}s on GPU".format(
                    b=server.batches,
                    m=server.positions / server.batches,
                    g=server.gpu_time,
                )
            )
        return batch
    finally:
        for conn in conns:
            try:
                conn.close()
            except OSError:
                pass
        for shm in shms.values():
            try:
                shm.close()
                shm.unlink()
            except (OSError, FileNotFoundError):  # pragma: no cover
                pass
        shutil.rmtree(workdir, ignore_errors=True)


def generate_selfplay_data(
    model,
    device=None,
    num_games: int = 1,
    num_workers: Optional[int] = None,
    games_in_flight: int = 256,
    use_amp: bool = True,
    seed: Optional[int] = None,
    scratch_dir: Optional[str] = None,
    verbose: bool = True,
    **cfg,
) -> SelfPlayBatch:
    """Generate ``num_games`` self-play games and return them as a batch.

    ``num_workers`` search processes feed one GPU inference server; each keeps
    ``games_in_flight`` games in flight, so the network sees batches of up to
    ``num_workers * games_in_flight`` positions.  ``num_workers=1`` runs
    everything in this process (useful for tests and for CPU-only machines).

    Remaining keyword arguments (``num_simulations``, ``c_puct``,
    ``temperature_moves``, ``max_moves``, ``resign_threshold``, ...) are passed
    through to :class:`SelfPlayEngine`.
    """
    from alpha_chess.network import get_device

    if device is None:
        device = get_device()
    if num_workers is None:
        num_workers = default_worker_count() if device.type == "cuda" else 1
    num_workers = max(1, min(int(num_workers), max(1, num_games)))

    if num_workers == 1:
        engine = SelfPlayEngine(
            TorchEvaluator(model, device, use_amp=use_amp),
            games_in_flight=min(games_in_flight, num_games),
            seed=seed,
            **cfg,
        )
        return engine.run(num_games)

    return _run_parallel(
        model, device, num_games, num_workers, games_in_flight,
        dict(cfg), seed, scratch_dir, verbose, use_amp,
    )
