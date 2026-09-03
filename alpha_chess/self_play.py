from __future__ import annotations

"""Self-play data generation and an experience replay buffer.

This module drives the network + MCTS to play games against itself and turns
each visited position into a training example of the form
``{"state", "policy", "value"}``.  It also provides a fixed-capacity
:class:`ReplayBuffer` used by the training loop.
"""

import random
from typing import Dict, List, Optional

import numpy as np
import chess

from alpha_chess.batched_selfplay import MAX_POLICY_TARGETS
from alpha_chess.encoding import (
    encode_board,
    move_to_index,
    NUM_PLANES,
    POLICY_SIZE,
)
from alpha_chess.mcts import MCTS


def _sample_move(
    policy: Dict[chess.Move, float], rng: random.Random
) -> chess.Move:
    """Sample a move from a visit distribution (temperature 1)."""
    moves = list(policy.keys())
    probs = np.asarray([policy[m] for m in moves], dtype=np.float64)
    total = probs.sum()
    if total <= 0:
        # Degenerate distribution: fall back to a uniform choice.
        idx = rng.randrange(len(moves))
        return moves[idx]
    probs = probs / total
    r = rng.random()
    cumulative = 0.0
    for move, p in zip(moves, probs):
        cumulative += p
        if r <= cumulative:
            return move
    return moves[-1]


def _argmax_move(policy: Dict[chess.Move, float]) -> chess.Move:
    """Return the move with the greatest visit probability."""
    return max(policy.items(), key=lambda kv: kv[1])[0]


def play_game(
    network,
    device=None,
    num_simulations: int = 100,
    c_puct: float = 1.5,
    temperature_moves: int = 30,
    max_moves: int = 400,
    seed: Optional[int] = None,
) -> List[dict]:
    """Play one self-play game and return a list of training examples.

    A single shared :class:`MCTS`/network plays both sides.  For each move the
    MCTS visit distribution becomes the policy target; the game result (from the
    perspective of the player to move at that position) becomes the value target.

    Returns a list of dicts, each ``{"state": np.ndarray(19,8,8) float32,
    "policy": np.ndarray(POLICY_SIZE,) float32, "value": float}``.
    """
    rng = random.Random(seed)
    mcts = MCTS(network, device=device, c_puct=c_puct)

    board = chess.Board()
    examples: List[dict] = []
    move_count = 0

    while not board.is_game_over(claim_draw=True) and move_count < max_moves:
        # MCTS search with root Dirichlet noise for exploration.
        policy = mcts.run(board, num_simulations, add_noise=True)

        # Build the sparse policy target over the full action space.
        target = np.zeros(POLICY_SIZE, dtype=np.float32)
        for move, prob in policy.items():
            target[move_to_index(move, board)] = prob

        examples.append(
            {
                "state": encode_board(board),
                "policy": target,
                "turn": board.turn,
            }
        )

        # Move selection: sample for the opening plies, then play greedily.
        if move_count < temperature_moves:
            chosen = _sample_move(policy, rng)
        else:
            chosen = _argmax_move(policy)

        board.push(chosen)
        move_count += 1

    # Determine the game result from White's perspective.
    if board.is_game_over(claim_draw=True):
        result_str = board.result(claim_draw=True)
        if result_str == "1-0":
            result = 1.0
        elif result_str == "0-1":
            result = -1.0
        else:
            result = 0.0
    else:
        # Hit the max-move cutoff: treat as a draw.
        result = 0.0

    # Assign value targets from each moving player's perspective.
    for example in examples:
        example["value"] = result if example["turn"] == chess.WHITE else -result
        del example["turn"]

    return examples


class ReplayBuffer:
    """Fixed-capacity ring buffer of self-play training examples.

    Storage is column-oriented and deliberately compact, because replay-buffer
    size is one of the levers on final playing strength and a dense buffer runs
    out of RAM long before it runs out of usefulness:

    * states are uint8-packed (1.3 KB per position instead of 5.4 KB, see
      :func:`alpha_chess.encoding.pack_state`);
    * policy targets are sparse -- the at-most-``MAX_POLICY_TARGETS`` moves that
      actually received visits, rather than a 4672-wide float32 row.

    Together that is ~1.9 KB per position against ~21 KB dense, so 500k
    positions cost ~0.9 GB rather than ~11.7 GB.

    Appends are vectorized ring writes and sampling is a single fancy-index
    gather, both O(batch).
    """

    def __init__(self, capacity: int = 100000):
        self.capacity = int(capacity)
        self._states = np.zeros(
            (self.capacity, NUM_PLANES, 8, 8), dtype=np.uint8
        )
        self._pol_idx = np.zeros(
            (self.capacity, MAX_POLICY_TARGETS), dtype=np.uint16
        )
        self._pol_val = np.zeros(
            (self.capacity, MAX_POLICY_TARGETS), dtype=np.float32
        )
        self._values = np.zeros((self.capacity, 1), dtype=np.float32)
        self._cursor = 0
        self._size = 0

    def append(self, batch) -> None:
        """Append a :class:`~alpha_chess.batched_selfplay.SelfPlayBatch`.

        Writes wrap around the end of the storage, overwriting the oldest
        positions once full.  If ``batch`` is larger than the whole buffer only
        its most recent ``capacity`` positions are kept.
        """
        n = len(batch)
        if self.capacity == 0 or n == 0:
            return

        states, pol_idx = batch.states, batch.pol_idx
        pol_val, values = batch.pol_val, batch.values
        if n > self.capacity:
            states = states[n - self.capacity:]
            pol_idx = pol_idx[n - self.capacity:]
            pol_val = pol_val[n - self.capacity:]
            values = values[n - self.capacity:]
            n = self.capacity

        start = self._cursor
        first = min(n, self.capacity - start)
        self._write(start, states[:first], pol_idx[:first],
                    pol_val[:first], values[:first])
        rest = n - first
        if rest:
            self._write(0, states[first:], pol_idx[first:],
                        pol_val[first:], values[first:])

        self._cursor = (start + n) % self.capacity
        self._size = min(self._size + n, self.capacity)

    def _write(self, at, states, pol_idx, pol_val, values) -> None:
        end = at + states.shape[0]
        self._states[at:end] = states
        self._pol_idx[at:end] = pol_idx
        self._pol_val[at:end] = pol_val
        self._values[at:end, 0] = values

    def __len__(self) -> int:
        return self._size

    def sample(self, batch_size: int):
        """Sample a minibatch uniformly with replacement.

        Returns ``(states_uint8 (B, NUM_PLANES, 8, 8), pol_idx (B, K) int64,
        pol_val (B, K) float32, values (B, 1) float32)``.  States stay packed
        and policies stay sparse; both are expanded on the GPU by the trainer,
        which keeps the host->device transfer ~11x smaller.
        """
        if self._size == 0:
            raise ValueError("cannot sample from an empty ReplayBuffer")
        indices = np.random.randint(0, self._size, size=batch_size)
        return (
            self._states[indices],
            self._pol_idx[indices].astype(np.int64),
            self._pol_val[indices],
            self._values[indices],
        )

    def state_dict(self) -> dict:
        """Serializable snapshot of the buffer's live contents."""
        return {
            "capacity": self.capacity,
            "cursor": self._cursor,
            "size": self._size,
            "states": self._states[:self._size],
            "pol_idx": self._pol_idx[:self._size],
            "pol_val": self._pol_val[:self._size],
            "values": self._values[:self._size],
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore a snapshot written by :meth:`state_dict`.

        A buffer saved at a different capacity is truncated to the most recent
        positions that fit, so ``--buffer-size`` can be changed across resumes.
        """
        size = int(state.get("size", 0))
        if size == 0:
            self._cursor, self._size = 0, 0
            return
        keep = min(size, self.capacity)
        self._states[:keep] = state["states"][size - keep:]
        self._pol_idx[:keep] = state["pol_idx"][size - keep:]
        self._pol_val[:keep] = state["pol_val"][size - keep:]
        self._values[:keep] = state["values"][size - keep:]
        self._size = keep
        self._cursor = keep % self.capacity
