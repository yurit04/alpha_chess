from __future__ import annotations

"""Self-play data generation and an experience replay buffer.

This module drives the network + MCTS to play games against itself and turns
each visited position into a training example of the form
``{"state", "policy", "value"}``.  It also provides a fixed-capacity
:class:`ReplayBuffer` used by the training loop.
"""

import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import chess

from alpha_chess.encoding import encode_board, move_to_index, POLICY_SIZE
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
    """Fixed-capacity ring buffer of self-play training examples."""

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)

    def append(self, examples: List[dict]) -> None:
        """Add a list of ``{"state","policy","value"}`` examples to the buffer."""
        for example in examples:
            self._buffer.append(example)

    def __len__(self) -> int:
        return len(self._buffer)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a minibatch (with replacement).

        Returns ``(states (B,19,8,8), policies (B,POLICY_SIZE), values (B,1))``
        as float32 numpy arrays.
        """
        chosen = [random.choice(self._buffer) for _ in range(batch_size)]

        states = np.stack(
            [np.asarray(ex["state"], dtype=np.float32) for ex in chosen]
        )
        policies = np.stack(
            [np.asarray(ex["policy"], dtype=np.float32) for ex in chosen]
        )
        values = np.asarray(
            [ex["value"] for ex in chosen], dtype=np.float32
        ).reshape(batch_size, 1)

        return states, policies, values
