from __future__ import annotations

"""High-level agent wrapping a trained network and MCTS for move suggestion and play.

This module exposes :class:`AlphaChessAgent`, which loads an ``AlphaZeroNet``
checkpoint and drives a PUCT MCTS search to suggest or play moves, plus a
convenience :func:`suggest` function that operates directly on a FEN string.
"""

from typing import Dict, List, Optional, Tuple

import chess
import torch

from alpha_chess.encoding import encode_board, move_to_index
from alpha_chess.mcts import MCTS
from alpha_chess.network import get_device, load_model


class AlphaChessAgent:
    """Wraps a trained network plus MCTS to suggest and play chess moves."""

    def __init__(
        self,
        model_path: str,
        device=None,
        simulations: int = 200,
        c_puct: float = 1.5,
    ) -> None:
        """Load the model and prepare an MCTS searcher.

        Args:
            model_path: Path to a checkpoint saved by ``network.save_model``.
            device: Torch device (or None to auto-select via ``get_device``).
            simulations: Default number of MCTS simulations per search.
            c_puct: PUCT exploration constant passed to the MCTS.
        """
        self.device = device if device is not None else get_device()
        self.model = load_model(model_path, device=self.device)
        self.simulations = simulations
        self.c_puct = c_puct
        self.mcts = MCTS(self.model, device=self.device, c_puct=c_puct)

    def _root_value(self, board: chess.Board) -> float:
        """Return the network's value estimate for ``board`` from the side-to-move
        perspective, using a single (non-search) network evaluation.

        The passed-in board is never mutated.
        """
        state = encode_board(board)  # (19, 8, 8) float32
        tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)  # (1, 19, 8, 8)
        self.model.eval()
        with torch.no_grad():
            _policy_logits, value = self.model(tensor)
        return float(value.item())

    def suggest_move(
        self,
        board: chess.Board,
        simulations: Optional[int] = None,
    ) -> dict:
        """Suggest a move for the current position.

        Runs MCTS (without Dirichlet noise) and returns a dict describing the
        best move, its evaluation, and the top candidate moves. If the game is
        already over, returns a dict with ``move`` set to None.

        Args:
            board: The position to analyze. It is not mutated.
            simulations: Optional override for the number of MCTS simulations.

        Returns:
            A dict with keys ``move``, ``uci``, ``san``, ``value`` and
            ``top_moves`` (a list of ``(san, prob)`` tuples, top 5 by visits).
            When the game is over, ``move``/``uci``/``san`` are None,
            ``top_moves`` is empty, and ``value`` is the network estimate.
        """
        num_sims = simulations if simulations is not None else self.simulations

        if board.is_game_over():
            return {
                "move": None,
                "uci": None,
                "san": None,
                "value": self._root_value(board),
                "top_moves": [],
            }

        # Value estimate from a single network eval of the root position.
        value = self._root_value(board)

        # Visit-count distribution over legal moves at the root.
        visit_dist = self.mcts.run(board, num_sims, add_noise=False)

        # Best move = highest visit share.
        best_move = max(visit_dist, key=visit_dist.get)

        # Top 5 candidate moves by visit probability, rendered as SAN.
        ranked = sorted(visit_dist.items(), key=lambda kv: kv[1], reverse=True)
        top_moves: List[Tuple[str, float]] = [
            (board.san(move), float(prob)) for move, prob in ranked[:5]
        ]

        return {
            "move": best_move,
            "uci": best_move.uci(),
            "san": board.san(best_move),
            "value": value,
            "top_moves": top_moves,
        }

    def play_move(self, board: chess.Board) -> chess.Move:
        """Return the argmax-visit (temperature 0) move for ``board``.

        The board is not mutated.
        """
        visit_dist = self.mcts.run(board, self.simulations, add_noise=False)
        return max(visit_dist, key=visit_dist.get)


def suggest(fen: str, model_path: str, simulations: int = 200) -> dict:
    """Build a board from ``fen`` and return an ``AlphaChessAgent`` suggestion.

    Args:
        fen: FEN string describing the position.
        model_path: Path to a trained model checkpoint.
        simulations: Number of MCTS simulations to run.

    Returns:
        The dict produced by :meth:`AlphaChessAgent.suggest_move`.
    """
    board = chess.Board(fen)
    agent = AlphaChessAgent(model_path, simulations=simulations)
    return agent.suggest_move(board)
