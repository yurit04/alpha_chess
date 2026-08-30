from __future__ import annotations

"""Evaluation tools for gauging the strength of a trained AlphaChess model.

This module provides light-weight baseline opponents, a match runner that pits
two move-choosers against each other over a series of games (optionally
alternating colors), an Elo-difference estimator, and a high-level
:func:`evaluate_model` entry point that loads a checkpoint via
:class:`alpha_chess.agent.AlphaChessAgent` and plays it against a chosen
opponent (``random``, ``material``, another ``model:PATH`` checkpoint, or an
external ``uci:PATH`` engine).

Honesty note: the Elo difference reported here is RELATIVE to whatever opponent
was used. It only maps to an absolute Elo when the opponent is a calibrated UCI
engine configured to a known strength.
"""

import math
import os
import random
from typing import Callable, Dict, List, Optional, Union

import chess

# Standard material values (king intentionally 0 for material counting).
PIECE_VALUES: Dict[int, float] = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

# A move-chooser is anything with a ``choose_move(board)`` method, or a bare
# callable ``chooser(board) -> chess.Move``. Objects exposing ``play_move`` (the
# agent API) are also accepted.
Chooser = Union[Callable[[chess.Board], chess.Move], object]


# --------------------------------------------------------------------------- #
# Baseline opponents
# --------------------------------------------------------------------------- #
class RandomOpponent:
    """Chooses a uniformly random legal move."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def choose_move(self, board: chess.Board) -> chess.Move:
        """Return a random legal move for ``board`` (board is not mutated)."""
        moves = list(board.legal_moves)
        if not moves:
            raise ValueError("RandomOpponent asked to move in a terminal position.")
        return self._rng.choice(moves)


class MaterialOpponent:
    """Greedy 1-ply material-maximizing player.

    For every legal move it evaluates the resulting position's material balance
    from the mover's perspective, strongly prefers checkmate, and breaks ties
    randomly. ``depth`` is accepted for API/forward-compatibility; only a 1-ply
    greedy search is implemented (depth is otherwise treated as informational).
    """

    def __init__(self, depth: int = 1, seed: Optional[int] = None) -> None:
        self.depth = depth
        self._rng = random.Random(seed)

    @staticmethod
    def _material_from_perspective(board: chess.Board, color: bool) -> float:
        """Material balance (our pieces minus theirs) from ``color``'s view."""
        score = 0.0
        for piece_type, value in PIECE_VALUES.items():
            score += value * len(board.pieces(piece_type, color))
            score -= value * len(board.pieces(piece_type, not color))
        return score

    def choose_move(self, board: chess.Board) -> chess.Move:
        """Return the greedily best material move (board is not mutated)."""
        moves = list(board.legal_moves)
        if not moves:
            raise ValueError("MaterialOpponent asked to move in a terminal position.")

        mover = board.turn
        best_score = -math.inf
        best_moves: List[chess.Move] = []
        for move in moves:
            board.push(move)
            try:
                if board.is_checkmate():
                    # We just delivered mate: best possible outcome.
                    score = math.inf
                else:
                    score = self._material_from_perspective(board, mover)
            finally:
                board.pop()

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return self._rng.choice(best_moves)


# --------------------------------------------------------------------------- #
# Chooser adapters
# --------------------------------------------------------------------------- #
class _AgentChooser:
    """Adapts an :class:`AlphaChessAgent` (``play_move``) to the chooser API."""

    def __init__(self, agent) -> None:
        self._agent = agent

    def choose_move(self, board: chess.Board) -> chess.Move:
        return self._agent.play_move(board)


class UCIOpponent:
    """Wraps an external UCI engine as a move-chooser.

    The engine process is opened lazily via ``chess.engine.SimpleEngine`` and
    must be closed with :meth:`close`. When ``uci_elo`` is provided and the
    engine advertises ``UCI_LimitStrength``/``UCI_Elo``, the engine is
    configured to that strength.
    """

    def __init__(
        self,
        engine_path: str,
        uci_elo: Optional[int] = None,
        time_limit: float = 0.1,
    ) -> None:
        # Import guarded here so the rest of the module works without an engine.
        import chess.engine  # noqa: F401  (import for side effect / availability)

        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"UCI engine not found: {engine_path}")

        self._engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self._limit = chess.engine.Limit(time=time_limit)

        if uci_elo is not None:
            try:
                options = self._engine.options
                config = {}
                if "UCI_LimitStrength" in options:
                    config["UCI_LimitStrength"] = True
                if "UCI_Elo" in options:
                    opt = options["UCI_Elo"]
                    lo = getattr(opt, "min", None)
                    hi = getattr(opt, "max", None)
                    elo = uci_elo
                    if lo is not None:
                        elo = max(elo, int(lo))
                    if hi is not None:
                        elo = min(elo, int(hi))
                    config["UCI_Elo"] = elo
                if config:
                    self._engine.configure(config)
            except Exception:
                # Strength limiting is best-effort; ignore unsupported engines.
                pass

    def choose_move(self, board: chess.Board) -> chess.Move:
        result = self._engine.play(board, self._limit)
        if result.move is None:
            raise ValueError("UCI engine returned no move.")
        return result.move

    def close(self) -> None:
        """Shut down the underlying engine process (safe to call twice)."""
        engine = getattr(self, "_engine", None)
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                pass
            self._engine = None


def _get_move(chooser: Chooser, board: chess.Board) -> chess.Move:
    """Obtain a move from a chooser, supporting choose_move/play_move/callable."""
    if hasattr(chooser, "choose_move"):
        return chooser.choose_move(board)
    if hasattr(chooser, "play_move"):
        return chooser.play_move(board)  # type: ignore[attr-defined]
    if callable(chooser):
        return chooser(board)
    raise TypeError(
        "Chooser must have choose_move(board), play_move(board), or be callable."
    )


# --------------------------------------------------------------------------- #
# Match runner
# --------------------------------------------------------------------------- #
def _random_opening(plies: int, rng: random.Random) -> chess.Board:
    """Return a board after ``plies`` random legal moves from the start.

    Retries so the opening is never already game-over; falls back to the
    standard start position. Used to decorrelate evaluation games (deterministic
    engines otherwise replay the identical game from the standard start).
    """
    if plies <= 0:
        return chess.Board()
    for _ in range(20):
        board = chess.Board()
        for _ in range(plies):
            if board.is_game_over(claim_draw=True):
                break
            moves = list(board.legal_moves)
            board.push(moves[rng.randrange(len(moves))])
        if not board.is_game_over(claim_draw=True):
            return board
    return chess.Board()


def play_match(
    white_chooser: Chooser,
    black_chooser: Chooser,
    games: int,
    max_moves: int = 300,
    swap_colors: bool = True,
    seed: Optional[int] = None,
    opening_random_plies: int = 4,
) -> Dict[str, float]:
    """Play ``games`` games between two choosers and tally results.

    Player A is ``white_chooser`` (the FIRST player) and player B is
    ``black_chooser`` (the SECOND player). When ``swap_colors`` is True the two
    players alternate colors game-by-game; results are always reported from
    player A's perspective regardless of the color A played.

    Each game (or color-swapped *pair* of games, when ``swap_colors``) starts
    from a seeded random opening of ``opening_random_plies`` plies. This
    decorrelates games so two *deterministic* players (e.g. AlphaChess vs a
    previous checkpoint, both argmax) produce varied game lines instead of
    replaying one game; set it to 0 to always start from the standard position.

    A game that reaches ``max_moves`` without a natural conclusion is scored as
    a draw. No board supplied by the caller is mutated (fresh boards are used).

    Returns:
        A dict with ``wins_a``, ``draws``, ``wins_b`` (counts) and
        ``score_a = (wins_a + 0.5 * draws) / games``.
    """
    wins_a = 0
    wins_b = 0
    draws = 0
    rng = random.Random(seed)
    opening = chess.Board()

    for g in range(games):
        # Determine which player takes White this game.
        a_is_white = True
        if swap_colors and (g % 2 == 1):
            a_is_white = False

        white_player = white_chooser if a_is_white else black_chooser
        black_player = black_chooser if a_is_white else white_chooser

        # Fresh opening per game, or per color-swapped pair (fair paired play).
        if (not swap_colors) or (g % 2 == 0):
            opening = _random_opening(opening_random_plies, rng)
        board = opening.copy()
        move_count = 0
        while not board.is_game_over(claim_draw=True) and move_count < max_moves:
            player = white_player if board.turn == chess.WHITE else black_player
            move = _get_move(player, board)
            board.push(move)
            move_count += 1

        # Decide the outcome (unfinished games at the move cap count as draws).
        if board.is_game_over(claim_draw=True):
            result = board.result(claim_draw=True)
        else:
            result = "1/2-1/2"

        if result == "1-0":
            white_won = True
            draw = False
        elif result == "0-1":
            white_won = False
            draw = False
        else:
            white_won = False
            draw = True

        if draw:
            draws += 1
        else:
            a_won = (white_won and a_is_white) or ((not white_won) and (not a_is_white))
            if a_won:
                wins_a += 1
            else:
                wins_b += 1

    denom = games if games > 0 else 1
    score_a = (wins_a + 0.5 * draws) / denom
    return {
        "wins_a": wins_a,
        "draws": draws,
        "wins_b": wins_b,
        "score_a": score_a,
    }


# --------------------------------------------------------------------------- #
# Elo estimation
# --------------------------------------------------------------------------- #
def estimate_elo_diff(score: float, games: int) -> float:
    """Estimate the Elo difference of player A relative to the opponent.

    Uses the standard logistic inversion ``elo = -400 * log10(1/score - 1)``.
    Scores of exactly 0 or 1 (or otherwise out of range) would yield an
    infinite estimate, so the score is clamped to ``[1/(2G), 1 - 1/(2G)]`` and
    the result is a capped BOUND tied to the sample size (more games -> a wider,
    more confident bound).

    Args:
        score: Player A's score fraction in ``[0, 1]``.
        games: Number of games played (used to set the clamp bound).

    Returns:
        Estimated Elo difference (positive means A is stronger).
    """
    if games <= 0:
        return 0.0
    eps = 1.0 / (2.0 * games)
    s = min(max(score, eps), 1.0 - eps)
    return -400.0 * math.log10(1.0 / s - 1.0)


# --------------------------------------------------------------------------- #
# High-level model evaluation
# --------------------------------------------------------------------------- #
def _make_opponent(opponent_spec: str, seed: Optional[int], uci_elo: Optional[int]):
    """Build an opponent chooser from a spec string.

    Returns ``(chooser, label, closer)`` where ``closer`` is a zero-arg callable
    to release resources (a no-op for stateless opponents).
    """
    spec = opponent_spec.strip()

    if spec == "random":
        return RandomOpponent(seed=seed), "random", (lambda: None)

    if spec == "material":
        return MaterialOpponent(seed=seed), "material", (lambda: None)

    if spec.startswith("model:"):
        path = spec[len("model:"):]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Opponent model checkpoint not found: {path}")
        # Imported lazily to avoid a hard torch dependency at import time.
        from alpha_chess.agent import AlphaChessAgent

        agent = AlphaChessAgent(path)
        return _AgentChooser(agent), f"model:{path}", (lambda: None)

    if spec.startswith("uci:"):
        path = spec[len("uci:"):]
        opp = UCIOpponent(path, uci_elo=uci_elo)
        return opp, f"uci:{path}", opp.close

    raise ValueError(
        f"Unknown opponent spec: {opponent_spec!r}. "
        "Expected 'random', 'material', 'model:PATH', or 'uci:PATH'."
    )


def evaluate_model(
    model_path: str,
    opponent_spec: str,
    games: int,
    simulations: int,
    device=None,
    max_moves: int = 300,
    seed: Optional[int] = None,
    uci_elo: Optional[int] = None,
    opening_random_plies: int = 4,
) -> Dict[str, object]:
    """Evaluate a trained model against a chosen opponent.

    Loads the model at ``model_path`` as an :class:`AlphaChessAgent` and plays it
    (player A) against ``opponent_spec``: one of ``"random"``, ``"material"``,
    ``"model:PATH"`` (another AlphaChess checkpoint), or ``"uci:PATH"`` (an
    external UCI engine; ``uci_elo`` configures its strength when supported).

    Args:
        model_path: Path to the AlphaChess checkpoint to evaluate.
        opponent_spec: Opponent selector string.
        games: Number of games to play.
        simulations: MCTS simulations per move for the evaluated model.
        device: Torch device (or None to auto-select).
        max_moves: Move cap per game (games hitting the cap are scored as draws).
        seed: Seed for stochastic opponents / color alternation.
        uci_elo: Optional target Elo for a UCI opponent.

    Returns:
        A dict with ``score``, ``wins``, ``draws``, ``losses``, ``elo_diff`` and
        ``opponent`` (all from the evaluated model's perspective). The
        ``elo_diff`` is RELATIVE to the chosen opponent.

    Raises:
        FileNotFoundError: If ``model_path`` (or a referenced opponent path) is
            missing.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Imported lazily so this module imports cleanly even before torch is ready.
    from alpha_chess.agent import AlphaChessAgent

    agent = AlphaChessAgent(model_path, device=device, simulations=simulations)
    model_chooser = _AgentChooser(agent)

    opponent, label, closer = _make_opponent(opponent_spec, seed=seed, uci_elo=uci_elo)
    try:
        result = play_match(
            model_chooser,
            opponent,
            games=games,
            max_moves=max_moves,
            swap_colors=True,
            seed=seed,
            opening_random_plies=opening_random_plies,
        )
    finally:
        closer()

    wins = int(result["wins_a"])
    draws = int(result["draws"])
    losses = int(result["wins_b"])
    score = float(result["score_a"])
    elo_diff = estimate_elo_diff(score, games)

    return {
        "score": score,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "elo_diff": elo_diff,
        "opponent": label,
    }
