"""Tests for strength limiting and Elo reporting (`evaluate.py`).

The interesting part is the UCI opponent's strength configuration. Engines
advertise a range for each option and quietly do their own thing with a value
outside it -- Stockfish 16's ``UCI_Elo`` floor is 1320, so a request for a
600-Elo opponent yields a 1320-Elo one -- which is easy to mistake for a real
result. These tests pin down that such a request is clamped, that it says so,
that ``Skill Level`` reaches below the Elo floor, and that the two throttles
are refused together (an engine limiting by Elo ignores its skill level).

The engine-backed tests skip when no UCI engine is installed.
"""

from __future__ import annotations

import os
import shutil

import chess
import pytest

from alpha_chess.evaluate import (
    UCIOpponent,
    _make_opponent,
    estimate_elo_diff,
)

# Where the common packages put Stockfish; apt's /usr/games is not on PATH.
_CANDIDATES = (
    "/usr/games/stockfish",
    "/usr/local/bin/stockfish",
    "/opt/homebrew/bin/stockfish",
)


def _find_engine():
    found = shutil.which("stockfish")
    if found:
        return found
    for path in _CANDIDATES:
        if os.path.exists(path):
            return path
    return None


ENGINE = _find_engine()
needs_engine = pytest.mark.skipif(
    ENGINE is None, reason="no UCI engine installed")


# --------------------------------------------------------------------------- #
# Elo estimation (no engine needed)
# --------------------------------------------------------------------------- #
def test_elo_diff_is_zero_at_even_score():
    assert estimate_elo_diff(0.5, games=40) == pytest.approx(0.0)


def test_elo_diff_signs_and_symmetry():
    assert estimate_elo_diff(0.75, games=40) > 0
    assert estimate_elo_diff(0.25, games=40) < 0
    assert estimate_elo_diff(0.75, games=40) == pytest.approx(
        -estimate_elo_diff(0.25, games=40))


def test_elo_diff_clamps_a_whitewash_instead_of_diverging():
    """A 0 or 1 score is finite, bounded by the 1/(2*games) clamp."""
    for games in (10, 40, 400):
        assert estimate_elo_diff(0.0, games=games) > -2000
        assert estimate_elo_diff(1.0, games=games) < 2000
    # More games -> a whitewash is stronger evidence -> a larger estimate.
    assert estimate_elo_diff(1.0, games=400) > estimate_elo_diff(1.0, games=10)


# --------------------------------------------------------------------------- #
# UCI strength limiting
# --------------------------------------------------------------------------- #
@needs_engine
def test_uci_elo_below_the_engine_floor_is_clamped_and_reported(capsys):
    opp = UCIOpponent(ENGINE, uci_elo=100)
    try:
        low = opp._engine.options["UCI_Elo"].min
        assert opp.strength_note == "UCI_Elo {}".format(int(low))
        assert "outside this engine's supported range" in capsys.readouterr().err
    finally:
        opp.close()


@needs_engine
def test_uci_elo_within_range_is_applied_verbatim(capsys):
    opp = UCIOpponent(ENGINE, uci_elo=1600)
    try:
        assert opp.strength_note == "UCI_Elo 1600"
        assert capsys.readouterr().err == ""
    finally:
        opp.close()


@needs_engine
def test_skill_level_reaches_below_the_elo_floor():
    """Skill Level 0 is the setting --uci-elo cannot express."""
    opp = UCIOpponent(ENGINE, skill_level=0)
    try:
        assert opp.strength_note == "Skill Level 0"
        # Still a working opponent, not a crippled one.
        assert opp.choose_move(chess.Board()) in chess.Board().legal_moves
    finally:
        opp.close()


@needs_engine
def test_skill_level_above_range_is_clamped_and_reported(capsys):
    opp = UCIOpponent(ENGINE, skill_level=999)
    try:
        high = opp._engine.options["Skill Level"].max
        assert opp.strength_note == "Skill Level {}".format(int(high))
        assert "outside this engine's supported range" in capsys.readouterr().err
    finally:
        opp.close()


@needs_engine
def test_elo_and_skill_together_are_refused():
    with pytest.raises(ValueError, match="only one"):
        UCIOpponent(ENGINE, uci_elo=1600, skill_level=3)


@needs_engine
def test_opponent_label_names_the_applied_strength():
    """The label must not read as the strength that was merely requested."""
    opp, label, closer = _make_opponent(
        "uci:" + ENGINE, seed=0, uci_elo=100)
    try:
        low = int(opp._engine.options["UCI_Elo"].min)
        assert label == "uci:{} @ UCI_Elo {}".format(ENGINE, low)
    finally:
        closer()


@needs_engine
def test_unlimited_opponent_has_no_strength_note():
    opp, label, closer = _make_opponent("uci:" + ENGINE, seed=0, uci_elo=None)
    try:
        assert opp.strength_note is None
        assert label == "uci:" + ENGINE
    finally:
        closer()


def test_missing_engine_path_raises_before_spawning():
    with pytest.raises(FileNotFoundError, match="UCI engine not found"):
        UCIOpponent("/nonexistent/stockfish", uci_elo=1600)
