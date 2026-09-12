"""Tests for the GUI's undo/redo history, last-move highlight and sounds.

Redo has to mirror whatever Undo did: against the agent an Undo takes back a
human/agent *pair*, so a Redo must restore the pair -- and must replay the
agent's recorded reply rather than re-searching, or Redo would not be the
inverse of Undo for a stochastic agent. Playing anything other than what was
taken back starts a new line and must discard the redo history, which is the
part that silently corrupts a game if it is wrong.

These run headless against an offscreen surface; no window is opened.
"""

from __future__ import annotations

import os

import chess
import pytest

# A dummy video driver lets pygame initialise with no display attached.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

pygame = pytest.importorskip("pygame")

from alpha_chess.gui import (  # noqa: E402
    MARGIN,
    SQUARE,
    _AUDIO_HZ,
    _GuiApp,
    _click_samples,
    _SoundBank,
    _square_to_screen,
)


class _FirstMoveAgent:
    """Deterministic stand-in for the network agent."""

    def play_move(self, board):
        return sorted(board.legal_moves, key=str)[0]

    def suggest_move(self, board, simulations=None):
        return {"value": 0.0, "san": "", "top_moves": []}


@pytest.fixture(scope="module")
def surface():
    pygame.init()
    pygame.font.init()
    return pygame.Surface((1200, 800))


def _app(surface, **kwargs):
    kwargs.setdefault("agent", None)
    kwargs.setdefault("agent_error", None)
    return _GuiApp(pygame, surface, has_display=False, **kwargs)


def _ucis(app):
    return [m.uci() for m in app.board.move_stack]


def _play(app, san_moves):
    for san in san_moves:
        app.board.push_san(san)


def _pixel(square, flipped=False):
    """Centre pixel of a board square, for driving _play_click."""
    file, rank = chess.square_file(square), chess.square_rank(square)
    col = 7 - file if flipped else file
    row = rank if flipped else 7 - rank
    return (MARGIN + col * SQUARE + SQUARE // 2,
            MARGIN + row * SQUARE + SQUARE // 2)


# --------------------------------------------------------------------------- #
# Undo/redo symmetry
# --------------------------------------------------------------------------- #
def test_redo_restores_the_pair_undo_took_back(surface):
    app = _app(surface, agent=_FirstMoveAgent())
    _play(app, ["e4", "e5", "Nf3", "Nc6"])
    app._undo()
    assert _ucis(app) == ["e2e4", "e7e5"]
    app._redo()
    assert _ucis(app) == ["e2e4", "e7e5", "g1f3", "b8c6"]
    assert app.redo_stack == []


def test_repeated_undo_then_redo_preserves_move_order(surface):
    app = _app(surface, agent=_FirstMoveAgent())
    _play(app, ["e4", "e5", "Nf3", "Nc6"])
    app._undo()
    app._undo()
    assert _ucis(app) == []
    app._redo()
    app._redo()
    assert _ucis(app) == ["e2e4", "e7e5", "g1f3", "b8c6"]


def test_advisor_mode_undoes_and_redoes_a_single_ply(surface):
    app = _app(surface, advisor=True)
    _play(app, ["e4", "e5"])
    app._undo()
    assert _ucis(app) == ["e2e4"]
    app._redo()
    assert _ucis(app) == ["e2e4", "e7e5"]


def test_human_vs_human_undoes_and_redoes_a_single_ply(surface):
    app = _app(surface)          # agent is None
    _play(app, ["e4", "e5"])
    app._undo()
    assert _ucis(app) == ["e2e4"]
    app._redo()
    assert _ucis(app) == ["e2e4", "e7e5"]


def test_redo_replays_the_recorded_reply_not_a_fresh_search(surface):
    """A stochastic agent must not get to pick a different move on redo."""

    class _ChangesItsMind:
        def __init__(self):
            self.calls = 0

        def play_move(self, board):
            self.calls += 1
            return sorted(board.legal_moves, key=str)[self.calls]

        def suggest_move(self, board, simulations=None):
            return {"value": 0.0, "san": "", "top_moves": []}

    app = _app(surface, agent=_ChangesItsMind())
    _play(app, ["e4", "e5"])
    app._undo()
    app._redo()
    assert _ucis(app) == ["e2e4", "e7e5"]
    assert app.agent.calls == 0


# --------------------------------------------------------------------------- #
# Discarding the redo line
# --------------------------------------------------------------------------- #
def test_playing_a_different_move_discards_the_redo_history(surface):
    app = _app(surface, advisor=True)
    _play(app, ["e4", "e5"])
    app._undo()
    assert len(app.redo_stack) == 1
    # Click out a different reply: 1...c5 instead of the taken-back 1...e5.
    app._play_click(_pixel(chess.C7))
    app._play_click(_pixel(chess.C5))
    assert _ucis(app) == ["e2e4", "c7c5"]
    assert app.redo_stack == []
    app._redo()
    assert _ucis(app) == ["e2e4", "c7c5"]


def test_replaying_the_same_move_by_hand_keeps_the_rest_redoable(surface):
    app = _app(surface, advisor=True)
    _play(app, ["e4", "e5", "Nf3"])
    app._undo()
    app._undo()
    assert len(app.redo_stack) == 2
    app._play_click(_pixel(chess.E7))
    app._play_click(_pixel(chess.E5))
    assert _ucis(app) == ["e2e4", "e7e5"]
    app._redo()
    assert _ucis(app) == ["e2e4", "e7e5", "g1f3"]


def test_new_game_discards_the_redo_history(surface):
    app = _app(surface, advisor=True)
    _play(app, ["e4", "e5"])
    app._undo()
    app._new_game()
    assert app.redo_stack == []
    app._redo()
    assert _ucis(app) == []


def test_adopting_an_edited_position_discards_the_redo_history(surface):
    app = _app(surface, advisor=True)
    _play(app, ["e4", "e5"])
    app._undo()
    # Adopt a hand-built position from the editor.
    app.setup_pieces = {
        chess.E1: chess.Piece(chess.KING, chess.WHITE),
        chess.E8: chess.Piece(chess.KING, chess.BLACK),
    }
    app.setup_turn = chess.WHITE
    app._play_from_position()
    assert app.mode == "play"
    assert app.redo_stack == []


def test_redo_on_an_empty_history_says_so_and_changes_nothing(surface):
    app = _app(surface, advisor=True)
    _play(app, ["e4"])
    app._redo()
    assert _ucis(app) == ["e2e4"]
    assert app.status_msg == "Nothing to redo"


def test_undo_past_the_start_then_redo_returns_the_whole_game(surface):
    app = _app(surface, agent=_FirstMoveAgent())
    _play(app, ["e4", "e5"])
    app._undo()
    app._undo()          # already empty; must not raise or desync
    assert _ucis(app) == []
    app._redo()
    assert _ucis(app) == ["e2e4", "e7e5"]


# --------------------------------------------------------------------------- #
# Key binding
# --------------------------------------------------------------------------- #
def test_r_key_redoes_in_play_mode(surface):
    app = _app(surface, advisor=True)
    _play(app, ["e4", "e5"])
    assert app.handle_keydown(pygame.K_u) is True
    assert _ucis(app) == ["e2e4"]
    assert app.handle_keydown(pygame.K_r) is True
    assert _ucis(app) == ["e2e4", "e7e5"]


def test_r_key_still_resets_the_editor_in_setup_mode(surface):
    """R is redo in play mode but must keep its editor meaning in setup."""
    app = _app(surface, advisor=True)
    app._enter_setup()
    assert app.mode == "setup"
    app.setup_pieces = {}
    app.handle_keydown(pygame.K_r)
    assert app.mode == "setup"
    assert app.setup_pieces != {}      # _reset_editor repopulated the board


# --------------------------------------------------------------------------- #
# Last-move highlight
# --------------------------------------------------------------------------- #
def _square_pixel(app, square):
    """Colour of a square's top-left corner region, clear of any piece glyph."""
    x, y = _square_to_screen(square, app.flipped)
    return app.surface.get_at((x + 4, y + 4))[:3]


def test_last_move_squares_are_tinted(surface):
    app = _app(surface, advisor=True)
    app.draw()
    before_from = _square_pixel(app, chess.E2)
    before_to = _square_pixel(app, chess.E4)

    _play(app, ["e4"])
    app.draw()
    assert _square_pixel(app, chess.E2) != before_from
    assert _square_pixel(app, chess.E4) != before_to


def test_untouched_squares_are_not_tinted(surface):
    app = _app(surface, advisor=True)
    app.draw()
    before = _square_pixel(app, chess.A5)
    _play(app, ["e4"])
    app.draw()
    assert _square_pixel(app, chess.A5) == before


def test_the_destination_is_tinted_more_strongly_than_the_origin(surface):
    """The piece's new home should read louder than the square it left."""
    app = _app(surface, advisor=True)
    # e2, e4 and the e6 reference are all light squares, so the only
    # difference between them is the tint.
    _play(app, ["e4"])
    app.draw()
    plain = _square_pixel(app, chess.E6)      # same colour, untouched
    origin = _square_pixel(app, chess.E2)
    dest = _square_pixel(app, chess.E4)
    assert 0 < _dist(origin, plain) < _dist(dest, plain)


def _dist(a, b):
    return sum(abs(int(x) - int(y)) for x, y in zip(a, b))


def test_highlight_follows_undo_and_redo(surface):
    """It is read off the move stack, so it must never go stale."""
    app = _app(surface, advisor=True)
    _play(app, ["e4", "e5"])
    app.draw()
    tinted_e5 = _square_pixel(app, chess.E5)

    app._undo()
    app.draw()
    assert _square_pixel(app, chess.E5) != tinted_e5   # no longer the last move

    app._redo()
    app.draw()
    assert _square_pixel(app, chess.E5) == tinted_e5


def test_no_highlight_on_a_fresh_board(surface):
    app = _app(surface, advisor=True)
    app.draw()
    baseline = {sq: _square_pixel(app, sq) for sq in chess.SQUARES}
    app._new_game()
    app.draw()
    assert {sq: _square_pixel(app, sq) for sq in chess.SQUARES} == baseline


# --------------------------------------------------------------------------- #
# Move sounds
# --------------------------------------------------------------------------- #
def test_click_samples_are_int16_stereo_and_decay_to_silence():
    """A click that does not decay ends in an audible pop."""
    import numpy

    samples = _click_samples(
        numpy, freq=880.0, ms=55, decay=70.0, noise=0.45, gain=0.35)
    assert samples.dtype == numpy.int16
    assert samples.shape == (int(_AUDIO_HZ * 0.055), 2)
    peak = abs(samples).max()
    tail = abs(samples[-int(_AUDIO_HZ * 0.005):]).max()
    assert peak > 1000                      # actually audible
    assert tail < peak * 0.1                # faded out by the end


def test_click_samples_are_deterministic():
    """The same move must not sound different each time it is played."""
    import numpy

    kw = dict(freq=880.0, ms=20, decay=70.0, noise=0.5, gain=0.3)
    assert (_click_samples(numpy, **kw) == _click_samples(numpy, **kw)).all()


def test_capture_and_quiet_moves_get_different_clicks(surface):
    bank = _SoundBank(pygame)
    board = chess.Board()
    board.push_san("e4")
    board.push_san("d5")
    assert bank.for_move(board, chess.Move.from_uci("e4d5")) == "capture"
    assert bank.for_move(board, chess.Move.from_uci("g1f3")) == "move"


def test_headless_app_makes_no_sound_bank(surface):
    """A test run or offscreen render must never touch an audio device."""
    app = _app(surface, advisor=True)
    assert app.sounds is None
    _play(app, ["e4"])
    app._undo()
    app._redo()                  # goes through _push_move; must not raise


def test_mute_toggle_reports_when_sound_is_unavailable(surface):
    app = _app(surface, advisor=True)     # headless -> no sound bank
    app.handle_keydown(pygame.K_m)
    assert app.status_msg == "Sound unavailable"


def test_mute_toggle_flips_and_reports(surface):
    app = _app(surface, advisor=True)
    app.sounds = _SoundBank(pygame)
    if not app.sounds.enabled:
        pytest.skip("no audio device available")
    app.handle_keydown(pygame.K_m)
    assert app.sounds.muted is True
    assert app.status_msg == "Sound off"
    app.handle_keydown(pygame.K_m)
    assert app.sounds.muted is False
    assert app.status_msg == "Sound on"


def test_muted_bank_plays_nothing(surface):
    bank = _SoundBank(pygame)
    bank.muted = True
    bank.play("move")            # must be a silent no-op, not an error
    bank.enabled = False
    bank.play("capture")
