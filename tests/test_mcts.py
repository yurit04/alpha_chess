"""Tests for the interactive PUCT search (`mcts.py`).

The search's terminal handling is resolved once per node at the leaf rather
than by calling ``is_game_over(claim_draw=True)`` at every node of every
descent, so these tests pin down that it still agrees with python-chess on
every way a game can end, and that the search itself behaves.
"""

from __future__ import annotations

import chess
import pytest
import torch

from alpha_chess.mcts import MCTS, Node
from alpha_chess.network import AlphaZeroNet


@pytest.fixture(scope="module")
def search():
    net = AlphaZeroNet(channels=8, num_blocks=1)
    return MCTS(net, device=torch.device("cpu"))


# (fen, expected terminal value from the side-to-move's view, description)
TERMINAL_CASES = [
    ("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", -1.0, "checkmate"),
    ("7k/5Q2/5K2/8/8/8/8/8 b - - 0 1", 0.0, "stalemate"),
    ("7k/8/6K1/8/8/8/8/8 w - - 0 1", 0.0, "king vs king"),
    ("7k/8/6KB/8/8/8/8/8 w - - 0 1", 0.0, "king+bishop vs king"),
    ("4k3/8/4K3/8/8/8/8/4R3 w - - 100 60", 0.0, "fifty-move rule"),
]


@pytest.mark.parametrize("fen,expected,label", TERMINAL_CASES)
def test_resolve_terminal_matches_expectation(search, fen, expected, label):
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    assert search._resolve_terminal(board, legal) == expected, label


def test_resolve_terminal_is_none_in_play(search):
    board = chess.Board()
    assert search._resolve_terminal(board, list(board.legal_moves)) is None


def test_resolve_terminal_detects_threefold_repetition(search):
    board = chess.Board()
    shuffle = ["g1f3", "g8f6", "f3g1", "f6g8"]
    for _ in range(2):
        for uci in shuffle:
            board.push(chess.Move.from_uci(uci))
    # The start position has now occurred three times.
    assert board.is_repetition(3)
    assert search._resolve_terminal(board, list(board.legal_moves)) == 0.0


def test_terminal_agrees_with_python_chess_over_a_random_game(search):
    """Whenever the search calls a position finished, python-chess agrees."""
    import random

    rng = random.Random(4)
    checked = 0
    for _ in range(40):
        board = chess.Board()
        while not board.is_game_over(claim_draw=True) and len(board.move_stack) < 220:
            board.push(rng.choice(list(board.legal_moves)))
            value = search._resolve_terminal(board, list(board.legal_moves))
            if value is not None:
                assert board.is_game_over(claim_draw=True), board.fen()
                checked += 1
                break
    assert checked > 0, "no random game reached a terminal position"


def test_search_returns_a_distribution_and_leaves_the_board_alone(search):
    board = chess.Board()
    before = board.fen()
    dist = search.run(board, 24)
    assert board.fen() == before
    assert set(dist) == set(board.legal_moves)
    assert dist and abs(sum(dist.values()) - 1.0) < 1e-6


def test_search_finds_mate_in_one(search):
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    assert search.best_move(board, 80) == chess.Move.from_uci("a1a8")


def test_search_in_a_terminal_position_is_empty(search):
    board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    assert search.run(board, 8) == {}


def test_terminal_leaves_are_cached_on_the_node(search):
    """A node found terminal keeps its value, so later descents stop free."""
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    root = Node()
    legal = list(board.legal_moves)
    search._expand(root, board, legal)
    mate = chess.Move.from_uci("a1a8")
    # Enough simulations that PUCT has tried every root move at least once.
    for _ in range(80):
        search._simulate(root, board)
    assert root.child_N[mate] > 0
    child = root.children[mate]
    # Ra8# is mate, so the child is terminal and worth -1 to the side to move.
    assert child.terminal_value == -1.0
    assert not child.is_expanded
