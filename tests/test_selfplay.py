"""Tests for the high-throughput self-play engine and the replay buffer."""

from __future__ import annotations

import numpy as np
import chess
import pytest
import torch

from alpha_chess.batched_selfplay import (
    MAX_POLICY_TARGETS,
    Node,
    SelfPlayBatch,
    SelfPlayEngine,
    TorchEvaluator,
    _Game,
    generate_selfplay_data,
)
from alpha_chess.encoding import NUM_PLANES
from alpha_chess.network import AlphaZeroNet
from alpha_chess.self_play import ReplayBuffer


def _engine(**kwargs):
    """A tiny CPU engine; the network only has to be shaped correctly."""
    net = AlphaZeroNet(channels=8, num_blocks=1)
    evaluator = TorchEvaluator(net, torch.device("cpu"), use_amp=False)
    params = dict(
        games_in_flight=4,
        num_simulations=16,
        max_moves=40,
        resign_threshold=None,
        seed=3,
    )
    params.update(kwargs)
    return SelfPlayEngine(evaluator, **params)


def _uniform_evaluator(states, idx, counts):
    """A network stand-in: uniform priors over legal moves, value 0."""
    n, width = idx.shape
    priors = np.zeros((n, width), dtype=np.float32)
    for i in range(n):
        priors[i, : counts[i]] = 1.0 / counts[i]
    return priors, np.zeros(n, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Output shape / invariants
# --------------------------------------------------------------------------- #
def test_batch_invariants():
    batch = _engine().run(4)
    assert len(batch) == int(batch.stats["plies"])
    assert batch.states.dtype == np.uint8
    assert batch.states.shape[1:] == (NUM_PLANES, 8, 8)
    assert batch.pol_idx.shape[1] == MAX_POLICY_TARGETS
    assert batch.pol_idx.max() < 4672
    # Policy rows are normalized and zero-padded past their length.
    assert np.allclose(batch.pol_val.sum(axis=1), 1.0, atol=1e-5)
    lengths = (batch.pol_val > 0).sum(axis=1)
    assert lengths.min() >= 1
    for row, length in zip(batch.pol_val, lengths):
        assert row[length:].sum() == 0.0
    # Values are game results seen from the mover.
    assert set(np.unique(batch.values)).issubset({-1.0, 0.0, 1.0})


def test_values_are_opposite_for_alternating_movers():
    """Within one game the value target must flip sign every ply."""
    engine = _engine(games_in_flight=1, num_simulations=8, max_moves=30)
    batch = engine.run(1)
    values = batch.values
    if values[0] != 0.0:  # a decisive game
        signs = np.sign(values)
        assert np.all(signs[1:] == -signs[:-1])


# --------------------------------------------------------------------------- #
# Terminal detection: the O(1) replacement for is_game_over(claim_draw=True)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "fen,expected",
    [
        # Black to move and checkmated (fool's mate).
        ("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", -1.0),
        # Stalemate, black to move.
        ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 0.0),
        # Bare kings: insufficient material.
        ("8/8/4k3/8/8/4K3/8/8 w - - 0 1", 0.0),
        # A normal position is not terminal.
        (chess.STARTING_FEN, None),
    ],
)
def test_resolve_terminal_matches_python_chess(fen, expected):
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    assert SelfPlayEngine._resolve_terminal(board, legal, 0) == expected


def test_fifty_move_and_repetition_draws():
    board = chess.Board("8/8/4k3/8/8/4K3/7R/8 w - - 100 90")
    legal = list(board.legal_moves)
    assert SelfPlayEngine._resolve_terminal(board, legal, 0) == 0.0
    # Third occurrence of a position is a claimable draw.
    board = chess.Board()
    assert SelfPlayEngine._resolve_terminal(board, list(board.legal_moves), 2) == 0.0
    assert SelfPlayEngine._resolve_terminal(board, list(board.legal_moves), 1) is None


def test_full_games_agree_with_python_chess():
    """Every game the engine calls finished must really be over."""
    engine = _engine(games_in_flight=1, num_simulations=8, max_moves=120)
    for _ in range(3):
        game = _Game(allow_resign=False)
        game.sims_left = 8
        plies = 0
        while True:
            while game.sims_left > 0:
                engine._simulate([game])
            result = engine._advance(game)
            plies += 1
            if result is not None:
                break

        replay = chess.Board()
        for move in game.board.move_stack:
            replay.push(move)
        if plies < 120:
            assert replay.is_game_over(claim_draw=True), replay.fen()
            expected = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}[
                replay.result(claim_draw=True)
            ]
            assert result == expected


# --------------------------------------------------------------------------- #
# Search behaviour
# --------------------------------------------------------------------------- #
def test_subtree_is_reused_across_plies():
    """The played move's subtree must carry its visits into the next ply."""
    # temperature_moves=0 makes move choice greedy, so the move sampled here and
    # the one _advance plays are the same.
    engine = _engine(games_in_flight=1, num_simulations=40, max_moves=10,
                     temperature_moves=0)
    game = _Game(allow_resign=False)
    game.sims_left = 40
    while game.sims_left > 0:
        engine._simulate([game])
    chosen = engine._choose_move(game)
    carried = game.root.children[chosen]
    expected_visits = game.root.N[chosen]
    assert expected_visits > 0
    engine._advance(game)
    assert game.root is carried
    # Every visit through the edge but the one that expanded the child went on
    # to visit a grandchild, so those visits survive as the new root's total.
    assert game.root.n_total == expected_visits - 1


def test_select_prefers_high_prior_on_first_descent():
    node = Node()
    node.expand([chess.Move.from_uci("e2e4")] * 3, [0, 1, 2], [0.1, 0.7, 0.2])
    engine = _engine()
    assert engine._select(node) == 1


def test_search_concentrates_visits_on_a_forced_mate():
    """With a uniform-prior network, search must still find mate in one."""
    engine = SelfPlayEngine(
        _uniform_evaluator, games_in_flight=1, num_simulations=160,
        max_moves=5, resign_threshold=None, dirichlet_epsilon=0.0, seed=1,
    )
    game = _Game(allow_resign=False)
    game.board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    game.rep = {game.board._transposition_key(): 1}
    game.sims_left = 160
    while game.sims_left > 0:
        engine._simulate([game])
    best = game.root.moves[engine._choose_move(game)]
    assert best == chess.Move.from_uci("a1a8"), best


# --------------------------------------------------------------------------- #
# Replay buffer
# --------------------------------------------------------------------------- #
def _fake_batch(n, tag):
    return SelfPlayBatch(
        np.full((n, NUM_PLANES, 8, 8), tag, np.uint8),
        np.zeros((n, MAX_POLICY_TARGETS), np.uint16),
        np.zeros((n, MAX_POLICY_TARGETS), np.float32),
        np.zeros(n, np.int16),
        np.full(n, tag, np.float32),
    )


def test_replay_buffer_ring_and_oversize_append():
    buffer = ReplayBuffer(capacity=10)
    buffer.append(_fake_batch(4, 1))
    assert len(buffer) == 4
    buffer.append(_fake_batch(8, 2))          # wraps around
    assert len(buffer) == 10 and buffer._cursor == 2
    assert set(buffer._values.ravel().tolist()) == {1.0, 2.0}
    buffer.append(_fake_batch(25, 3))         # larger than the whole buffer
    assert len(buffer) == 10
    assert set(buffer._values.ravel().tolist()) == {3.0}


def test_replay_buffer_roundtrip_and_resize():
    buffer = ReplayBuffer(capacity=8)
    buffer.append(_fake_batch(8, 5))
    smaller = ReplayBuffer(capacity=3)
    smaller.load_state_dict(buffer.state_dict())
    assert len(smaller) == 3
    states, pol_idx, pol_val, values = smaller.sample(4)
    assert states.shape == (4, NUM_PLANES, 8, 8) and states.dtype == np.uint8
    assert pol_idx.dtype == np.int64
    assert values.shape == (4, 1)


def test_replay_buffer_rejects_empty_sample():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=4).sample(1)


# --------------------------------------------------------------------------- #
# Multi-process pipeline
# --------------------------------------------------------------------------- #
def test_parallel_matches_single_process_contract(tmp_path):
    net = AlphaZeroNet(channels=8, num_blocks=1)
    batch = generate_selfplay_data(
        net, torch.device("cpu"), num_games=4, num_workers=2,
        games_in_flight=2, num_simulations=8, max_moves=30,
        resign_threshold=None, seed=2, scratch_dir=str(tmp_path), verbose=False,
    )
    assert len(batch) == int(batch.stats["plies"]) > 0
    assert batch.states.dtype == np.uint8
    assert np.allclose(batch.pol_val.sum(axis=1), 1.0, atol=1e-5)


# --------------------------------------------------------------------------- #
# Resignation
# --------------------------------------------------------------------------- #
def test_should_resign_reads_root_values_from_the_movers_side():
    """Resignation triggers on the mover's best root Q, after two of its turns."""
    engine = _engine(resign_threshold=-0.90, resign_plies=2)
    game = _Game(allow_resign=True)
    game.root.expand(
        [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")],
        [0, 1],
        [0.5, 0.5],
    )
    game.root.N = [4, 4]
    game.root.Q = [-0.95, -0.97]      # every reply loses for the mover
    game.root.n_total = 8

    assert engine._should_resign(game) is False   # one turn is not enough
    assert engine._should_resign(game) is True    # two of White's own turns
    assert game.would_resign_side == chess.WHITE

    # A single good reply is enough to keep playing, and resets the streak.
    game.root.Q = [-0.95, 0.2]
    assert engine._should_resign(game) is False
    assert game.resign_streak[int(chess.WHITE)] == 0


def test_the_resign_streak_survives_the_opponents_reply():
    """Regression: consecutive plies alternate the side to move.

    The streak used to be one counter shared by both sides. In a zero-sum game
    the losing side's -0.95 is always followed by the winner's +0.95, so a
    shared counter reset on every reply, could never exceed 1, and any
    ``resign_plies`` above 1 was unsatisfiable -- resignation never fired at
    all, at any threshold. It must survive the opponent's turn.
    """
    engine = _engine(resign_threshold=-0.90, resign_plies=2)
    game = _Game(allow_resign=True)
    game.root.expand(
        [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")],
        [0, 1],
        [0.5, 0.5],
    )
    game.root.N = [4, 4]
    game.root.n_total = 8

    # White to move, and lost: first of White's turns below the threshold.
    game.root.Q = [-0.95, -0.97]
    assert game.board.turn == chess.WHITE
    assert engine._should_resign(game) is False

    # Black replies. The same position is winning from Black's side, which is
    # what used to clear the counter.
    game.board.push(chess.Move.from_uci("e2e4"))
    assert game.board.turn == chess.BLACK
    game.root.Q = [0.95, 0.97]
    assert engine._should_resign(game) is False
    assert game.resign_streak[int(chess.WHITE)] == 1      # White's streak kept

    # Back to White, still lost: the second of White's OWN turns fires it.
    game.board.push(chess.Move.from_uci("e7e5"))
    assert game.board.turn == chess.WHITE
    game.root.Q = [-0.95, -0.97]
    assert engine._should_resign(game) is True
    assert game.would_resign_side == chess.WHITE


def test_resignation_actually_fires_at_the_default_ply_count():
    """End-to-end guard: the default resign_plies must be reachable.

    Pinning this end to end, rather than only on _should_resign, is what would
    have caught resignation being dead: the unit test above passed throughout,
    because it called _should_resign twice without ever advancing the board.
    """
    engine = _engine(
        games_in_flight=4, num_simulations=16, max_moves=200,
        resign_threshold=1.0, resign_plies=2, resign_disable_fraction=0.0,
    )
    batch = engine.run(4)
    assert batch.stats["resigned"] == 4, batch.stats


def test_resign_disabled_game_is_played_out_but_still_measured():
    """A game in the verification fraction records the criterion, ignores it."""
    engine = _engine(resign_threshold=-0.90, resign_plies=1)
    game = _Game(allow_resign=False)
    game.root.expand([chess.Move.from_uci("e2e4")], [0], [1.0])
    game.root.N = [4]
    game.root.Q = [-0.99]
    game.root.n_total = 4

    assert engine._should_resign(game) is False   # suppressed...
    assert game.would_resign_at == 0              # ...but recorded


def test_resignation_keeps_the_ply_count_consistent():
    """A resigning game records the position it gives up in.

    ``resign_threshold=1.0`` is above every possible value, so every game
    resigns as soon as the streak is met -- which is what makes the accounting
    deterministic to check.
    """
    engine = _engine(
        games_in_flight=4, num_simulations=16, max_moves=200,
        resign_threshold=1.0, resign_disable_fraction=0.0,
    )
    batch = engine.run(4)
    assert batch.stats["resigned"] == 4, batch.stats
    assert batch.stats["plies"] / batch.stats["games"] < 20
    assert len(batch) == int(batch.stats["plies"])
    assert set(np.unique(batch.values)).issubset({-1.0, 1.0})


def test_resign_verification_fraction_is_counted():
    engine = _engine(
        games_in_flight=4, num_simulations=16, max_moves=30,
        resign_threshold=1.0, resign_disable_fraction=1.0,
    )
    batch = engine.run(4)
    assert batch.stats["resigned"] == 0            # nothing actually resigned
    assert batch.stats["resign_checked"] == 4      # but all four were checked
    assert len(batch) == int(batch.stats["plies"])
