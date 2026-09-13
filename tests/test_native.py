"""Tests for the native self-play core.

The native core reimplements move generation, the board encoding and the PUCT
tree in C, so the risk it carries is *divergence*: a rule it gets subtly wrong,
or a feature plane that does not line up with what the Python side feeds the
network at play time.  These tests pin both down -- perft for the rules, a
direct diff against :mod:`alpha_chess.encoding` for the features -- and then
check that a short self-play run produces well-formed training data.

Everything is skipped when the extension cannot be built (no C compiler), which
is exactly when the training loop falls back to the Python engine.
"""

from __future__ import annotations

import random

import chess
import numpy as np
import pytest
import torch

from alpha_chess import native
from alpha_chess.encoding import NUM_PLANES, encode_board, move_to_index
from alpha_chess.network import AlphaZeroNet
from alpha_chess.self_play import ReplayBuffer

fc = native.load()
pytestmark = pytest.mark.skipif(
    fc is None, reason="native core unavailable: {0}".format(native.last_error())
)


# Standard perft positions; between them they cover castling, en passant,
# promotion, discovered check, pins and stalemate.
PERFT_CASES = [
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4, 197281),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674624),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4, 422333),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 0 1", 4, 2103487),
    ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 1", 4, 3894594),
]


@pytest.mark.parametrize("fen,depth,expected", PERFT_CASES)
def test_perft(fen, depth, expected):
    assert fc.perft(fen, depth) == expected


def _walk(seed, plies):
    """Yield boards from a random game, restarting when one ends."""
    rng = random.Random(seed)
    board = chess.Board()
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves or board.is_game_over() or len(board.move_stack) > 100:
            board = chess.Board()
            continue
        yield board
        board.push(rng.choice(moves))


def test_legal_moves_match_python_chess():
    for board in _walk(11, 1500):
        fen = board.fen(en_passant="fen")
        expected = sorted(m.uci() for m in board.legal_moves)
        got = sorted(uci for uci, _ in fc.legal_moves(fen))
        assert got == expected, fen


def test_move_indices_match_python_encoding():
    for board in _walk(12, 1500):
        fen = board.fen(en_passant="fen")
        native_idx = dict(fc.legal_moves(fen))
        for move in board.legal_moves:
            assert native_idx[move.uci()] == move_to_index(move, board), (
                fen, move.uci()
            )


def test_encoding_matches_python_encoding():
    buf = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    for board in _walk(13, 900):
        fen = board.fen(en_passant="fen")
        for rep in (0, 1, 2):
            buf[:] = 0.0
            fc.encode(fen, buf.reshape(-1), rep)
            assert np.array_equal(buf, encode_board(board, rep)), (fen, rep)


def test_encoding_covers_en_passant_positions():
    """The ep plane is the one feature the two encoders could plausibly
    disagree on, so make sure the comparison above actually exercises it."""
    seen = sum(1 for b in _walk(13, 900) if b.ep_square is not None)
    assert seen > 0


def _run_engine(engine, batch_evaluator, max_steps=20000):
    """Drive one engine to completion against ``batch_evaluator``."""
    n_games = engine.width()
    states = np.zeros((n_games, NUM_PLANES * 64), dtype=np.float32)
    idx = np.zeros((n_games, fc.MAX_LEGAL), dtype=np.int32)
    counts = np.zeros((n_games,), dtype=np.int32)
    priors = np.zeros((n_games, fc.MAX_LEGAL), dtype=np.float32)
    values = np.zeros((n_games,), dtype=np.float32)

    for _ in range(max_steps):
        if engine.done():
            break
        n = engine.collect(states.reshape(-1), idx, counts)
        if n:
            batch_evaluator(n, states, idx, counts, priors, values)
            engine.apply(priors, values)
        else:
            engine.apply(None, None)
    assert engine.done(), "engine did not finish within the step budget"


def _uniform(n, states, idx, counts, priors, values):
    priors[:] = 0.0
    for k in range(n):
        c = int(counts[k])
        priors[k, :c] = 1.0 / max(c, 1)
    values[:n] = 0.0


def test_selfplay_produces_wellformed_examples():
    games = 6
    width = 4
    engine = fc.Engine(
        games_in_flight=width, num_games=games, num_simulations=12,
        max_moves=30, temperature_moves=4, seed=5,
    )
    assert engine.width() == width
    _run_engine(engine, _uniform)

    n = engine.out_count()
    assert n > 0
    states = np.zeros((n, NUM_PLANES, 8, 8), dtype=np.uint8)
    pol_idx = np.zeros((n, fc.MAX_POLICY_TARGETS), dtype=np.uint16)
    pol_val = np.zeros((n, fc.MAX_POLICY_TARGETS), dtype=np.float32)
    pol_len = np.zeros((n,), dtype=np.int16)
    values = np.zeros((n,), dtype=np.float32)
    copied = engine.drain_into(
        states.reshape(-1), pol_idx.reshape(-1), pol_val.reshape(-1),
        pol_len, values,
    )
    assert copied == n
    assert engine.out_count() == 0

    stats = engine.stats()
    assert stats["games"] == games
    assert stats["plies"] == n

    # Policy targets are proper distributions over in-range policy indices.
    sums = pol_val.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-5)
    assert (pol_idx < fc.POLICY_SIZE).all()
    assert (pol_len > 0).all()
    assert set(np.unique(values)).issubset({-1.0, 0.0, 1.0})

    # Every stored state is a valid packed encoding: the constant plane is all
    # ones and the piece planes hold exactly one king per side.
    assert (states[:, 20] == 1).all()
    assert (states[:, 5].reshape(n, -1).sum(axis=1) == 1).all()
    assert (states[:, 11].reshape(n, -1).sum(axis=1) == 1).all()

    # ...and it round-trips into the replay buffer the trainer uses.
    buffer = ReplayBuffer(capacity=n)
    from alpha_chess.batched_selfplay import SelfPlayBatch

    buffer.append(SelfPlayBatch(states, pol_idx, pol_val, pol_len, values))
    assert len(buffer) == n


def test_playout_cap_records_fewer_plies():
    """With playout-cap randomisation on, only full-search plies train."""
    def build(full_prob):
        eng = fc.Engine(
            games_in_flight=4, num_games=8, num_simulations=24,
            fast_simulations=4, full_search_prob=full_prob,
            max_moves=40, seed=9,
        )
        return eng

    full = build(1.0)
    _run_engine(full, _uniform)
    capped = build(0.25)
    _run_engine(capped, _uniform)

    a, b = full.stats(), capped.stats()
    assert a["fast_plies"] == 0
    assert b["fast_plies"] > b["full_plies"]
    # Recorded examples track full-search plies exactly.
    assert a["plies"] == a["full_plies"]
    assert b["plies"] == b["full_plies"]


def test_native_runner_matches_batch_layout():
    """The native driver must hand back exactly what the trainer expects."""
    from alpha_chess.native_selfplay import generate_selfplay_data_native

    net = AlphaZeroNet(channels=8, num_blocks=1)
    device = torch.device("cpu")
    batch = generate_selfplay_data_native(
        net, device, num_games=4, games_in_flight=2, pools=2,
        use_amp=False, num_simulations=8, max_moves=24,
        temperature_moves=2, resign_threshold=None, seed=2, verbose=False,
    )
    assert len(batch) > 0
    assert batch.states.dtype == np.uint8
    assert batch.states.shape[1:] == (NUM_PLANES, 8, 8)
    assert batch.pol_idx.shape[1] == fc.MAX_POLICY_TARGETS
    assert batch.values.shape[0] == len(batch)
    assert batch.stats["games"] == 4


def test_resignation_fires_at_the_default_ply_count():
    """Regression: the default ``resign_plies`` must actually be reachable.

    Consecutive plies alternate the side to move, and in a zero-sum game the
    losing side's -0.95 is always followed by the winner's +0.95. The streak
    used to be a single counter shared by both sides, so it reset on every
    reply, could never exceed 1, and every ``resign_plies`` above 1 was
    unsatisfiable -- resignation never fired at any threshold, in any run.

    A threshold of 1.0 is above every attainable value, so the criterion is met
    on every ply and the only thing under test is whether the streak survives
    the opponent's turn.
    """
    def _uniform(n, states, idx, counts, priors, values):
        for i in range(n):
            c = int(counts[i])
            priors[i, :c] = 1.0 / max(c, 1)
            values[i] = 0.0

    for resign_plies in (1, 2, 3):
        engine = fc.Engine(
            games_in_flight=4, num_games=4, num_simulations=8, max_moves=200,
            resign_threshold=1.0, resign_plies=resign_plies,
            resign_disable_fraction=0.0, seed=5,
        )
        _run_engine(engine, _uniform)
        stats = engine.stats()
        assert stats["resigned"] == 4, (resign_plies, stats)


def test_resignation_is_off_when_no_threshold_is_given():
    def _uniform(n, states, idx, counts, priors, values):
        for i in range(n):
            c = int(counts[i])
            priors[i, :c] = 1.0 / max(c, 1)
            values[i] = 0.0

    engine = fc.Engine(
        games_in_flight=4, num_games=4, num_simulations=8, max_moves=40,
        resign_threshold=None, resign_disable_fraction=0.0, seed=5,
    )
    _run_engine(engine, _uniform)
    assert engine.stats()["resigned"] == 0


def _uniform_eval(n, states, idx, counts, priors, values):
    for i in range(n):
        c = int(counts[i])
        priors[i, :c] = 1.0 / max(c, 1)
        values[i] = 0.0


def _noise_stats(**kwargs):
    engine = fc.Engine(
        games_in_flight=4, num_games=8, num_simulations=8, fast_simulations=2,
        full_search_prob=0.25, max_moves=40, resign_threshold=None, seed=3,
        **kwargs,
    )
    _run_engine(engine, _uniform_eval)
    st = engine.stats()
    assert st["fast_plies"] > 0, st          # playout caps really engaged
    return st


def test_root_noise_defaults_to_recorded_plies_only():
    """The historical behaviour: noise rides along with target recording.

    ``noise_plies`` can fall a little short of ``full_plies`` because a game's
    last ply is counted when it starts but ends terminally before the root is
    expanded and noise is applied. What matters is that it never EXCEEDS the
    recorded plies -- i.e. no fast ply ever gets noise.
    """
    st = _noise_stats()
    assert st["noise_plies"] <= st["full_plies"], st
    assert st["noise_plies"] > 0, st
    # ...and that is the gap: most plies played carry no exploration noise.
    assert st["noise_plies"] < 0.5 * (st["full_plies"] + st["fast_plies"]), st


def test_noise_all_plies_covers_the_plies_that_recording_skips():
    st = _noise_stats(noise_all_plies=1)
    assert st["noise_plies"] > st["full_plies"], st
    assert st["noise_plies"] <= st["full_plies"] + st["fast_plies"], st
    # Nearly everything played is now explored, not a quarter of it.
    assert st["noise_plies"] > 0.9 * (st["full_plies"] + st["fast_plies"]), st


def test_noise_all_plies_leaves_the_recording_budget_alone():
    """It must move exploration only.

    The games themselves necessarily differ -- that is the point of the flag --
    so the invariant is the *share* of plies that record a target, not the
    counts, which shift with the games that get played.
    """
    for kwargs in ({}, {"noise_all_plies": 1}):
        st = _noise_stats(**kwargs)
        share = st["full_plies"] / (st["full_plies"] + st["fast_plies"])
        assert 0.15 < share < 0.4, (kwargs, share, st)
        # Recorded examples still track full-search plies exactly.
        assert st["plies"] == st["full_plies"], st


def test_dirichlet_epsilon_changes_the_games_that_are_played():
    """Noise weight must reach the search, not just the constructor."""
    def run(eps):
        e = fc.Engine(
            games_in_flight=4, num_games=4, num_simulations=16, max_moves=60,
            full_search_prob=1.0, dirichlet_epsilon=eps, resign_threshold=None,
            seed=9,
        )
        _run_engine(e, _uniform_eval)
        return e.stats()["plies"]

    # Same seed, same evaluator: only the noise weight differs. With uniform
    # priors, no noise makes selection tie-break identically every game.
    assert run(0.0) != run(0.95)
