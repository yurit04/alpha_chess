from __future__ import annotations

import random

import numpy as np
import chess
import pytest

from alpha_chess.encoding import (
    NUM_PLANES,
    POLICY_SIZE,
    encode_board,
    move_to_index,
    index_to_move,
    pack_state,
    unpack_state,
)


def test_constants():
    assert NUM_PLANES == 21
    assert POLICY_SIZE == 4672
    assert POLICY_SIZE == 64 * 73


def test_encode_board_shape_dtype():
    board = chess.Board()
    arr = encode_board(board)
    assert arr.shape == (NUM_PLANES, 8, 8)
    assert arr.dtype == np.float32
    # Our king on e1 -> plane 5, rank 0, file 4; their king on e8 -> plane 11.
    assert arr[5, 0, 4] == 1.0
    assert arr[11, 7, 4] == 1.0
    # Our pawns fill rank 2.
    assert arr[0, 1, :].sum() == 8
    # Castling rights all available at the start.
    for p in (12, 13, 14, 15):
        assert np.all(arr[p] == 1.0)
    # No repetitions yet; the constant plane is all ones.
    assert np.all(arr[18] == 0.0) and np.all(arr[19] == 0.0)
    assert np.all(arr[20] == 1.0)


def test_encoding_is_side_to_move_relative():
    """A position and its colour mirror must encode identically."""
    board = chess.Board()
    for san in ("e4", "c5", "Nf3", "d6", "d4", "cxd4"):
        board.push_san(san)
    mirrored = board.mirror()
    assert np.array_equal(
        encode_board(board, rep_count=0), encode_board(mirrored, rep_count=0)
    )
    # ...and corresponding moves must land on the same policy index.
    for move in board.legal_moves:
        twin = chess.Move(
            move.from_square ^ 56, move.to_square ^ 56, promotion=move.promotion
        )
        assert move_to_index(move, board) == move_to_index(twin, mirrored)


def test_repetition_planes():
    board = chess.Board()
    assert np.all(encode_board(board, rep_count=0)[18] == 0.0)
    assert np.all(encode_board(board, rep_count=1)[18] == 1.0)
    assert np.all(encode_board(board, rep_count=1)[19] == 0.0)
    assert np.all(encode_board(board, rep_count=2)[19] == 1.0)


def test_halfmove_clock_plane_and_packing():
    board = chess.Board("8/8/4k3/8/8/4K3/8/8 w - - 37 90")
    arr = encode_board(board, rep_count=0)
    assert np.allclose(arr[17], 0.37)
    packed = pack_state(arr)
    assert packed.dtype == np.uint8
    assert np.allclose(unpack_state(packed), arr, atol=1e-6)


def _check_position(board: chess.Board):
    seen = set()
    for move in board.legal_moves:
        idx = move_to_index(move, board)
        assert 0 <= idx < POLICY_SIZE, (board.fen(), move.uci(), idx)
        assert idx not in seen, (
            "duplicate index",
            board.fen(),
            move.uci(),
            idx,
        )
        seen.add(idx)
        rt = index_to_move(idx, board)
        assert rt == move, (board.fen(), move.uci(), idx, rt.uci() if rt else None)


def test_opening_position():
    _check_position(chess.Board())


def test_random_playouts():
    rng = random.Random(1234)
    for _ in range(20):
        board = chess.Board()
        for _ply in range(20):
            _check_position(board)
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(rng.choice(legal))


HANDPICKED_FENS = [
    # White pawn on 7th rank ready to promote (queen + underpromotions).
    "8/P7/8/8/8/8/8/k1K5 w - - 0 1",
    # Black pawn on 2nd rank ready to promote.
    "K1k5/8/8/8/8/8/p7/8 b - - 0 1",
    # Promotion with captures available (underpromotion file deltas).
    "1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
    "rnbqk3/1P6/8/8/8/8/8/4K3 w q - 0 1",
    # Full castling availability for both sides.
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    # En-passant available.
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    # Black en-passant.
    "rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3",
    # A middlegame-ish position with many piece types.
    "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w - - 6 6",
]


@pytest.mark.parametrize("fen", HANDPICKED_FENS)
def test_handpicked_positions(fen):
    board = chess.Board(fen)
    _check_position(board)


def test_underpromotion_indices_present():
    # Ensure underpromotions actually map into the 64-72 plane range.
    board = chess.Board("8/P7/8/8/8/8/8/k1K5 w - - 0 1")
    planes = set()
    for move in board.legal_moves:
        if move.promotion is not None and move.promotion != chess.QUEEN:
            idx = move_to_index(move, board)
            planes.add(idx % 73)
    assert planes  # some underpromotions exist
    assert all(64 <= p < 73 for p in planes)
