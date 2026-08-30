from __future__ import annotations

from typing import Optional

import numpy as np
import chess

NUM_PLANES = 19
POLICY_SIZE = 4672  # 64 * 73

# ---------------------------------------------------------------------------
# Move-encoding tables
# ---------------------------------------------------------------------------

# Queen-like sliding directions, indexed by direction d in 0..7 as (df, dr).
DIRECTIONS = [
    (0, 1),    # 0 N
    (1, 1),    # 1 NE
    (1, 0),    # 2 E
    (1, -1),   # 3 SE
    (0, -1),   # 4 S
    (-1, -1),  # 5 SW
    (-1, 0),   # 6 W
    (-1, 1),   # 7 NW
]
_DIRECTION_TO_D = {v: i for i, v in enumerate(DIRECTIONS)}

# Knight offsets for planes 56..63, indexed by (plane - 56) as (df, dr).
KNIGHT_OFFSETS = [
    (1, 2),    # 56
    (2, 1),    # 57
    (2, -1),   # 58
    (1, -2),   # 59
    (-1, -2),  # 60
    (-2, -1),  # 61
    (-2, 1),   # 62
    (-1, 2),   # 63
]
_KNIGHT_OFFSET_TO_PLANE = {off: 56 + i for i, off in enumerate(KNIGHT_OFFSETS)}

# Underpromotion piece indices.
_UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]  # idx 0,1,2
_UNDERPROMO_PIECE_TO_IDX = {p: i for i, p in enumerate(_UNDERPROMO_PIECES)}


def _sign(x: int) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Board encoding
# ---------------------------------------------------------------------------

# Order of piece types for planes 0-5 (white) and 6-11 (black).
_PIECE_ORDER = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a chess.Board into a (19, 8, 8) float32 array (absolute orientation)."""
    arr = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        row = chess.square_rank(square)
        col = chess.square_file(square)
        idx = _PIECE_ORDER.index(piece.piece_type)
        plane = idx if piece.color == chess.WHITE else idx + 6
        arr[plane, row, col] = 1.0

    # Plane 12: side to move.
    if board.turn == chess.WHITE:
        arr[12, :, :] = 1.0

    # Planes 13-16: castling rights.
    if board.has_kingside_castling_rights(chess.WHITE):
        arr[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        arr[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        arr[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        arr[16, :, :] = 1.0

    # Plane 17: en-passant target square.
    if board.ep_square is not None:
        row = chess.square_rank(board.ep_square)
        col = chess.square_file(board.ep_square)
        arr[17, row, col] = 1.0

    # Plane 18: halfmove clock.
    arr[18, :, :] = board.halfmove_clock / 100.0

    return arr


# ---------------------------------------------------------------------------
# Move <-> index
# ---------------------------------------------------------------------------

def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """Map a move to its policy index [0, POLICY_SIZE)."""
    from_sq = move.from_square
    to_sq = move.to_square
    from_file = chess.square_file(from_sq)
    from_rank = chess.square_rank(from_sq)
    to_file = chess.square_file(to_sq)
    to_rank = chess.square_rank(to_sq)
    df = to_file - from_file
    dr = to_rank - from_rank

    # Underpromotion (to knight/bishop/rook).
    if move.promotion is not None and move.promotion != chess.QUEEN:
        piece_idx = _UNDERPROMO_PIECE_TO_IDX[move.promotion]
        plane = 64 + (df + 1) * 3 + piece_idx
        return from_sq * 73 + plane

    # Knight moves.
    if (df, dr) in _KNIGHT_OFFSET_TO_PLANE:
        plane = _KNIGHT_OFFSET_TO_PLANE[(df, dr)]
        return from_sq * 73 + plane

    # Queen-like moves (includes queen promotions, king moves, castling, ep).
    d = _DIRECTION_TO_D[(_sign(df), _sign(dr))]
    n = max(abs(df), abs(dr))
    plane = d * 7 + (n - 1)
    return from_sq * 73 + plane


def index_to_move(index: int, board: chess.Board) -> Optional[chess.Move]:
    """Reconstruct a move from its index; return it only if legal on board."""
    from_sq = index // 73
    plane = index % 73
    from_file = chess.square_file(from_sq)
    from_rank = chess.square_rank(from_sq)

    promotion = None

    if plane < 56:
        # Queen-like.
        d = plane // 7
        n = (plane % 7) + 1
        df, dr = DIRECTIONS[d]
        to_file = from_file + df * n
        to_rank = from_rank + dr * n
    elif plane < 64:
        # Knight.
        df, dr = KNIGHT_OFFSETS[plane - 56]
        to_file = from_file + df
        to_rank = from_rank + dr
    else:
        # Underpromotion.
        rel = plane - 64
        dd = rel // 3 - 1
        piece_idx = rel % 3
        promotion = _UNDERPROMO_PIECES[piece_idx]
        piece = board.piece_at(from_sq)
        if piece is not None and piece.color == chess.WHITE:
            forward = 1
        else:
            forward = -1
        to_file = from_file + dd
        to_rank = from_rank + forward

    if not (0 <= to_file < 8 and 0 <= to_rank < 8):
        return None

    to_sq = chess.square(to_file, to_rank)

    # Determine queen promotion for queen-like pawn moves reaching last rank.
    if promotion is None and plane < 56:
        piece = board.piece_at(from_sq)
        if piece is not None and piece.piece_type == chess.PAWN:
            if (piece.color == chess.WHITE and to_rank == 7) or (
                piece.color == chess.BLACK and to_rank == 0
            ):
                promotion = chess.QUEEN

    move = chess.Move(from_sq, to_sq, promotion=promotion)
    if move in board.legal_moves:
        return move
    return None
