from __future__ import annotations

"""Board/move encoding for the AlphaChess network.

The encoding is **side-to-move relative**: the board is vertically flipped when
Black is to move, so the network always sees the player to move at the bottom of
the board moving "up".  This halves what the network has to learn (it never has
to represent the same motif twice, once per colour) and is the standard
AlphaZero/Leela convention.

The flip is an implementation detail of this module: :func:`encode_board`,
:func:`move_to_index` and :func:`index_to_move` all take the ``board`` and apply
(or undo) the orientation themselves, so callers work in ordinary absolute
``chess`` coordinates throughout.

Plane layout (21 planes, all from the side-to-move's perspective)::

     0- 5  our    pawn, knight, bishop, rook, queen, king
     6-11  their  pawn, knight, bishop, rook, queen, king
    12     our kingside castling right
    13     our queenside castling right
    14     their kingside castling right
    15     their queenside castling right
    16     en-passant target square
    17     halfmove clock / 100
    18     this position has occurred at least once before
    19     this position has occurred at least twice before
    20     constant ones (lets padded convolutions locate the board edge)
"""

from typing import Optional

import numpy as np
import chess

NUM_PLANES = 21
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

# Plane 20 is a constant; sharing one immutable row avoids rebuilding it.
_ONES_PLANE = np.ones((8, 8), dtype=np.float32)


def _piece_planes(board: chess.Board) -> np.ndarray:
    """Return the 12 (ours-then-theirs) piece planes as ``(12, 8, 8)`` float32.

    Built straight from python-chess's bitboards: the twelve 64-bit masks are
    concatenated into 96 little-endian bytes and expanded in one
    :func:`numpy.unpackbits` call.  Bit ``i`` of a mask is square ``i``, so the
    unpacked bits reshape directly to ``[plane, rank, file]``.  This replaces a
    per-piece Python loop over ``board.piece_map()``.
    """
    if board.turn == chess.WHITE:
        ours = board.occupied_co[chess.WHITE]
        theirs = board.occupied_co[chess.BLACK]
    else:
        ours = board.occupied_co[chess.BLACK]
        theirs = board.occupied_co[chess.WHITE]

    pawns, knights = board.pawns, board.knights
    bishops, rooks = board.bishops, board.rooks
    queens, kings = board.queens, board.kings

    raw = b"".join(
        (bb & occ).to_bytes(8, "little")
        for occ in (ours, theirs)
        for bb in (pawns, knights, bishops, rooks, queens, kings)
    )
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")
    return bits.reshape(12, 8, 8).astype(np.float32)


def encode_board(board: chess.Board, rep_count: Optional[int] = None) -> np.ndarray:
    """Encode ``board`` into a ``(NUM_PLANES, 8, 8)`` float32 array.

    The result is side-to-move relative: when Black is to move the board is
    vertically flipped so the mover always advances toward increasing rank.

    ``rep_count`` is how many times this exact position has already occurred
    earlier in the game (0 when it is new).  Self-play tracks this incrementally
    and passes it in; when omitted it is derived from ``board`` directly, which
    is much slower and is only intended for one-off calls (GUI, analysis).
    """
    arr = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    arr[0:12] = _piece_planes(board)

    white_to_move = board.turn == chess.WHITE
    us = chess.WHITE if white_to_move else chess.BLACK
    them = chess.BLACK if white_to_move else chess.WHITE

    # Planes 12-15: castling rights, ours first.
    if board.has_kingside_castling_rights(us):
        arr[12] = 1.0
    if board.has_queenside_castling_rights(us):
        arr[13] = 1.0
    if board.has_kingside_castling_rights(them):
        arr[14] = 1.0
    if board.has_queenside_castling_rights(them):
        arr[15] = 1.0

    # Plane 16: en-passant target square (in absolute coords for now; the
    # whole board is flipped below in one go when Black is to move).
    if board.ep_square is not None:
        arr[16, chess.square_rank(board.ep_square), chess.square_file(board.ep_square)] = 1.0

    # Plane 17: halfmove clock, scaled into roughly [0, 1].
    arr[17] = min(board.halfmove_clock, 100) / 100.0

    # Planes 18-19: repetition indicators.
    if rep_count is None:
        rep_count = _derive_rep_count(board)
    if rep_count >= 1:
        arr[18] = 1.0
    if rep_count >= 2:
        arr[19] = 1.0

    # Plane 20: constant ones.
    arr[20] = _ONES_PLANE

    if not white_to_move:
        # Vertical flip of every spatial plane. The constant/broadcast planes
        # are unaffected by this, so flipping the whole stack is safe.
        arr = arr[:, ::-1, :].copy()

    return arr


def _derive_rep_count(board: chess.Board) -> int:
    """How many times the current position occurred before (slow fallback).

    Only used when a caller does not track repetitions itself.  Costs a scan of
    the move stack, so self-play passes ``rep_count`` explicitly instead.
    """
    if not board.move_stack:
        return 0
    if board.is_repetition(3):
        return 2
    if board.is_repetition(2):
        return 1
    return 0


# ---------------------------------------------------------------------------
# Move <-> index
# ---------------------------------------------------------------------------

def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """Map a move to its policy index ``[0, POLICY_SIZE)``.

    Squares are taken in the side-to-move-relative frame used by
    :func:`encode_board`, so a position and its colour-mirror map identical
    moves to identical indices.
    """
    from_sq = move.from_square
    to_sq = move.to_square
    if board.turn != chess.WHITE:
        # Vertical flip; ``sq ^ 56`` maps rank r to rank 7-r, file unchanged.
        from_sq ^= 56
        to_sq ^= 56

    from_file = from_sq & 7
    from_rank = from_sq >> 3
    df = (to_sq & 7) - from_file
    dr = (to_sq >> 3) - from_rank

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
    """Reconstruct a move from its index; return it only if legal on ``board``.

    Inverse of :func:`move_to_index`, including the side-to-move orientation
    flip, so the returned move is in absolute coordinates.
    """
    from_sq = index // 73
    plane = index % 73
    from_file = from_sq & 7
    from_rank = from_sq >> 3

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
        # Underpromotion. In the relative frame the mover always advances one
        # rank "up", so the forward direction is unconditionally +1.
        rel = plane - 64
        to_file = from_file + (rel // 3 - 1)
        to_rank = from_rank + 1
        promotion = _UNDERPROMO_PIECES[rel % 3]

    if not (0 <= to_file < 8 and 0 <= to_rank < 8):
        return None

    to_sq = to_rank * 8 + to_file

    # Queen promotion for a queen-like pawn move reaching the last rank. The
    # relative frame puts our promotion rank at 7 for both colours.
    if promotion is None and plane < 56 and to_rank == 7:
        abs_from = from_sq if board.turn == chess.WHITE else from_sq ^ 56
        piece = board.piece_at(abs_from)
        if piece is not None and piece.piece_type == chess.PAWN:
            promotion = chess.QUEEN

    if board.turn != chess.WHITE:
        from_sq ^= 56
        to_sq ^= 56

    move = chess.Move(from_sq, to_sq, promotion=promotion)
    if move in board.legal_moves:
        return move
    return None


# ---------------------------------------------------------------------------
# Compact storage
# ---------------------------------------------------------------------------

# Every plane is binary except plane 17 (halfmove clock / 100), so encoded
# states round-trip losslessly through uint8 once that plane is rescaled to its
# raw 0..100 counter. Storing states this way costs 1.3 KB instead of 5.4 KB,
# which is what lets the replay buffer hold millions of positions.
STORE_SCALE = np.ones((NUM_PLANES, 1, 1), dtype=np.float32)
STORE_SCALE[17] = 100.0
LOAD_SCALE = (1.0 / STORE_SCALE).astype(np.float32)


def pack_state(arr: np.ndarray) -> np.ndarray:
    """Compress an encoded state (or batch of states) to uint8."""
    return np.rint(arr * STORE_SCALE).astype(np.uint8)


def unpack_state(arr: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pack_state`."""
    return arr.astype(np.float32) * LOAD_SCALE
