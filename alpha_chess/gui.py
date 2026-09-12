"""Pygame GUI for playing against (or getting advice from) the AlphaChess agent.

Modes:

* **play**  -- click to move against the agent (H=hint, U=undo, R=redo,
  M=mute, ...).
* **advisor** -- a variant of play mode (``mode == "play"`` with
  ``advisor=True``): the user makes every move for BOTH colours -- mirroring a
  game played in a separate application -- and asks the engine for the best
  move on demand (H / SPACE). The engine NEVER moves a piece on its own; it
  only recommends. This is what the ``analyze`` command opens.
* **setup** -- an interactive board editor: place/remove pieces with a palette,
  set the side to move and castling rights, then either analyze the position or
  adopt it (P) to play/advise from. Reachable via E; used to set up a mid-game
  position when joining a game in progress.

Importing this module does NOT open a window; only :func:`launch_gui`
initializes the pygame display. The drawing is structured so a single frame
can be rendered onto an offscreen :class:`pygame.Surface` (see
:class:`_GuiApp`) without running the interactive loop -- this is what the
headless self-test relies on.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import chess


# ---------------------------------------------------------------------------
# Layout / drawing constants
# ---------------------------------------------------------------------------
MARGIN = 24                        # coordinate border around the 8x8 grid
SQUARE = 74                        # size of a single square
GRID = SQUARE * 8                  # 8x8 playing area (592 px)
BOARD_PX = GRID + 2 * MARGIN       # board region incl. margins (640 px)
PANEL_PX = 240                     # width of the side panel
WINDOW_W = BOARD_PX + PANEL_PX
WINDOW_H = BOARD_PX
FPS = 30

PALETTE_CELL = 30                  # size of a palette cell in setup mode

# Colours (R, G, B)
LIGHT_SQ = (235, 210, 173)
DARK_SQ = (150, 110, 74)
BOARD_BG = (34, 33, 38)            # frame/margin behind the board
FRAME_LINE = (16, 15, 18)          # thin line around the grid
COORD_COLOR = (206, 197, 184)      # coordinate labels drawn in the margin
SEL_COLOR = (246, 232, 96)         # selected square highlight
DEST_COLOR = (90, 150, 70)         # legal-destination marker
HINT_COLOR = (74, 144, 226)        # hint / analysis from-to highlight
LASTMOVE_COLOR = (54, 211, 92)     # border around the piece that just moved
LASTMOVE_WIDTH = 5                 # thickness of that border, in pixels
PANEL_BG = (40, 40, 45)
PANEL_FG = (230, 230, 230)
PANEL_DIM = (170, 170, 175)
PANEL_ACCENT = (250, 220, 120)     # setup accent / brush highlight
PANEL_INFO = (170, 200, 240)       # suggestions / hint text
PANEL_ERR = (240, 160, 160)        # error text
PALETTE_BG = (60, 60, 66)
PALETTE_EDGE = (90, 90, 96)
WHITE_PIECE = (250, 250, 250)
BLACK_PIECE = (20, 20, 20)
CIRCLE_WHITE = (245, 245, 245)
CIRCLE_BLACK = (60, 60, 60)

# We always draw the *solid* (filled) glyph shapes for both colours and
# distinguish white vs black by fill colour + outline. Solid shapes read far
# better than the hollow "white" outline glyphs, especially on light squares.
# These are the filled pieces U+265A..U+265F, keyed by piece type letter.
SOLID_GLYPHS = {"K": "♚", "Q": "♛", "R": "♜", "B": "♝", "N": "♞", "P": "♟"}

# Fonts to try (in order) for rendering the chess glyphs. Names are resolved via
# SysFont; absolute paths (loaded via pygame.font.Font) are robust fallbacks for
# platforms where the family name does not resolve to a glyph-bearing font.
GLYPH_FONTS = [
    "Apple Symbols",            # macOS – has the chess glyphs
    "DejaVu Sans",              # Linux
    "Arial Unicode MS",         # some Windows/macOS installs
    "Segoe UI Symbol",          # Windows
    "Menlo",                    # macOS fallback
    "/System/Library/Fonts/Apple Symbols.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

# A codepoint essentially guaranteed to be absent from any font, so its rendered
# surface is the "tofu"/missing-glyph box. Comparing a real glyph's pixels to
# this tells us whether the font actually contains the glyph.
_MISSING_GLYPH = "\U000F0000"


# Sample rate for the synthesised move sounds.
_AUDIO_HZ = 44100


def _click_samples(numpy, freq: float, ms: int, decay: float,
                   noise: float, gain: float):
    """Synthesise a short wooden click as an int16 stereo array.

    A piece landing on a board is a percussive transient, not a tone: a burst
    of noise for the contact plus a fast-decaying low sine for the body of the
    board. Generating it here keeps the package free of binary audio assets.
    """
    n = int(_AUDIO_HZ * ms / 1000)
    t = numpy.arange(n, dtype=numpy.float64) / _AUDIO_HZ
    envelope = numpy.exp(-decay * t)
    # Deterministic noise, so the click sounds the same every time.
    rng = numpy.random.default_rng(0)
    body = numpy.sin(2.0 * numpy.pi * freq * t)
    wave = (1.0 - noise) * body + noise * rng.uniform(-1.0, 1.0, n)
    wave *= envelope * gain
    # A couple of ms of fade-in stops the attack from clipping into a pop.
    attack = min(n, int(_AUDIO_HZ * 0.002))
    if attack:
        wave[:attack] *= numpy.linspace(0.0, 1.0, attack)
    mono = numpy.clip(wave, -1.0, 1.0) * 32767.0
    return numpy.ascontiguousarray(
        numpy.column_stack([mono, mono]).astype(numpy.int16))


class _SoundBank:
    """Move/capture click sounds, degrading to silence when unavailable.

    Audio is entirely optional: a machine with no sound device (or a headless
    test run) must still play chess, so every step is guarded and any failure
    leaves :attr:`enabled` False rather than raising.
    """

    def __init__(self, pygame) -> None:
        self.pygame = pygame
        self.enabled = False
        self.muted = False
        self._sounds = {}
        try:
            import numpy

            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=_AUDIO_HZ, size=-16, channels=2)
            if not pygame.mixer.get_init():
                return
            make = pygame.sndarray.make_sound
            self._sounds = {
                # A crisp tap for a quiet move...
                "move": make(_click_samples(
                    numpy, freq=880.0, ms=55, decay=70.0,
                    noise=0.45, gain=0.35)),
                # ...and a lower, fuller knock when wood hits wood.
                "capture": make(_click_samples(
                    numpy, freq=520.0, ms=85, decay=45.0,
                    noise=0.6, gain=0.5)),
            }
            self.enabled = True
        except Exception:  # pragma: no cover - platform/audio specific
            self.enabled = False
            self._sounds = {}

    def play(self, name: str) -> None:
        """Play a named click; a no-op when muted or unavailable."""
        if not self.enabled or self.muted:
            return
        sound = self._sounds.get(name)
        if sound is None:
            return
        try:
            sound.play()
        except Exception:  # pragma: no cover - platform/audio specific
            pass

    def for_move(self, board, move) -> str:
        """Pick the click for ``move``, which must not yet be pushed."""
        try:
            if board.is_capture(move):
                return "capture"
        except Exception:  # pragma: no cover - defensive
            pass
        return "move"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _square_to_screen(square: int, flipped: bool) -> Tuple[int, int]:
    """Return the top-left (x, y) pixel of the given board square.

    The 8x8 grid is inset by MARGIN on all sides to leave room for the
    coordinate labels drawn in the surrounding border.
    """
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    if flipped:
        col = 7 - file
        row = rank                 # rank1 at top when flipped
    else:
        col = file
        row = 7 - rank             # rank8 at top (standard orientation)
    return MARGIN + col * SQUARE, MARGIN + row * SQUARE


def _screen_to_square(pos: Tuple[int, int], flipped: bool) -> Optional[int]:
    """Map a screen pixel to a board square, or None if outside the grid."""
    x, y = pos
    gx, gy = x - MARGIN, y - MARGIN
    if gx < 0 or gx >= GRID or gy < 0 or gy >= GRID:
        return None
    col = gx // SQUARE
    row = gy // SQUARE
    if flipped:
        file = 7 - col
        rank = row
    else:
        file = col
        rank = 7 - row
    return chess.square(int(file), int(rank))


def _draw_board_backdrop(pygame, screen, flipped: bool) -> None:
    """Fill the board region with the frame colour and draw the 8x8 squares.

    The frame (MARGIN border) is where coordinates go, so nothing is ever
    drawn on top of a piece.
    """
    # Frame / margin backdrop for the whole board region.
    pygame.draw.rect(screen, BOARD_BG, (0, 0, BOARD_PX, BOARD_PX))
    # The 8x8 squares.
    for square in chess.SQUARES:
        x, y = _square_to_screen(square, flipped)
        is_light = (chess.square_file(square) + chess.square_rank(square)) % 2 == 1
        pygame.draw.rect(screen, LIGHT_SQ if is_light else DARK_SQ,
                         (x, y, SQUARE, SQUARE))
    # A thin crisp line framing the grid.
    pygame.draw.rect(screen, FRAME_LINE,
                     (MARGIN - 1, MARGIN - 1, GRID + 2, GRID + 2), 1)


def _fill_square(pygame, screen, square: int, flipped: bool, rgba) -> None:
    """Blit a translucent colour over a square (rgba = (r, g, b, alpha))."""
    x, y = _square_to_screen(square, flipped)
    overlay = pygame.Surface((SQUARE, SQUARE), pygame.SRCALPHA)
    overlay.fill(rgba)
    screen.blit(overlay, (x, y))


def _draw_coordinates(pygame, screen, flipped: bool, font) -> None:
    """Draw file letters (a-h) and rank numbers (1-8) in the board margin.

    Labels live in the MARGIN border around the grid -- never over a square --
    on all four sides. Flip-aware, so every label matches the file/rank of the
    grid line it sits against.
    """
    for i in range(8):
        # Column i (screen) -> its file; row i (screen) -> its rank.
        file = (7 - i) if flipped else i
        rank = i if flipped else (7 - i)
        letter = font.render(chess.FILE_NAMES[file], True, COORD_COLOR)
        number = font.render(str(rank + 1), True, COORD_COLOR)
        col_cx = MARGIN + i * SQUARE + SQUARE // 2
        row_cy = MARGIN + i * SQUARE + SQUARE // 2
        # Files on the top and bottom margins.
        screen.blit(letter, (col_cx - letter.get_width() // 2,
                             (MARGIN - letter.get_height()) // 2))
        screen.blit(letter, (col_cx - letter.get_width() // 2,
                             MARGIN + GRID + (MARGIN - letter.get_height()) // 2))
        # Ranks on the left and right margins.
        screen.blit(number, ((MARGIN - number.get_width()) // 2,
                             row_cy - number.get_height() // 2))
        screen.blit(number, (MARGIN + GRID + (MARGIN - number.get_width()) // 2,
                             row_cy - number.get_height() // 2))


def _glyph_bytes(font, glyph: str):
    """Return the raw RGBA pixels of a rendered glyph, or None on failure."""
    import pygame
    try:
        surf = font.render(glyph, True, (0, 0, 0))
    except Exception:
        return None
    if surf.get_width() <= 0 or surf.get_height() <= 0:
        return None
    return pygame.image.tostring(surf, "RGBA")


def _glyph_renders(font, glyph: str) -> bool:
    """True only if the font really has the glyph (not the tofu/missing box).

    A missing glyph still renders a non-empty box, so we compare the glyph's
    pixels against a guaranteed-absent codepoint: if they match, it's tofu.
    """
    real = _glyph_bytes(font, glyph)
    if real is None:
        return False
    tofu = _glyph_bytes(font, _MISSING_GLYPH)
    return real != tofu


class _Renderer:
    """Bundles fonts and knows how to draw one piece (glyph or fallback).

    A piece can be drawn at an arbitrary centre and size via
    :meth:`draw_piece_at`; :meth:`draw_piece` is the square-geometry
    convenience used by the board. Glyph and fallback fonts are cached per
    pixel size so the (relatively expensive) glyph-font detection runs once.
    """

    def __init__(self, pygame) -> None:
        self.pygame = pygame
        self.has_glyphs = False
        self._glyph_spec: Optional[str] = None      # font name/path that works
        self._glyph_cache: Dict[int, object] = {}   # size -> font
        self._fallback_cache: Dict[int, object] = {}
        self._detect_glyph_font()

    def _load_font(self, name: str, size: int):
        """Load a font by family name (SysFont) or by absolute .ttf path."""
        pygame = self.pygame
        try:
            if name.endswith((".ttf", ".otf", ".ttc")) or name.startswith("/"):
                import os
                if not os.path.exists(name):
                    return None
                return pygame.font.Font(name, size)
            return pygame.font.SysFont(name, size)
        except Exception:
            return None

    def _detect_glyph_font(self) -> None:
        """Find (once) a font family/path that renders all six chess glyphs."""
        base = int(SQUARE * 0.8)
        for name in GLYPH_FONTS:
            font = self._load_font(name, base)
            if font is None:
                continue
            # Require ALL six solid glyphs to render, not just the king.
            if all(_glyph_renders(font, g) for g in SOLID_GLYPHS.values()):
                self.has_glyphs = True
                self._glyph_spec = name
                self._glyph_cache[base] = font
                return
        # No glyph font found anywhere: fall back to drawn pieces.
        self.has_glyphs = False
        self._glyph_spec = None

    def _glyph_font(self, size: int):
        """Return a cached glyph font at the given pixel size."""
        font = self._glyph_cache.get(size)
        if font is None:
            if self._glyph_spec is not None:
                font = self._load_font(self._glyph_spec, size)
            if font is None:
                font = self.pygame.font.SysFont(None, size)
            self._glyph_cache[size] = font
        return font

    def _fallback_font(self, size: int):
        """Return a cached bold letter font at the given pixel size."""
        font = self._fallback_cache.get(size)
        if font is None:
            font = self.pygame.font.SysFont("Arial", size, bold=True)
            self._fallback_cache[size] = font
        return font

    def draw_piece_at(self, surface, piece: chess.Piece,
                      cx: int, cy: int, size: int) -> None:
        """Draw ``piece`` centred at (cx, cy), scaled to ``size`` pixels."""
        pygame = self.pygame
        symbol = piece.symbol()
        if self.has_glyphs:
            font = self._glyph_font(max(8, int(size * 0.8)))
            glyph = SOLID_GLYPHS[symbol.upper()]
            is_white = piece.color == chess.WHITE
            fill = WHITE_PIECE if is_white else BLACK_PIECE
            outline = BLACK_PIECE if is_white else WHITE_PIECE
            base = font.render(glyph, True, fill)
            rect = base.get_rect(center=(cx, cy))
            # Stroke: blit the outline-coloured glyph at 8 small offsets first.
            stroke = font.render(glyph, True, outline)
            off = max(1, int(size / 40))
            for dx, dy in ((-off, 0), (off, 0), (0, -off), (0, off),
                           (-off, -off), (off, -off), (-off, off), (off, off)):
                surface.blit(stroke, stroke.get_rect(center=(cx + dx, cy + dy)))
            surface.blit(base, rect)
        else:
            # Fallback: coloured circle with the piece letter.
            radius = int(size * 0.4)
            fill = CIRCLE_WHITE if piece.color == chess.WHITE else CIRCLE_BLACK
            pygame.draw.circle(surface, fill, (cx, cy), radius)
            pygame.draw.circle(surface, (0, 0, 0), (cx, cy), radius, 2)
            txt_color = BLACK_PIECE if piece.color == chess.WHITE else WHITE_PIECE
            letter = symbol.upper() if piece.color == chess.WHITE else symbol.lower()
            font = self._fallback_font(max(8, int(size * 0.45)))
            surf = font.render(letter, True, txt_color)
            surface.blit(surf, surf.get_rect(center=(cx, cy)))

    def draw_piece(self, surface, piece: chess.Piece, x: int, y: int) -> None:
        """Draw a piece centred inside the board square at (x, y)."""
        self.draw_piece_at(
            surface, piece, x + SQUARE // 2, y + SQUARE // 2, SQUARE
        )


# ---------------------------------------------------------------------------
# Application state / drawing / event handling
# ---------------------------------------------------------------------------
class _GuiApp:
    """Holds all interactive state and knows how to draw a single frame.

    The app can be constructed against any :class:`pygame.Surface` (a live
    display surface for :func:`launch_gui`, or an offscreen surface for the
    headless self-test). :meth:`draw` renders one frame; :meth:`run` drives
    the interactive event loop.
    """

    def __init__(
        self,
        pygame,
        surface,
        agent=None,
        agent_error: Optional[str] = None,
        human_is_white: bool = True,
        simulations: int = 200,
        start_in_setup: bool = False,
        initial_fen: Optional[str] = None,
        has_display: bool = True,
        advisor: bool = False,
    ) -> None:
        self.pygame = pygame
        self.surface = surface
        self.has_display = has_display
        self.agent = agent
        self.agent_error = agent_error
        self.human_is_white = human_is_white
        self.simulations = simulations
        # Advisor mode: the user makes every move for BOTH colours (mirroring an
        # external game) and only asks the engine for suggestions -- the engine
        # never moves a piece on its own. This is a variant of "play" mode.
        self.advisor = advisor

        self.renderer = _Renderer(pygame)
        self.panel_font = pygame.font.SysFont("Arial", 20)
        self.panel_small = pygame.font.SysFont("Arial", 15)
        self.panel_tiny = pygame.font.SysFont("Arial", 13)
        self.panel_bold = pygame.font.SysFont("Arial", 22, bold=True)
        self.coord_font = pygame.font.SysFont("Arial", 14, bold=True)

        # ----- play-mode state -----
        self._fen_error: Optional[str] = None
        if initial_fen:
            try:
                self.board = chess.Board(initial_fen)
            except ValueError as exc:
                # A malformed FEN must not crash the window; fall back to the
                # standard start position and surface the problem.
                self.board = chess.Board()
                self._fen_error = "Invalid FEN: %s" % exc
        else:
            self.board = chess.Board()
        self.flipped = not human_is_white
        self.selected: Optional[int] = None
        self.legal_dests: List[int] = []
        self.hint_squares: Optional[Tuple[int, int]] = None
        self.status_msg = ""
        self.eval_msg = ""
        self.hint_info: List[str] = []
        self.thinking = False
        # Plies taken back by Undo, newest first, so Redo can replay them.
        # Cleared whenever a move is played onto a different line.
        self.redo_stack: List[chess.Move] = []
        # Move sounds. Skipped entirely without a display, so headless
        # rendering and tests never touch an audio device.
        self.sounds = _SoundBank(pygame) if has_display else None

        # ----- setup-mode state -----
        self.mode = "play"
        self.setup_pieces: Dict[int, chess.Piece] = {}
        self.setup_turn = chess.WHITE
        self.setup_castling_cleared = False
        self.brush: Optional[str] = None          # symbol str, "eraser", or None
        self.setup_status = ""
        self.suggest_lines: List[str] = []
        self._status_is_error = False

        self._build_palette_cells()

        if start_in_setup:
            self._enter_setup()
        else:
            self._refresh_status()

        # Surface a bad initial FEN (window already fell back to start position).
        if self._fen_error:
            self.setup_status = self._fen_error
            self._status_is_error = True
            if self.agent_error is None:
                self.agent_error = self._fen_error

    # ------------------------------------------------------------------ setup
    def _build_palette_cells(self) -> None:
        """Compute the palette cell rects (panel/screen coords, flip-agnostic)."""
        Rect = self.pygame.Rect
        cell = PALETTE_CELL
        gap = 3
        x0 = BOARD_PX + 12
        self.palette_white_y = 120
        self.palette_black_y = self.palette_white_y + cell + 6
        self.palette_eraser_y = self.palette_black_y + cell + 6
        self.palette_cells: List[Tuple[object, str]] = []
        for i, s in enumerate(["K", "Q", "R", "B", "N", "P"]):
            self.palette_cells.append(
                (Rect(x0 + i * (cell + gap), self.palette_white_y, cell, cell), s)
            )
        for i, s in enumerate(["k", "q", "r", "b", "n", "p"]):
            self.palette_cells.append(
                (Rect(x0 + i * (cell + gap), self.palette_black_y, cell, cell), s)
            )
        self.palette_cells.append(
            (Rect(x0, self.palette_eraser_y, cell * 2 + gap, cell), "eraser")
        )

    def _enter_setup(self) -> None:
        """Initialise the editor FROM the current board (pieces, turn, castling)."""
        self.mode = "setup"
        self.setup_pieces = {
            sq: self.board.piece_at(sq)
            for sq in chess.SQUARES
            if self.board.piece_at(sq) is not None
        }
        self.setup_turn = self.board.turn
        # Adopt the live board's castling situation: if it has none, start
        # cleared; otherwise auto-derive from home squares.
        self.setup_castling_cleared = self.board.castling_rights == 0
        self.hint_squares = None
        self.suggest_lines = []
        self.setup_status = ""
        self._status_is_error = False
        if self.brush is None:
            self.brush = "P"

    def _reset_editor(self) -> None:
        """Reset the editor to the standard starting position."""
        start = chess.Board()
        self.setup_pieces = {
            sq: start.piece_at(sq)
            for sq in chess.SQUARES
            if start.piece_at(sq) is not None
        }
        self.setup_turn = chess.WHITE
        self.setup_castling_cleared = False
        self._clear_analysis()

    def _clear_analysis(self) -> None:
        self.hint_squares = None
        self.suggest_lines = []
        self.setup_status = ""
        self._status_is_error = False

    def _brush_label(self) -> str:
        if self.brush is None:
            return "(none - pick one)"
        if self.brush == "eraser":
            return "eraser"
        return self.brush

    def _derive_castling_fen(self) -> str:
        """Auto-grant castling rights when the king AND rook are home."""
        if self.setup_castling_cleared:
            return "-"

        def has(sq: int, sym: str) -> bool:
            p = self.setup_pieces.get(sq)
            return p is not None and p.symbol() == sym

        rights = ""
        if has(chess.E1, "K"):
            if has(chess.H1, "R"):
                rights += "K"
            if has(chess.A1, "R"):
                rights += "Q"
        if has(chess.E8, "k"):
            if has(chess.H8, "r"):
                rights += "k"
            if has(chess.A8, "r"):
                rights += "q"
        return rights or "-"

    def _build_setup_board(self) -> chess.Board:
        """Construct a chess.Board reflecting placement, turn and castling."""
        board = chess.Board.empty()
        for sq, piece in self.setup_pieces.items():
            board.set_piece_at(sq, piece)
        board.turn = self.setup_turn
        board.set_castling_fen(self._derive_castling_fen())
        return board

    @staticmethod
    def _validate(board: chess.Board) -> Optional[str]:
        """Return None if the position is legal, else a human-readable reason."""
        status = board.status()
        if status == chess.STATUS_VALID:
            return None
        msgs: List[str] = []
        if status & chess.STATUS_EMPTY:
            msgs.append("Board is empty")
        if status & chess.STATUS_NO_WHITE_KING:
            msgs.append("Missing white king")
        if status & chess.STATUS_NO_BLACK_KING:
            msgs.append("Missing black king")
        if status & chess.STATUS_TOO_MANY_KINGS:
            msgs.append("Too many kings")
        if status & chess.STATUS_PAWNS_ON_BACKRANK:
            msgs.append("Pawns on back rank")
        if status & chess.STATUS_OPPOSITE_CHECK:
            msgs.append("Side not to move is in check")
        if status & (chess.STATUS_TOO_MANY_WHITE_PIECES
                     | chess.STATUS_TOO_MANY_BLACK_PIECES):
            msgs.append("Too many pieces")
        if status & (chess.STATUS_TOO_MANY_WHITE_PAWNS
                     | chess.STATUS_TOO_MANY_BLACK_PAWNS):
            msgs.append("Too many pawns")
        if status & chess.STATUS_TOO_MANY_CHECKERS:
            msgs.append("Too many checkers")
        if status & chess.STATUS_IMPOSSIBLE_CHECK:
            msgs.append("Impossible check")
        if not msgs:
            msgs.append("Invalid position")
        return "; ".join(msgs)

    # -------------------------------------------------------------- play state
    def _human_turn(self) -> bool:
        # In advisor mode the user moves for BOTH colours, so it is always the
        # human's turn regardless of whose side is on move.
        if self.advisor or self.agent is None:
            return True
        return self.board.turn == (chess.WHITE if self.human_is_white else chess.BLACK)

    def _refresh_status(self) -> None:
        if self.board.is_game_over(claim_draw=True):
            self.status_msg = _game_over_text(self.board)
        elif self.board.is_check():
            self.status_msg = "%s to move - CHECK" % _side_name(self.board.turn)
        else:
            self.status_msg = "%s to move" % _side_name(self.board.turn)

    def _clear_selection(self) -> None:
        self.selected = None
        self.legal_dests = []

    def _new_game(self) -> None:
        self.board = chess.Board()
        self.redo_stack.clear()
        self._clear_selection()
        self.hint_squares = None
        self.eval_msg = ""
        self.hint_info = []
        self._refresh_status()

    def _undo(self) -> None:
        # Advisor: the user made the last move -> pop a single ply. Same when
        # there is no agent (human-vs-human). Only vs-agent play pops the pair.
        plies = 1 if (self.advisor or self.agent is None) else 2
        for _ in range(plies):
            if self.board.move_stack:
                self.redo_stack.append(self.board.pop())
        self._clear_selection()
        self.hint_squares = None
        self.hint_info = []
        self._refresh_status()

    def _push_move(self, move: chess.Move) -> None:
        """Play ``move`` on the board, with its sound.

        Every piece movement goes through here -- human, agent and redo -- so
        the board and the audio cannot drift apart.
        """
        if self.sounds is not None:
            self.sounds.play(self.sounds.for_move(self.board, move))
        self.board.push(move)

    def _redo(self) -> None:
        """Replay plies taken back by Undo, in the order they were played.

        Undo pops a human/agent pair when playing against the agent, so Redo
        restores the pair too -- the agent's reply is replayed rather than
        re-searched, which keeps Redo instant and exactly reverses the Undo.
        """
        if not self.redo_stack:
            self.status_msg = "Nothing to redo"
            return
        plies = 1 if (self.advisor or self.agent is None) else 2
        for _ in range(plies):
            if not self.redo_stack:
                break
            move = self.redo_stack[-1]
            if move not in self.board.legal_moves:
                # The board moved onto a different line (or was replaced by the
                # editor) since these plies were taken back; drop them.
                self.redo_stack.clear()
                self.status_msg = "Nothing to redo"
                break
            self.redo_stack.pop()
            self._push_move(move)
        self._clear_selection()
        self.hint_squares = None
        self.hint_info = []
        self._refresh_status()

    def _toggle_mute(self) -> None:
        if self.sounds is None or not self.sounds.enabled:
            self.status_msg = "Sound unavailable"
            return
        self.sounds.muted = not self.sounds.muted
        self.status_msg = "Sound off" if self.sounds.muted else "Sound on"

    def _hint(self) -> None:
        if self.agent is None:
            self.hint_squares = None
            self.hint_info = ["Load a model (--model) for suggestions."]
            return
        if self.board.is_game_over(claim_draw=True):
            self.hint_squares = None
            self.hint_info = [_game_over_text(self.board)]
            return
        try:
            info = self.agent.suggest_move(self.board)
            mv = info.get("move")
            if mv is not None:
                self.hint_squares = (mv.from_square, mv.to_square)
                self.hint_info = [
                    "Hint: %s" % info.get("san", ""),
                    "value: %+.3f" % float(info.get("value", 0.0)),
                ]
                for san, prob in info.get("top_moves", [])[:5]:
                    self.hint_info.append("  %-6s %4.1f%%" % (san, float(prob) * 100))
        except Exception as exc:  # pragma: no cover - defensive
            self.hint_info = ["Hint failed: %s" % exc]

    def _play_click(self, pos: Tuple[int, int]) -> None:
        if not self._human_turn() or self.board.is_game_over(claim_draw=True):
            return
        sq = _screen_to_square(pos, self.flipped)
        if sq is None:
            return
        piece = self.board.piece_at(sq)
        if self.selected is None:
            if piece is not None and piece.color == self.board.turn:
                self.selected = sq
                self.legal_dests = [
                    m.to_square for m in self.board.legal_moves
                    if m.from_square == sq
                ]
        else:
            move = _build_move(self.board, self.selected, sq)
            if move is not None and move in self.board.legal_moves:
                if not self.redo_stack or move != self.redo_stack[-1]:
                    # Playing something other than what Undo took back starts a
                    # new line, so the taken-back plies are no longer reachable.
                    self.redo_stack.clear()
                else:
                    self.redo_stack.pop()
                self._push_move(move)
                self._clear_selection()
                self.hint_squares = None
                self.hint_info = []
                self._refresh_status()
            elif piece is not None and piece.color == self.board.turn:
                self.selected = sq
                self.legal_dests = [
                    m.to_square for m in self.board.legal_moves
                    if m.from_square == sq
                ]
            else:
                self._clear_selection()

    def _agent_reply(self) -> None:
        try:
            move = self.agent.play_move(self.board)
            if move is not None:
                try:
                    info = self.agent.suggest_move(self.board, simulations=1)
                    self.eval_msg = "eval: %+.3f" % float(info.get("value", 0.0))
                except Exception:
                    self.eval_msg = ""
                if self.redo_stack and move == self.redo_stack[-1]:
                    self.redo_stack.pop()
                else:
                    self.redo_stack.clear()
                self._push_move(move)
        except Exception as exc:  # pragma: no cover - defensive
            self.status_msg = "Agent error: %s" % exc
        self._refresh_status()
        self.thinking = False

    # ------------------------------------------------------------- setup edit
    def _setup_click(self, pos: Tuple[int, int], button: int) -> None:
        x, _y = pos
        if x < BOARD_PX:
            sq = _screen_to_square(pos, self.flipped)
            if sq is None:
                return
            if button == 3:                     # right-click: quick erase
                if self.setup_pieces.pop(sq, None) is not None:
                    self._clear_analysis()
                return
            if button == 1:
                if self.brush is None:
                    self.setup_status = "Pick a brush from the palette"
                    self._status_is_error = True
                    return
                if self.brush == "eraser":
                    self.setup_pieces.pop(sq, None)
                else:
                    self.setup_pieces[sq] = chess.Piece.from_symbol(self.brush)
                self._clear_analysis()
        else:
            if button == 1:
                for rect, key in self.palette_cells:
                    if rect.collidepoint(pos):
                        self.brush = key
                        break

    def _play_from_position(self) -> None:
        board = self._build_setup_board()
        err = self._validate(board)
        if err is not None:
            self.setup_status = "Invalid: " + err
            self._status_is_error = True
            self.suggest_lines = []
            self.hint_squares = None
            return
        # Adopt as a fresh game (empty move stack) and switch to play mode.
        self.board = board
        self.redo_stack.clear()
        self.mode = "play"
        self._clear_selection()
        self.hint_squares = None
        self.eval_msg = ""
        self.hint_info = []
        self._refresh_status()

    def _analyze(self) -> None:
        board = self._build_setup_board()
        err = self._validate(board)
        if err is not None:
            self.setup_status = "Invalid: " + err
            self._status_is_error = True
            self.suggest_lines = []
            self.hint_squares = None
            return
        if self.agent is None:
            self.setup_status = "Load a model to get suggestions."
            self._status_is_error = True
            self.suggest_lines = []
            self.hint_squares = None
            return
        if board.is_game_over(claim_draw=True):
            self.setup_status = _game_over_text(board)
            self._status_is_error = False
            self.suggest_lines = []
            self.hint_squares = None
            return

        # Draw an "analyzing..." frame before the blocking search.
        self.setup_status = "analyzing..."
        self._status_is_error = False
        self.suggest_lines = []
        self.hint_squares = None
        if self.has_display:
            self.draw()
            self.pygame.display.flip()

        try:
            info = self.agent.suggest_move(board)
        except Exception as exc:  # pragma: no cover - defensive
            self.setup_status = "Analysis failed: %s" % exc
            self._status_is_error = True
            return

        mv = info.get("move")
        if mv is None:
            self.setup_status = "No move available"
            self._status_is_error = True
            return
        self.hint_squares = (mv.from_square, mv.to_square)
        self.setup_status = "Best: %s   eval %+.2f" % (
            info.get("san", ""), float(info.get("value", 0.0))
        )
        self._status_is_error = False
        self.suggest_lines = ["Suggestions (visits):"]
        for san, prob in info.get("top_moves", [])[:5]:
            self.suggest_lines.append("  %-7s %5.1f%%" % (san, float(prob) * 100))

    # -------------------------------------------------------------- dispatch
    def handle_keydown(self, key) -> bool:
        """Handle a KEYDOWN; return False to quit the app."""
        K = self.pygame
        if key == K.K_q:                        # Q quits from anywhere
            return False
        if self.mode == "play":
            if key == K.K_ESCAPE:               # ESC quits in play mode
                return False
            elif key == K.K_e:
                self._enter_setup()
            elif key == K.K_n:
                self._new_game()
            elif key == K.K_f:
                self.flipped = not self.flipped
            elif key == K.K_u:
                self._undo()
            elif key == K.K_r:
                self._redo()
            elif key == K.K_m:
                self._toggle_mute()
            elif key in (K.K_h, K.K_SPACE, K.K_a):
                self._hint()
        else:  # setup mode
            if key in (K.K_ESCAPE, K.K_e):      # ESC/E returns to play mode
                self.mode = "play"
                self._refresh_status()
            elif key == K.K_t:
                self.setup_turn = not self.setup_turn
                self._clear_analysis()
            elif key == K.K_c:
                self.setup_pieces = {}
                self._clear_analysis()
            elif key == K.K_r:
                self._reset_editor()
            elif key == K.K_x:
                self.brush = "eraser"
            elif key == K.K_k:
                self.setup_castling_cleared = True
                self._clear_analysis()
            elif key == K.K_f:
                self.flipped = not self.flipped
            elif key in (K.K_SPACE, K.K_a):
                self._analyze()
            elif key == K.K_p:
                self._play_from_position()
        return True

    def handle_mouse(self, pos: Tuple[int, int], button: int) -> None:
        if self.mode == "play":
            if button == 1:
                self._play_click(pos)
        else:
            self._setup_click(pos, button)

    # ------------------------------------------------------------------ draw
    def draw(self) -> None:
        if self.mode == "play":
            _draw(
                self.pygame, self.surface, self.renderer, self.board,
                self.flipped, self.selected, self.legal_dests,
                self.hint_squares, self.panel_font, self.panel_small,
                self.panel_bold, self.status_msg, self.eval_msg,
                self.hint_info, self.agent, self.agent_error, self.thinking,
                self.coord_font, self.advisor,
            )
        else:
            self._draw_setup()

    def _wrap(self, text: str, font, maxw: int) -> List[str]:
        words = text.split(" ")
        lines: List[str] = []
        cur = ""
        for w in words:
            trial = (cur + " " + w).strip()
            if not cur or font.size(trial)[0] <= maxw:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    def _draw_palette(self) -> None:
        pygame = self.pygame
        screen = self.surface
        r = self.renderer
        for rect, key in self.palette_cells:
            pygame.draw.rect(screen, PALETTE_BG, rect)
            pygame.draw.rect(screen, PALETTE_EDGE, rect, 1)
            if key == "eraser":
                txt = self.panel_small.render("erase", True, PANEL_FG)
                screen.blit(txt, txt.get_rect(center=rect.center))
            else:
                piece = chess.Piece.from_symbol(key)
                r.draw_piece_at(screen, piece, rect.centerx, rect.centery,
                                int(rect.height * 0.95))
            if self.brush == key:
                pygame.draw.rect(screen, PANEL_ACCENT, rect, 3)

    def _draw_setup(self) -> None:
        pygame = self.pygame
        screen = self.surface
        r = self.renderer

        # --- board backdrop + squares ---
        _draw_board_backdrop(pygame, screen, self.flipped)

        # --- analysis best-move highlight (reuses the hint mechanism) ---
        if self.hint_squares is not None:
            for sq in self.hint_squares:
                _fill_square(pygame, screen, sq, self.flipped, (*HINT_COLOR, 90))
                x, y = _square_to_screen(sq, self.flipped)
                pygame.draw.rect(screen, HINT_COLOR, (x, y, SQUARE, SQUARE), 4)

        # --- pieces (from the editor piece map) ---
        for sq, piece in self.setup_pieces.items():
            x, y = _square_to_screen(sq, self.flipped)
            r.draw_piece(screen, piece, x, y)

        # --- board coordinates (a-h / 1-8) in the margin ---
        _draw_coordinates(pygame, screen, self.flipped, self.coord_font)

        # --- side panel ---
        pygame.draw.rect(screen, PANEL_BG, (BOARD_PX, 0, PANEL_PX, WINDOW_H))
        px = BOARD_PX + 12
        y = 12

        def line(text: str, font, color=PANEL_FG, dy: int = 24) -> None:
            nonlocal y
            screen.blit(font.render(text, True, color), (px, y))
            y += dy

        line("SETUP MODE", self.panel_bold, PANEL_ACCENT, dy=30)
        side = "White to move" if self.setup_turn == chess.WHITE else "Black to move"
        line(side, self.panel_font, PANEL_FG, dy=26)
        line("Castling: %s" % self._derive_castling_fen(),
             self.panel_small, PANEL_DIM, dy=22)
        line("Brush: %s" % self._brush_label(), self.panel_small, PANEL_DIM, dy=20)

        # palette (fixed y coordinates)
        self._draw_palette()

        # status + suggestions below the palette
        y = self.palette_eraser_y + PALETTE_CELL + 12
        if self.agent is None and self.agent_error:
            for ln in self._wrap(self.agent_error, self.panel_tiny, PANEL_PX - 24):
                line(ln, self.panel_tiny, PANEL_ERR, dy=17)
            y += 4
        if self.setup_status:
            col = PANEL_ERR if self._status_is_error else PANEL_INFO
            for ln in self._wrap(self.setup_status, self.panel_small, PANEL_PX - 24):
                line(ln, self.panel_small, col, dy=20)
        for ln in self.suggest_lines:
            line(ln, self.panel_small, PANEL_INFO, dy=20)

        # --- key help pinned near the bottom ---
        help_lines = [
            "SPACE/A - analyze",
            "P - use position" if self.advisor else "P - play from position",
            "T - side  C - clear board",
            "R - reset  X - eraser",
            "K - clear castling  F - flip",
            "E/ESC - exit setup   Q - quit",
        ]
        hy = WINDOW_H - 17 * len(help_lines) - 10
        for text in help_lines:
            screen.blit(self.panel_tiny.render(text, True, PANEL_DIM), (px, hy))
            hy += 17

    # ------------------------------------------------------------------- loop
    def run(self, clock) -> None:
        pygame = self.pygame
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self.handle_keydown(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse(event.pos, event.button)
                if not running:
                    break
            if not running:
                break

            # Decide AFTER the event loop so Undo / New-game / mode switches
            # handled this frame are respected before the agent replies.
            agent_should_move = (
                self.mode == "play"
                and not self.advisor
                and self.agent is not None
                and not self.board.is_game_over(claim_draw=True)
                and not self._human_turn()
            )
            self.thinking = agent_should_move

            self.draw()
            pygame.display.flip()

            if agent_should_move:
                self._agent_reply()

            clock.tick(FPS)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def launch_gui(
    model_path: Optional[str] = None,
    simulations: int = 200,
    human_color: str = "white",
    device=None,
    start_in_setup: bool = False,
    initial_fen: Optional[str] = None,
    advisor: bool = False,
) -> None:
    """Launch an interactive pygame window to play against the agent.

    Args:
        model_path: Path to a saved model checkpoint. If None or missing, the
            GUI runs human-vs-human (play) with hints/agent disabled; in setup
            mode analysis is disabled until a model is loaded.
        simulations: MCTS simulations per agent move / hint / analysis.
        human_color: "white" or "black" — the side the human plays.
        device: Optional torch device passed to the agent.
        start_in_setup: Start directly in the board-editor / analysis mode.
        initial_fen: Optional FEN used to initialise the board (and, when
            ``start_in_setup``, the editor).
        advisor: Open the ADVISOR board -- the user makes every move for BOTH
            colours (mirroring an external game) and asks the engine for the
            best move on demand; the engine never moves on its own.
    """
    import os
    import pygame

    pygame.init()
    pygame.display.set_caption("AlphaChess")
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock = pygame.time.Clock()

    agent = None
    agent_error: Optional[str] = None
    if model_path and os.path.exists(model_path):
        try:
            from alpha_chess.agent import AlphaChessAgent
            agent = AlphaChessAgent(
                model_path, device=device, simulations=simulations
            )
        except Exception as exc:  # pragma: no cover - defensive
            agent = None
            agent_error = "Failed to load model: %s" % exc
    elif model_path:
        agent_error = "Model not found: %s" % model_path

    human_is_white = str(human_color).lower() != "black"

    app = _GuiApp(
        pygame, screen,
        agent=agent, agent_error=agent_error,
        human_is_white=human_is_white, simulations=simulations,
        start_in_setup=start_in_setup, initial_fen=initial_fen,
        has_display=True, advisor=advisor,
    )
    app.run(clock)
    pygame.quit()


# ---------------------------------------------------------------------------
# Drawing / text helpers
# ---------------------------------------------------------------------------
def _side_name(turn: bool) -> str:
    return "White" if turn == chess.WHITE else "Black"


def _game_over_text(board: chess.Board) -> str:
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        return "Checkmate - %s wins" % winner
    if board.is_stalemate():
        return "Stalemate - draw"
    if board.is_insufficient_material():
        return "Draw - insufficient material"
    if board.can_claim_fifty_moves():
        return "Draw - fifty-move rule"
    if board.can_claim_threefold_repetition():
        return "Draw - threefold repetition"
    return "Game over - %s" % board.result(claim_draw=True)


def _wrap_text(text: str, font, maxw: int) -> List[str]:
    """Word-wrap ``text`` to fit within ``maxw`` pixels for ``font``."""
    words = text.split(" ")
    lines: List[str] = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        if not cur or font.size(trial)[0] <= maxw:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _build_move(board: chess.Board, frm: int, to: int) -> Optional[chess.Move]:
    """Construct a move, auto-promoting pawns to a queen on the last rank."""
    piece = board.piece_at(frm)
    promotion = None
    if piece is not None and piece.piece_type == chess.PAWN:
        to_rank = chess.square_rank(to)
        if to_rank in (0, 7):
            promotion = chess.QUEEN
    return chess.Move(frm, to, promotion=promotion)


def _draw(
    pygame, screen, renderer, board, flipped, selected, legal_dests,
    hint_squares, panel_font, panel_small, panel_bold, status_msg,
    eval_msg, hint_info, agent, agent_error, thinking, coord_font=None,
    advisor=False,
) -> None:
    """Render the board, highlights, pieces and side panel (PLAY mode).

    When ``advisor`` is True the panel/chrome switches to the advisor board:
    the user moves both colours and asks the engine for suggestions on demand;
    the engine never moves, so no "thinking" indicator is shown.
    """
    # --- board backdrop + squares ---
    _draw_board_backdrop(pygame, screen, flipped)

    # --- the piece that just moved, boxed in green ---
    # A translucent tint washes out against both square colours, so this is a
    # hard-edged border on the destination only -- the square the piece now
    # occupies. Read straight off the move stack rather than tracked
    # separately, so undo, redo and new-game need no bookkeeping to keep it
    # honest. Whenever it is your turn, the piece boxed is your opponent's.
    last = board.move_stack[-1] if board.move_stack else None
    if last is not None:
        x, y = _square_to_screen(last.to_square, flipped)
        pygame.draw.rect(screen, LASTMOVE_COLOR,
                         (x, y, SQUARE, SQUARE), LASTMOVE_WIDTH)

    # --- selected square highlight (translucent) ---
    if selected is not None:
        _fill_square(pygame, screen, selected, flipped, (*SEL_COLOR, 130))

    # --- hint from/to highlight (translucent fill + border) ---
    if hint_squares is not None:
        for sq in hint_squares:
            _fill_square(pygame, screen, sq, flipped, (*HINT_COLOR, 90))
            x, y = _square_to_screen(sq, flipped)
            pygame.draw.rect(screen, HINT_COLOR, (x, y, SQUARE, SQUARE), 4)

    # --- legal destination markers ---
    for sq in legal_dests:
        x, y = _square_to_screen(sq, flipped)
        cx, cy = x + SQUARE // 2, y + SQUARE // 2
        target = board.piece_at(sq) is not None
        if target:
            # Capture: a ring around the square.
            pygame.draw.circle(screen, DEST_COLOR, (cx, cy), SQUARE // 2 - 3, 4)
        else:
            _dot = pygame.Surface((SQUARE, SQUARE), pygame.SRCALPHA)
            pygame.draw.circle(_dot, (*DEST_COLOR, 150),
                               (SQUARE // 2, SQUARE // 2), SQUARE // 7)
            screen.blit(_dot, (x, y))

    # --- pieces ---
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            x, y = _square_to_screen(square, flipped)
            renderer.draw_piece(screen, piece, x, y)

    # --- board coordinates (a-h / 1-8) in the margin ---
    _draw_coordinates(pygame, screen, flipped, coord_font or panel_small)

    # --- side panel ---
    pygame.draw.rect(screen, PANEL_BG, (BOARD_PX, 0, PANEL_PX, WINDOW_H))
    px = BOARD_PX + 14
    y = 16

    def line(text: str, font, color=PANEL_FG, dy: int = 26) -> None:
        nonlocal y
        surf = font.render(text, True, color)
        screen.blit(surf, (px, y))
        y += dy

    if advisor:
        line("ADVISOR", panel_bold, PANEL_ACCENT, dy=30)
        line("You move both sides", panel_small, PANEL_DIM, dy=24)
        line(status_msg, panel_font, PANEL_FG, dy=30)
        y += 4
        # Surface any error (e.g. a bad --fen) whether or not a model is loaded.
        if agent_error:
            for _t in _wrap_text(agent_error, panel_small, PANEL_PX - 24):
                line(_t, panel_small, (240, 160, 160), dy=22)
        if agent is None:
            line("Load a model (--model)", panel_small, PANEL_DIM, dy=22)
            line("for suggestions.", panel_small, PANEL_DIM, dy=26)
    else:
        line("AlphaChess", panel_bold, PANEL_FG, dy=34)
        line(status_msg, panel_font, PANEL_FG, dy=30)

        if thinking:
            line("thinking...", panel_font, (250, 220, 120), dy=30)
        elif eval_msg:
            line(eval_msg, panel_small, PANEL_DIM, dy=26)

        y += 6
        if agent is None:
            if agent_error:
                line(agent_error, panel_small, (240, 160, 160), dy=22)
            line("Human vs Human", panel_small, PANEL_DIM, dy=22)
            line("(agent / hints disabled)", panel_small, PANEL_DIM, dy=26)

    # --- hint info block (best-move suggestion) ---
    if hint_info:
        y += 6
        for text in hint_info:
            line(text, panel_small, (170, 200, 240), dy=22)

    # --- key help pinned near the bottom ---
    if advisor:
        help_lines = [
            "H / SPACE - best move",
            "click - move either side",
            "U / R - undo / redo",
            "M - mute sound",
            "E - set up position",
            "F - flip",
            "N - new game",
            "Q - quit",
        ]
    else:
        help_lines = [
            "H - hint",
            "U / R - undo / redo",
            "M - mute sound",
            "N - new game",
            "F - flip board",
            "E - setup / editor",
            "ESC/Q - quit",
        ]
    hy = WINDOW_H - 22 * len(help_lines) - 12
    for text in help_lines:
        surf = panel_small.render(text, True, PANEL_DIM)
        screen.blit(surf, (px, hy))
        hy += 22
