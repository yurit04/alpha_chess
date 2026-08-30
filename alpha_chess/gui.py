"""Pygame GUI for playing against the AlphaChess agent.

Importing this module does NOT open a window; only :func:`launch_gui`
initializes the pygame display.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import chess


# ---------------------------------------------------------------------------
# Layout / drawing constants
# ---------------------------------------------------------------------------
BOARD_PX = 640                     # width/height of the board in pixels
SQUARE = BOARD_PX // 8             # size of a single square
PANEL_PX = 240                     # width of the side panel
WINDOW_W = BOARD_PX + PANEL_PX
WINDOW_H = BOARD_PX
FPS = 30

# Colours (R, G, B)
LIGHT_SQ = (240, 217, 181)
DARK_SQ = (181, 136, 99)
SEL_COLOR = (246, 246, 105)        # selected square highlight
DEST_COLOR = (106, 168, 79)        # legal-destination marker
HINT_COLOR = (80, 140, 220)        # hint from/to highlight
PANEL_BG = (40, 40, 45)
PANEL_FG = (230, 230, 230)
PANEL_DIM = (170, 170, 175)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _square_to_screen(square: int, flipped: bool) -> Tuple[int, int]:
    """Return the top-left (x, y) pixel of the given board square."""
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    if flipped:
        col = 7 - file
        row = rank                 # rank1 at top when flipped
    else:
        col = file
        row = 7 - rank             # rank8 at top (standard orientation)
    return col * SQUARE, row * SQUARE


def _screen_to_square(pos: Tuple[int, int], flipped: bool) -> Optional[int]:
    """Map a screen pixel to a board square, or None if outside the board."""
    x, y = pos
    if x < 0 or x >= BOARD_PX or y < 0 or y >= BOARD_PX:
        return None
    col = x // SQUARE
    row = y // SQUARE
    if flipped:
        file = 7 - col
        rank = row
    else:
        file = col
        rank = 7 - row
    return chess.square(int(file), int(rank))


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
    """Bundles fonts and knows how to draw one piece (glyph or fallback)."""

    def __init__(self, pygame) -> None:
        self.pygame = pygame
        # A font per usable size.
        self.glyph_font = self._find_glyph_font(int(SQUARE * 0.8))
        self.fallback_font = pygame.font.SysFont(
            "Arial", int(SQUARE * 0.45), bold=True
        )

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

    def _find_glyph_font(self, size: int):
        pygame = self.pygame
        for name in GLYPH_FONTS:
            font = self._load_font(name, size)
            if font is None:
                continue
            # Require ALL six solid glyphs to render, not just the king.
            if all(_glyph_renders(font, g) for g in SOLID_GLYPHS.values()):
                self.has_glyphs = True
                return font
        # No glyph font found anywhere: fall back to drawn pieces.
        self.has_glyphs = False
        return pygame.font.SysFont(None, size)

    def draw_piece(self, surface, piece: chess.Piece, x: int, y: int) -> None:
        """Draw a piece centred inside the square at (x, y)."""
        pygame = self.pygame
        symbol = piece.symbol()
        cx, cy = x + SQUARE // 2, y + SQUARE // 2
        if self.has_glyphs:
            # Always render the solid glyph; distinguish colour by fill, and add
            # a contrasting outline so pieces are visible on either square shade.
            glyph = SOLID_GLYPHS[symbol.upper()]
            is_white = piece.color == chess.WHITE
            fill = WHITE_PIECE if is_white else BLACK_PIECE
            outline = BLACK_PIECE if is_white else WHITE_PIECE
            base = self.glyph_font.render(glyph, True, fill)
            rect = base.get_rect(center=(cx, cy))
            # Stroke: blit the outline-coloured glyph at 8 small offsets first.
            stroke = self.glyph_font.render(glyph, True, outline)
            for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2),
                           (-2, -2), (2, -2), (-2, 2), (2, 2)):
                surface.blit(stroke, stroke.get_rect(center=(cx + dx, cy + dy)))
            surface.blit(base, rect)
        else:
            # Fallback: coloured circle with the piece letter.
            radius = int(SQUARE * 0.4)
            fill = CIRCLE_WHITE if piece.color == chess.WHITE else CIRCLE_BLACK
            edge = (0, 0, 0)
            pygame.draw.circle(surface, fill, (cx, cy), radius)
            pygame.draw.circle(surface, edge, (cx, cy), radius, 2)
            txt_color = BLACK_PIECE if piece.color == chess.WHITE else WHITE_PIECE
            letter = symbol.upper() if piece.color == chess.WHITE else symbol.lower()
            surf = self.fallback_font.render(letter, True, txt_color)
            rect = surf.get_rect(center=(cx, cy))
            surface.blit(surf, rect)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def launch_gui(
    model_path: Optional[str] = None,
    simulations: int = 200,
    human_color: str = "white",
    device=None,
) -> None:
    """Launch an interactive pygame window to play against the agent.

    Args:
        model_path: Path to a saved model checkpoint. If None or missing, the
            GUI runs human-vs-human with hints/agent disabled.
        simulations: MCTS simulations per agent move / hint.
        human_color: "white" or "black" — the side the human plays.
        device: Optional torch device string passed to the agent.
    """
    import os
    import pygame

    pygame.init()
    pygame.display.set_caption("AlphaChess")
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock = pygame.time.Clock()

    renderer = _Renderer(pygame)
    panel_font = pygame.font.SysFont("Arial", 20)
    panel_small = pygame.font.SysFont("Arial", 16)
    panel_bold = pygame.font.SysFont("Arial", 22, bold=True)

    # ------------------------------------------------------------------ agent
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

    # ------------------------------------------------------------- game state
    board = chess.Board()
    flipped = not human_is_white          # keep human at the bottom by default
    selected: Optional[int] = None
    legal_dests: List[int] = []
    hint_squares: Optional[Tuple[int, int]] = None
    status_msg = "Your move" if agent else "Human vs Human"
    eval_msg = ""
    hint_info: List[str] = []
    thinking = False

    def human_turn() -> bool:
        """True if it is currently the human's turn (or no agent loaded)."""
        if agent is None:
            return True
        return board.turn == (chess.WHITE if human_is_white else chess.BLACK)

    def refresh_status() -> None:
        nonlocal status_msg
        if board.is_game_over(claim_draw=True):
            status_msg = _game_over_text(board)
        elif board.is_check():
            status_msg = "%s to move - CHECK" % _side_name(board.turn)
        else:
            status_msg = "%s to move" % _side_name(board.turn)

    def clear_selection() -> None:
        nonlocal selected, legal_dests
        selected = None
        legal_dests = []

    def new_game() -> None:
        nonlocal board, hint_squares, eval_msg, hint_info
        board = chess.Board()
        clear_selection()
        hint_squares = None
        eval_msg = ""
        hint_info = []
        refresh_status()

    refresh_status()

    # -------------------------------------------------------------- main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_n:
                    new_game()
                elif event.key == pygame.K_f:
                    flipped = not flipped
                elif event.key == pygame.K_u:
                    # Undo a full move pair (human + agent), or one ply in H-v-H.
                    plies = 2 if agent is not None else 1
                    for _ in range(plies):
                        if board.move_stack:
                            board.pop()
                    clear_selection()
                    hint_squares = None
                    hint_info = []
                    refresh_status()
                elif event.key == pygame.K_h:
                    if agent is not None and not board.is_game_over():
                        try:
                            info = agent.suggest_move(board)
                            mv = info.get("move")
                            if mv is not None:
                                hint_squares = (mv.from_square, mv.to_square)
                                hint_info = [
                                    "Hint: %s" % info.get("san", ""),
                                    "value: %+.3f" % float(info.get("value", 0.0)),
                                ]
                                top = info.get("top_moves", [])
                                for san, prob in top[:5]:
                                    hint_info.append("  %-6s %.2f" % (san, prob))
                        except Exception as exc:  # pragma: no cover
                            hint_info = ["Hint failed: %s" % exc]

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if not human_turn() or board.is_game_over(claim_draw=True):
                    continue
                sq = _screen_to_square(event.pos, flipped)
                if sq is None:
                    continue
                piece = board.piece_at(sq)
                if selected is None:
                    if piece is not None and piece.color == board.turn:
                        selected = sq
                        legal_dests = [
                            m.to_square
                            for m in board.legal_moves
                            if m.from_square == sq
                        ]
                else:
                    move = _build_move(board, selected, sq)
                    if move is not None and move in board.legal_moves:
                        board.push(move)
                        clear_selection()
                        hint_squares = None
                        hint_info = []
                        refresh_status()
                    elif piece is not None and piece.color == board.turn:
                        # Reselect a different own piece.
                        selected = sq
                        legal_dests = [
                            m.to_square
                            for m in board.legal_moves
                            if m.from_square == sq
                        ]
                    else:
                        clear_selection()

        # Decide AFTER the event loop so Undo / New-game handled this frame are
        # respected before the agent replies (avoids acting on a stale board).
        agent_should_move = (
            agent is not None
            and not board.is_game_over(claim_draw=True)
            and not human_turn()
        )

        # ---- draw everything (also shows "thinking..." before agent moves) --
        thinking = agent_should_move
        _draw(
            pygame, screen, renderer, board, flipped, selected, legal_dests,
            hint_squares, panel_font, panel_small, panel_bold, status_msg,
            eval_msg, hint_info, agent, agent_error, thinking,
        )
        pygame.display.flip()

        # ---- let the agent reply (blocking) after the frame is shown --------
        if agent_should_move and running:
            try:
                move = agent.play_move(board)
                if move is not None:
                    # Best-effort eval for the panel from the pre-move position.
                    try:
                        info = agent.suggest_move(board, simulations=1)
                        eval_msg = "eval: %+.3f" % float(info.get("value", 0.0))
                    except Exception:
                        eval_msg = ""
                    board.push(move)
            except Exception as exc:  # pragma: no cover
                status_msg = "Agent error: %s" % exc
            refresh_status()
            thinking = False

        clock.tick(FPS)

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
    eval_msg, hint_info, agent, agent_error, thinking,
) -> None:
    """Render the board, highlights, pieces and side panel."""
    # --- board squares ---
    for square in chess.SQUARES:
        x, y = _square_to_screen(square, flipped)
        is_light = (chess.square_file(square) + chess.square_rank(square)) % 2 == 1
        color = LIGHT_SQ if is_light else DARK_SQ
        pygame.draw.rect(screen, color, (x, y, SQUARE, SQUARE))

    # --- selected square highlight ---
    if selected is not None:
        x, y = _square_to_screen(selected, flipped)
        pygame.draw.rect(screen, SEL_COLOR, (x, y, SQUARE, SQUARE))

    # --- hint from/to highlight ---
    if hint_squares is not None:
        for sq in hint_squares:
            x, y = _square_to_screen(sq, flipped)
            pygame.draw.rect(screen, HINT_COLOR, (x, y, SQUARE, SQUARE), 5)

    # --- legal destination markers ---
    for sq in legal_dests:
        x, y = _square_to_screen(sq, flipped)
        cx, cy = x + SQUARE // 2, y + SQUARE // 2
        pygame.draw.circle(screen, DEST_COLOR, (cx, cy), SQUARE // 6)

    # --- pieces ---
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            x, y = _square_to_screen(square, flipped)
            renderer.draw_piece(screen, piece, x, y)

    # --- side panel ---
    pygame.draw.rect(screen, PANEL_BG, (BOARD_PX, 0, PANEL_PX, WINDOW_H))
    px = BOARD_PX + 14
    y = 16

    def line(text: str, font, color=PANEL_FG, dy: int = 26) -> None:
        nonlocal y
        surf = font.render(text, True, color)
        screen.blit(surf, (px, y))
        y += dy

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

    # --- hint info block ---
    if hint_info:
        y += 6
        for text in hint_info:
            line(text, panel_small, (170, 200, 240), dy=22)

    # --- key help pinned near the bottom ---
    help_lines = [
        "H - hint",
        "U - undo",
        "N - new game",
        "F - flip board",
        "ESC/Q - quit",
    ]
    hy = WINDOW_H - 22 * len(help_lines) - 12
    for text in help_lines:
        surf = panel_small.render(text, True, PANEL_DIM)
        screen.blit(surf, (px, hy))
        hy += 22
