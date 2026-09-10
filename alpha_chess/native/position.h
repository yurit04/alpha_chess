/* Position representation, legal move generation and move making.
 *
 * Standard chess only (no Chess960).  Move generation is fully legal: pins and
 * check evasions are resolved up front rather than by generating pseudo-legal
 * moves and filtering them, which is what makes it ~40x faster than the
 * python-chess path it replaces.  Correctness is pinned down by perft (see
 * tests/test_native.py), which is exhaustive enough to catch any slip here.
 */
#ifndef ALPHACHESS_POSITION_H
#define ALPHACHESS_POSITION_H

#include <stdlib.h>
#include "bitboard.h"

enum { WHITE = 0, BLACK = 1 };
enum { PAWN = 0, KNIGHT, BISHOP, ROOK, QUEEN, KING };

/* Castling-rights bits. */
#define CR_WK 1
#define CR_WQ 2
#define CR_BK 4
#define CR_BQ 8

/* Packed move: from | to<<6 | promo<<12 | flags<<15.
 * ``promo`` is the promotion piece type (KNIGHT..QUEEN) or 0 for none. */
typedef uint32_t Move;

#define MV_EP 1u
#define MV_CASTLE 2u

#define MV_FROM(m) ((int)((m) & 63u))
#define MV_TO(m) ((int)(((m) >> 6) & 63u))
#define MV_PROMO(m) ((int)(((m) >> 12) & 7u))
#define MV_FLAGS(m) ((m) >> 15)
#define MK_MOVE(f, t, p, fl) \
    ((Move)(f) | ((Move)(t) << 6) | ((Move)(p) << 12) | ((Move)(fl) << 15))

#define MAX_MOVES 256

typedef struct {
    U64 bb[2][6];
    U64 occ[2];
    U64 all;
    U64 key;
    int8_t ep;        /* en-passant target square, or -1 */
    /* Square whose file was folded into ``key``.  An en-passant square only
     * distinguishes two positions when the capture is actually available, so
     * only then does it take part in the repetition key -- while ``ep`` itself
     * always tracks the double push, because that is what encode_board marks. */
    int8_t ep_key;
    uint8_t side;
    uint8_t castling;
    uint16_t halfmove;
} Pos;

static U64 Z_PIECE[2][6][64];
static U64 Z_SIDE;
static U64 Z_CASTLE[16];
static U64 Z_EP[8];
static int POS_READY = 0;

static void pos_init(void)
{
    if (POS_READY) return;
    bb_init();
    bb_rng_state = 0xD1B54A32D192ED03ULL;
    for (int c = 0; c < 2; c++)
        for (int p = 0; p < 6; p++)
            for (int s = 0; s < 64; s++) Z_PIECE[c][p][s] = bb_rand64();
    Z_SIDE = bb_rand64();
    for (int i = 0; i < 16; i++) Z_CASTLE[i] = bb_rand64();
    for (int i = 0; i < 8; i++) Z_EP[i] = bb_rand64();
    POS_READY = 1;
}

static inline int pos_piece_at(const Pos *p, int color, int sq)
{
    U64 b = SQ(sq);
    if (!(p->occ[color] & b)) return -1;
    for (int t = 0; t < 6; t++)
        if (p->bb[color][t] & b) return t;
    return -1;
}

static void pos_set_start(Pos *p)
{
    memset(p, 0, sizeof(*p));
    p->bb[WHITE][PAWN] = 0x000000000000FF00ULL;
    p->bb[WHITE][KNIGHT] = 0x0000000000000042ULL;
    p->bb[WHITE][BISHOP] = 0x0000000000000024ULL;
    p->bb[WHITE][ROOK] = 0x0000000000000081ULL;
    p->bb[WHITE][QUEEN] = 0x0000000000000008ULL;
    p->bb[WHITE][KING] = 0x0000000000000010ULL;
    p->bb[BLACK][PAWN] = 0x00FF000000000000ULL;
    p->bb[BLACK][KNIGHT] = 0x4200000000000000ULL;
    p->bb[BLACK][BISHOP] = 0x2400000000000000ULL;
    p->bb[BLACK][ROOK] = 0x8100000000000000ULL;
    p->bb[BLACK][QUEEN] = 0x0800000000000000ULL;
    p->bb[BLACK][KING] = 0x1000000000000000ULL;
    p->side = WHITE;
    p->ep = -1;
    p->ep_key = -1;
    p->castling = CR_WK | CR_WQ | CR_BK | CR_BQ;
    p->halfmove = 0;
    for (int c = 0; c < 2; c++) {
        p->occ[c] = 0;
        for (int t = 0; t < 6; t++) p->occ[c] |= p->bb[c][t];
    }
    p->all = p->occ[WHITE] | p->occ[BLACK];
    p->key = 0;
    for (int c = 0; c < 2; c++)
        for (int t = 0; t < 6; t++) {
            U64 b = p->bb[c][t];
            while (b) p->key ^= Z_PIECE[c][t][bb_pop_lsb(&b)];
        }
    p->key ^= Z_CASTLE[p->castling];
}

/* Every piece of either colour attacking ``sq`` under occupancy ``occ``. */
static inline U64 pos_attackers_to(const Pos *p, int sq, U64 occ)
{
    U64 bq = p->bb[WHITE][BISHOP] | p->bb[BLACK][BISHOP]
           | p->bb[WHITE][QUEEN] | p->bb[BLACK][QUEEN];
    U64 rq = p->bb[WHITE][ROOK] | p->bb[BLACK][ROOK]
           | p->bb[WHITE][QUEEN] | p->bb[BLACK][QUEEN];
    return (PAWN_ATT[WHITE][sq] & p->bb[BLACK][PAWN])
         | (PAWN_ATT[BLACK][sq] & p->bb[WHITE][PAWN])
         | (KNIGHT_ATT[sq] & (p->bb[WHITE][KNIGHT] | p->bb[BLACK][KNIGHT]))
         | (KING_ATT[sq] & (p->bb[WHITE][KING] | p->bb[BLACK][KING]))
         | (bb_bishop_attacks(sq, occ) & bq)
         | (bb_rook_attacks(sq, occ) & rq);
}

static inline int pos_in_check(const Pos *p)
{
    int ksq = bb_lsb(p->bb[p->side][KING]);
    return (pos_attackers_to(p, ksq, p->all) & p->occ[!p->side]) != 0;
}

/* python-chess's ``has_insufficient_material`` for one colour, matched exactly
 * so the native and Python engines score the same games as draws. */
static int pos_insufficient_one(const Pos *p, int c)
{
    int them = !c;
    if (p->occ[c] & (p->bb[c][PAWN] | p->bb[c][ROOK] | p->bb[c][QUEEN])) return 0;
    if (p->bb[c][KNIGHT]) {
        U64 kings = p->bb[WHITE][KING] | p->bb[BLACK][KING];
        U64 queens = p->bb[WHITE][QUEEN] | p->bb[BLACK][QUEEN];
        return bb_popcount(p->occ[c]) <= 2
            && !(p->occ[them] & ~kings & ~queens);
    }
    if (p->bb[c][BISHOP]) {
        U64 bishops = p->bb[WHITE][BISHOP] | p->bb[BLACK][BISHOP];
        int same_color = !(bishops & DARK_SQUARES) || !(bishops & ~DARK_SQUARES);
        U64 pawns = p->bb[WHITE][PAWN] | p->bb[BLACK][PAWN];
        U64 knights = p->bb[WHITE][KNIGHT] | p->bb[BLACK][KNIGHT];
        return same_color && !pawns && !knights;
    }
    return 1;
}

static inline int pos_insufficient_material(const Pos *p)
{
    return pos_insufficient_one(p, WHITE) && pos_insufficient_one(p, BLACK);
}

/* ------------------------------------------------------------------ */
/* Move generation                                                     */
/* ------------------------------------------------------------------ */

static inline void gen_add_pawn_moves(Move *out, int *n, int from, U64 targets,
                                      int promo_rank)
{
    while (targets) {
        int to = bb_pop_lsb(&targets);
        if ((to >> 3) == promo_rank) {
            out[(*n)++] = MK_MOVE(from, to, QUEEN, 0);
            out[(*n)++] = MK_MOVE(from, to, ROOK, 0);
            out[(*n)++] = MK_MOVE(from, to, BISHOP, 0);
            out[(*n)++] = MK_MOVE(from, to, KNIGHT, 0);
        } else {
            out[(*n)++] = MK_MOVE(from, to, 0, 0);
        }
    }
}

static int pos_gen_legal(const Pos *p, Move *out)
{
    int n = 0;
    const int us = p->side, them = !p->side;
    const U64 all = p->all;
    const U64 mine = p->occ[us], theirs = p->occ[them];
    const int ksq = bb_lsb(p->bb[us][KING]);

    U64 checkers = pos_attackers_to(p, ksq, all) & theirs;
    int n_checkers = bb_popcount(checkers);

    /* King moves: legal iff the destination is unattacked once the king has
     * left its square (so it cannot walk backwards along a checking ray). */
    U64 king_targets = KING_ATT[ksq] & ~mine;
    U64 occ_no_king = all ^ SQ(ksq);
    while (king_targets) {
        int to = bb_pop_lsb(&king_targets);
        if (!(pos_attackers_to(p, to, occ_no_king) & (theirs & ~SQ(to))))
            out[n++] = MK_MOVE(ksq, to, 0, 0);
    }

    if (n_checkers >= 2) return n; /* double check: only the king may move */

    /* Squares a non-king move may land on to leave the king safe. */
    U64 target_mask = ~mine;
    if (n_checkers == 1) {
        int csq = bb_lsb(checkers);
        target_mask &= BETWEEN_BB[ksq][csq] | checkers;
    }

    /* Absolute pins: an enemy slider whose ray to our king holds exactly one
     * piece, and that piece is ours. */
    U64 pinned = 0;
    U64 snipers = (bb_rook_attacks(ksq, 0)
                   & (p->bb[them][ROOK] | p->bb[them][QUEEN]))
                | (bb_bishop_attacks(ksq, 0)
                   & (p->bb[them][BISHOP] | p->bb[them][QUEEN]));
    U64 s = snipers;
    while (s) {
        int sq = bb_pop_lsb(&s);
        U64 between = BETWEEN_BB[ksq][sq] & all;
        if (between && !(between & (between - 1)) && (between & mine))
            pinned |= between;
    }

    if (n_checkers == 0) {
        /* Castling: rights, empty path, and no attacked square along the
         * king's route (including its start square). */
        if (us == WHITE) {
            if ((p->castling & CR_WK) && !(all & 0x60ULL)
                && !(pos_attackers_to(p, 4, all) & theirs)
                && !(pos_attackers_to(p, 5, all) & theirs)
                && !(pos_attackers_to(p, 6, all) & theirs))
                out[n++] = MK_MOVE(4, 6, 0, MV_CASTLE);
            if ((p->castling & CR_WQ) && !(all & 0x0EULL)
                && !(pos_attackers_to(p, 4, all) & theirs)
                && !(pos_attackers_to(p, 3, all) & theirs)
                && !(pos_attackers_to(p, 2, all) & theirs))
                out[n++] = MK_MOVE(4, 2, 0, MV_CASTLE);
        } else {
            if ((p->castling & CR_BK) && !(all & 0x6000000000000000ULL)
                && !(pos_attackers_to(p, 60, all) & theirs)
                && !(pos_attackers_to(p, 61, all) & theirs)
                && !(pos_attackers_to(p, 62, all) & theirs))
                out[n++] = MK_MOVE(60, 62, 0, MV_CASTLE);
            if ((p->castling & CR_BQ) && !(all & 0x0E00000000000000ULL)
                && !(pos_attackers_to(p, 60, all) & theirs)
                && !(pos_attackers_to(p, 59, all) & theirs)
                && !(pos_attackers_to(p, 58, all) & theirs))
                out[n++] = MK_MOVE(60, 58, 0, MV_CASTLE);
        }
    }

    /* Knights (a pinned knight can never move). */
    U64 b = p->bb[us][KNIGHT] & ~pinned;
    while (b) {
        int from = bb_pop_lsb(&b);
        U64 t = KNIGHT_ATT[from] & target_mask;
        while (t) out[n++] = MK_MOVE(from, bb_pop_lsb(&t), 0, 0);
    }

    /* Bishops and queens on diagonals. */
    b = p->bb[us][BISHOP] | p->bb[us][QUEEN];
    while (b) {
        int from = bb_pop_lsb(&b);
        U64 t = bb_bishop_attacks(from, all) & target_mask;
        if (SQ(from) & pinned) t &= LINE_BB[ksq][from];
        while (t) out[n++] = MK_MOVE(from, bb_pop_lsb(&t), 0, 0);
    }

    /* Rooks and queens on files/ranks. */
    b = p->bb[us][ROOK] | p->bb[us][QUEEN];
    while (b) {
        int from = bb_pop_lsb(&b);
        U64 t = bb_rook_attacks(from, all) & target_mask;
        if (SQ(from) & pinned) t &= LINE_BB[ksq][from];
        while (t) out[n++] = MK_MOVE(from, bb_pop_lsb(&t), 0, 0);
    }

    /* Pawns. */
    const int up = (us == WHITE) ? 8 : -8;
    const int promo_rank = (us == WHITE) ? 7 : 0;
    const U64 rank3 = (us == WHITE) ? 0x0000000000FF0000ULL : 0x0000FF0000000000ULL;
    b = p->bb[us][PAWN];
    while (b) {
        int from = bb_pop_lsb(&b);
        U64 line = (SQ(from) & pinned) ? LINE_BB[ksq][from] : ~0ULL;

        int one = from + up;
        if (one >= 0 && one < 64 && !(all & SQ(one))) {
            U64 pushes = SQ(one);
            int two = one + up;
            if ((SQ(one) & rank3) && !(all & SQ(two))) pushes |= SQ(two);
            gen_add_pawn_moves(out, &n, from, pushes & target_mask & line, promo_rank);
        }
        U64 caps = PAWN_ATT[us][from] & theirs & target_mask & line;
        gen_add_pawn_moves(out, &n, from, caps, promo_rank);
    }

    /* En passant: rare and awkward (it removes two pieces from one rank at
     * once), so it is validated by testing the resulting position directly. */
    if (p->ep >= 0) {
        U64 cands = PAWN_ATT[them][p->ep] & p->bb[us][PAWN];
        int capsq = p->ep - up;
        while (cands) {
            int from = bb_pop_lsb(&cands);
            U64 occ2 = (all ^ SQ(from) ^ SQ(capsq)) | SQ(p->ep);
            U64 their_after = theirs ^ SQ(capsq);
            U64 bq = (p->bb[them][BISHOP] | p->bb[them][QUEEN]) & their_after;
            U64 rq = (p->bb[them][ROOK] | p->bb[them][QUEEN]) & their_after;
            U64 att = (bb_bishop_attacks(ksq, occ2) & bq)
                    | (bb_rook_attacks(ksq, occ2) & rq)
                    | (KNIGHT_ATT[ksq] & p->bb[them][KNIGHT] & their_after)
                    | (PAWN_ATT[us][ksq] & p->bb[them][PAWN] & their_after)
                    | (KING_ATT[ksq] & p->bb[them][KING]);
            if (!att) out[n++] = MK_MOVE(from, p->ep, 0, MV_EP);
        }
    }

    return n;
}

/* ------------------------------------------------------------------ */
/* Make move                                                           */
/* ------------------------------------------------------------------ */

static inline void pos_put(Pos *p, int c, int t, int sq)
{
    p->bb[c][t] |= SQ(sq);
    p->occ[c] |= SQ(sq);
    p->key ^= Z_PIECE[c][t][sq];
}

static inline void pos_clear(Pos *p, int c, int t, int sq)
{
    p->bb[c][t] &= ~SQ(sq);
    p->occ[c] &= ~SQ(sq);
    p->key ^= Z_PIECE[c][t][sq];
}

static const uint8_t CASTLE_MASK[64] = {
    ~(uint8_t)CR_WQ, 0xFF, 0xFF, 0xFF, (uint8_t)~(CR_WK | CR_WQ), 0xFF, 0xFF, ~(uint8_t)CR_WK,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    ~(uint8_t)CR_BQ, 0xFF, 0xFF, 0xFF, (uint8_t)~(CR_BK | CR_BQ), 0xFF, 0xFF, ~(uint8_t)CR_BK,
};

static void pos_make(Pos *p, Move m)
{
    const int us = p->side, them = !p->side;
    const int from = MV_FROM(m), to = MV_TO(m);
    const int promo = MV_PROMO(m);
    const unsigned flags = MV_FLAGS(m);
    const int up = (us == WHITE) ? 8 : -8;

    if (p->ep_key >= 0) p->key ^= Z_EP[p->ep_key & 7];
    p->key ^= Z_CASTLE[p->castling];

    int moved = pos_piece_at(p, us, from);
    int captured = -1;

    if (flags & MV_EP) {
        captured = PAWN;
        pos_clear(p, them, PAWN, to - up);
    } else if (p->occ[them] & SQ(to)) {
        captured = pos_piece_at(p, them, to);
        pos_clear(p, them, captured, to);
    }

    pos_clear(p, us, moved, from);
    pos_put(p, us, promo ? promo : moved, to);

    if (flags & MV_CASTLE) {
        int rf, rt;
        if (to == 6) { rf = 7; rt = 5; }
        else if (to == 2) { rf = 0; rt = 3; }
        else if (to == 62) { rf = 63; rt = 61; }
        else { rf = 56; rt = 59; }
        pos_clear(p, us, ROOK, rf);
        pos_put(p, us, ROOK, rt);
    }

    p->castling &= CASTLE_MASK[from] & CASTLE_MASK[to];
    p->key ^= Z_CASTLE[p->castling];

    p->ep = -1;
    p->ep_key = -1;
    if (moved == PAWN && (to - from == 2 * up)) {
        int epsq = from + up;
        p->ep = (int8_t)epsq;
        if (PAWN_ATT[us][epsq] & p->bb[them][PAWN]) {
            p->ep_key = (int8_t)epsq;
            p->key ^= Z_EP[epsq & 7];
        }
    }

    p->halfmove = (moved == PAWN || captured >= 0) ? 0 : (uint16_t)(p->halfmove + 1);
    p->all = p->occ[WHITE] | p->occ[BLACK];
    p->side = (uint8_t)them;
    p->key ^= Z_SIDE;
}

/* ------------------------------------------------------------------ */
/* FEN parsing (test/diagnostic paths only, never the search hot path)  */
/* ------------------------------------------------------------------ */
static int pos_from_fen(Pos *p, const char *fen)
{
    memset(p, 0, sizeof(*p));
    p->ep = -1;
    p->ep_key = -1;
    int rank = 7, file = 0;
    const char *s = fen;
    for (; *s && *s != ' '; s++) {
        char c = *s;
        if (c == '/') { rank--; file = 0; continue; }
        if (c >= '1' && c <= '8') { file += c - '0'; continue; }
        int color = (c >= 'a' && c <= 'z') ? BLACK : WHITE;
        char lc = (char)((c >= 'A' && c <= 'Z') ? c + 32 : c);
        int type;
        switch (lc) {
            case 'p': type = PAWN; break;
            case 'n': type = KNIGHT; break;
            case 'b': type = BISHOP; break;
            case 'r': type = ROOK; break;
            case 'q': type = QUEEN; break;
            case 'k': type = KING; break;
            default: return 0;
        }
        if (rank < 0 || rank > 7 || file < 0 || file > 7) return 0;
        p->bb[color][type] |= SQ(rank * 8 + file);
        file++;
    }
    while (*s == ' ') s++;
    p->side = (*s == 'b') ? BLACK : WHITE;
    while (*s && *s != ' ') s++;
    while (*s == ' ') s++;
    for (; *s && *s != ' '; s++) {
        if (*s == 'K') p->castling |= CR_WK;
        else if (*s == 'Q') p->castling |= CR_WQ;
        else if (*s == 'k') p->castling |= CR_BK;
        else if (*s == 'q') p->castling |= CR_BQ;
    }
    while (*s == ' ') s++;
    if (*s && *s != '-') {
        int f = s[0] - 'a', r = s[1] - '1';
        if (f >= 0 && f < 8 && r >= 0 && r < 8) p->ep = (int8_t)(r * 8 + f);
        while (*s && *s != ' ') s++;
    } else if (*s == '-') {
        s++;
    }
    while (*s == ' ') s++;
    if (*s >= '0' && *s <= '9') p->halfmove = (uint16_t)atoi(s);

    for (int c = 0; c < 2; c++) {
        p->occ[c] = 0;
        for (int t = 0; t < 6; t++) p->occ[c] |= p->bb[c][t];
    }
    p->all = p->occ[WHITE] | p->occ[BLACK];

    if (p->ep >= 0 && (PAWN_ATT[!p->side][p->ep] & p->bb[p->side][PAWN]))
        p->ep_key = p->ep;

    p->key = 0;
    for (int c = 0; c < 2; c++)
        for (int t = 0; t < 6; t++) {
            U64 b = p->bb[c][t];
            while (b) p->key ^= Z_PIECE[c][t][bb_pop_lsb(&b)];
        }
    p->key ^= Z_CASTLE[p->castling];
    if (p->ep_key >= 0) p->key ^= Z_EP[p->ep_key & 7];
    if (p->side == BLACK) p->key ^= Z_SIDE;
    return 1;
}

static void pos_move_uci(Move m, char *out)
{
    int from = MV_FROM(m), to = MV_TO(m);
    out[0] = (char)('a' + (from & 7));
    out[1] = (char)('1' + (from >> 3));
    out[2] = (char)('a' + (to & 7));
    out[3] = (char)('1' + (to >> 3));
    int promo = MV_PROMO(m);
    if (promo) {
        static const char pc[6] = {'p', 'n', 'b', 'r', 'q', 'k'};
        out[4] = pc[promo];
        out[5] = 0;
    } else {
        out[4] = 0;
    }
}

#endif /* ALPHACHESS_POSITION_H */
