/* Bitboard attack tables for the native search core.
 *
 * Everything here is initialised once by bb_init(): leaper attacks by direct
 * enumeration, slider attacks through fancy magic bitboards whose multipliers
 * are searched for at start-up with a fixed PRNG seed (deterministic, ~30ms,
 * and avoids shipping a table of hand-copied constants).
 */
#ifndef ALPHACHESS_BITBOARD_H
#define ALPHACHESS_BITBOARD_H

#include <stdint.h>
#include <string.h>

typedef uint64_t U64;

#define BB_ONE 1ULL
#define SQ(s) (BB_ONE << (s))

#define FILE_A_BB 0x0101010101010101ULL
#define FILE_H_BB 0x8080808080808080ULL
#define RANK_1_BB 0x00000000000000FFULL
#define RANK_8_BB 0xFF00000000000000ULL
#define DARK_SQUARES 0xAA55AA55AA55AA55ULL

static inline int bb_popcount(U64 b) { return __builtin_popcountll(b); }
static inline int bb_lsb(U64 b) { return __builtin_ctzll(b); }
static inline int bb_pop_lsb(U64 *b) { int s = __builtin_ctzll(*b); *b &= *b - 1; return s; }
/* Vertical mirror: rank r <-> 7-r, file unchanged (square s -> s ^ 56). */
static inline U64 bb_flip_vertical(U64 b) { return __builtin_bswap64(b); }

static U64 KNIGHT_ATT[64];
static U64 KING_ATT[64];
static U64 PAWN_ATT[2][64];      /* PAWN_ATT[color][sq] = squares that pawn attacks */
static U64 BETWEEN_BB[64][64];   /* exclusive of both endpoints, 0 if not aligned */
static U64 LINE_BB[64][64];      /* full line through both squares, 0 if not aligned */

typedef struct {
    U64 mask;
    U64 magic;
    U64 *table;
    unsigned shift;
} Magic;

static Magic ROOK_MAGIC[64];
static Magic BISHOP_MAGIC[64];
static U64 ROOK_TABLE[102400];
static U64 BISHOP_TABLE[5248];

static const int ROOK_DELTA[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
static const int BISHOP_DELTA[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};

static U64 bb_ray_attacks(int sq, U64 occ, const int deltas[4][2])
{
    U64 att = 0;
    int r = sq >> 3, f = sq & 7;
    for (int d = 0; d < 4; d++) {
        int rr = r, ff = f;
        for (;;) {
            rr += deltas[d][0];
            ff += deltas[d][1];
            if (rr < 0 || rr > 7 || ff < 0 || ff > 7) break;
            int s = rr * 8 + ff;
            att |= SQ(s);
            if (occ & SQ(s)) break;
        }
    }
    return att;
}

static U64 bb_rng_state = 0x9E3779B97F4A7C15ULL;

static U64 bb_rand64(void)
{
    U64 x = bb_rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    bb_rng_state = x;
    return x * 2685821657736338717ULL;
}

static U64 bb_sparse_rand(void) { return bb_rand64() & bb_rand64() & bb_rand64(); }

static void bb_init_magics(Magic *magics, U64 *table, const int deltas[4][2])
{
    U64 occupancies[4096], references[4096];
    int epoch[4096];
    int used[4096];
    memset(epoch, 0, sizeof(epoch));
    (void)used;
    int cur_epoch = 0;
    size_t offset = 0;

    for (int sq = 0; sq < 64; sq++) {
        /* Relevant occupancy: the ray attacks from an empty board, minus the
         * board edges (a blocker on the edge can never block anything). */
        U64 edges = ((RANK_1_BB | RANK_8_BB) & ~(RANK_1_BB << (8 * (sq >> 3))))
                  | ((FILE_A_BB | FILE_H_BB) & ~(FILE_A_BB << (sq & 7)));
        Magic *m = &magics[sq];
        m->mask = bb_ray_attacks(sq, 0, deltas) & ~edges;
        m->shift = 64 - (unsigned)bb_popcount(m->mask);
        m->table = table + offset;

        int size = 0;
        U64 b = 0;
        do { /* carry-rippler enumeration of every subset of the mask */
            occupancies[size] = b;
            references[size] = bb_ray_attacks(sq, b, deltas);
            size++;
            b = (b - m->mask) & m->mask;
        } while (b);
        offset += (size_t)size;

        for (;;) {
            do {
                m->magic = bb_sparse_rand();
            } while (bb_popcount((m->mask * m->magic) >> 56) < 6);

            cur_epoch++;
            int ok = 1;
            for (int i = 0; i < size; i++) {
                unsigned key = (unsigned)((occupancies[i] * m->magic) >> m->shift);
                if (epoch[key] != cur_epoch) {
                    epoch[key] = cur_epoch;
                    m->table[key] = references[i];
                } else if (m->table[key] != references[i]) {
                    ok = 0;
                    break;
                }
            }
            if (ok) break;
        }
    }
}

static inline U64 bb_rook_attacks(int sq, U64 occ)
{
    const Magic *m = &ROOK_MAGIC[sq];
    return m->table[((occ & m->mask) * m->magic) >> m->shift];
}

static inline U64 bb_bishop_attacks(int sq, U64 occ)
{
    const Magic *m = &BISHOP_MAGIC[sq];
    return m->table[((occ & m->mask) * m->magic) >> m->shift];
}

static inline U64 bb_queen_attacks(int sq, U64 occ)
{
    return bb_rook_attacks(sq, occ) | bb_bishop_attacks(sq, occ);
}

static int BB_READY = 0;

static void bb_init(void)
{
    if (BB_READY) return;

    static const int knight_off[8][2] = {
        {1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}
    };
    static const int king_off[8][2] = {
        {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}
    };

    for (int sq = 0; sq < 64; sq++) {
        int r = sq >> 3, f = sq & 7;
        U64 n = 0, k = 0;
        for (int i = 0; i < 8; i++) {
            int ff = f + knight_off[i][0], rr = r + knight_off[i][1];
            if (ff >= 0 && ff < 8 && rr >= 0 && rr < 8) n |= SQ(rr * 8 + ff);
            ff = f + king_off[i][0];
            rr = r + king_off[i][1];
            if (ff >= 0 && ff < 8 && rr >= 0 && rr < 8) k |= SQ(rr * 8 + ff);
        }
        KNIGHT_ATT[sq] = n;
        KING_ATT[sq] = k;

        U64 wp = 0, bp = 0;
        if (r < 7) {
            if (f > 0) wp |= SQ(sq + 7);
            if (f < 7) wp |= SQ(sq + 9);
        }
        if (r > 0) {
            if (f > 0) bp |= SQ(sq - 9);
            if (f < 7) bp |= SQ(sq - 7);
        }
        PAWN_ATT[0][sq] = wp;
        PAWN_ATT[1][sq] = bp;
    }

    bb_init_magics(ROOK_MAGIC, ROOK_TABLE, ROOK_DELTA);
    bb_init_magics(BISHOP_MAGIC, BISHOP_TABLE, BISHOP_DELTA);

    for (int a = 0; a < 64; a++) {
        for (int b = 0; b < 64; b++) {
            BETWEEN_BB[a][b] = 0;
            LINE_BB[a][b] = 0;
            if (a == b) continue;
            if (bb_rook_attacks(a, 0) & SQ(b)) {
                LINE_BB[a][b] = (bb_rook_attacks(a, 0) & bb_rook_attacks(b, 0)) | SQ(a) | SQ(b);
                BETWEEN_BB[a][b] = bb_rook_attacks(a, SQ(b)) & bb_rook_attacks(b, SQ(a));
            } else if (bb_bishop_attacks(a, 0) & SQ(b)) {
                LINE_BB[a][b] = (bb_bishop_attacks(a, 0) & bb_bishop_attacks(b, 0)) | SQ(a) | SQ(b);
                BETWEEN_BB[a][b] = bb_bishop_attacks(a, SQ(b)) & bb_bishop_attacks(b, SQ(a));
            }
        }
    }

    BB_READY = 1;
}

#endif /* ALPHACHESS_BITBOARD_H */
