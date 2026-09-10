/* Board/move encoding, bit-for-bit identical to alpha_chess.encoding.
 *
 * Both are side-to-move relative: when Black is to move the board is mirrored
 * vertically, which for a bitboard is a single byte swap.
 */
#ifndef ALPHACHESS_ENCODE_H
#define ALPHACHESS_ENCODE_H

#include "position.h"

#define ENC_PLANES 21
#define ENC_SIZE (ENC_PLANES * 64)
#define POLICY_SIZE 4672

/* Direction index for (sign(df), sign(dr)), laid out as (sdf+1)*3 + (sdr+1).
 * Matches encoding.DIRECTIONS; the centre entry (no movement) is unused. */
static const int8_t ENC_DIR[9] = {
    5,  /* (-1,-1) SW */
    6,  /* (-1, 0) W  */
    7,  /* (-1, 1) NW */
    4,  /* ( 0,-1) S  */
    -1, /* ( 0, 0)    */
    0,  /* ( 0, 1) N  */
    3,  /* ( 1,-1) SE */
    2,  /* ( 1, 0) E  */
    1,  /* ( 1, 1) NE */
};

static inline int enc_move_index(const Pos *p, Move m)
{
    int from = MV_FROM(m), to = MV_TO(m);
    if (p->side != WHITE) { from ^= 56; to ^= 56; }

    int df = (to & 7) - (from & 7);
    int dr = (to >> 3) - (from >> 3);
    int promo = MV_PROMO(m);

    if (promo && promo != QUEEN) {
        /* Underpromotion planes 64..72: (file delta, piece) with
         * KNIGHT/BISHOP/ROOK at offsets 0/1/2. */
        return from * 73 + 64 + (df + 1) * 3 + (promo - 1);
    }

    int adf = df < 0 ? -df : df;
    int adr = dr < 0 ? -dr : dr;
    if ((adf == 1 && adr == 2) || (adf == 2 && adr == 1)) {
        /* Knight planes 56..63, in encoding.KNIGHT_OFFSETS order. */
        static const int8_t KOFF[5][5] = {
            /* df+2, dr+2 */
            {-1, 61, -1, 62, -1},
            {60, -1, -1, -1, 63},
            {-1, -1, -1, -1, -1},
            {59, -1, -1, -1, 56},
            {-1, 58, -1, 57, -1},
        };
        return from * 73 + KOFF[df + 2][dr + 2];
    }

    int sdf = (df > 0) - (df < 0);
    int sdr = (dr > 0) - (dr < 0);
    int d = ENC_DIR[(sdf + 1) * 3 + (sdr + 1)];
    int n = adf > adr ? adf : adr;
    return from * 73 + d * 7 + (n - 1);
}

/* Scatter the set bits of ``b`` into ``plane`` as 1.0f. */
static inline void enc_scatter_f32(float *plane, U64 b)
{
    while (b) plane[bb_pop_lsb(&b)] = 1.0f;
}

static inline void enc_scatter_u8(uint8_t *plane, U64 b)
{
    while (b) plane[bb_pop_lsb(&b)] = 1;
}

/* Piece bitboards in the side-to-move frame: ours first, then theirs, each
 * mirrored vertically when Black is to move. */
static inline void enc_oriented(const Pos *p, U64 out[12])
{
    const int us = p->side, them = !p->side;
    if (us == WHITE) {
        for (int t = 0; t < 6; t++) {
            out[t] = p->bb[WHITE][t];
            out[6 + t] = p->bb[BLACK][t];
        }
    } else {
        for (int t = 0; t < 6; t++) {
            out[t] = bb_flip_vertical(p->bb[BLACK][t]);
            out[6 + t] = bb_flip_vertical(p->bb[WHITE][t]);
        }
    }
}

static inline int enc_castle_bits(const Pos *p, int color, int kingside)
{
    if (color == WHITE) return (p->castling & (kingside ? CR_WK : CR_WQ)) != 0;
    return (p->castling & (kingside ? CR_BK : CR_BQ)) != 0;
}

/* float32 network input; ``out`` must have room for ENC_SIZE floats. */
static void enc_planes_f32(const Pos *p, int rep_count, float *out)
{
    memset(out, 0, ENC_SIZE * sizeof(float));
    U64 oriented[12];
    enc_oriented(p, oriented);
    for (int i = 0; i < 12; i++) enc_scatter_f32(out + i * 64, oriented[i]);

    const int us = p->side, them = !p->side;
    float scalars[9];
    scalars[0] = (float)enc_castle_bits(p, us, 1);
    scalars[1] = (float)enc_castle_bits(p, us, 0);
    scalars[2] = (float)enc_castle_bits(p, them, 1);
    scalars[3] = (float)enc_castle_bits(p, them, 0);
    for (int i = 0; i < 4; i++)
        if (scalars[i] != 0.0f)
            for (int s = 0; s < 64; s++) out[(12 + i) * 64 + s] = 1.0f;

    if (p->ep >= 0) {
        int sq = (us == WHITE) ? p->ep : (p->ep ^ 56);
        out[16 * 64 + sq] = 1.0f;
    }

    int hm = p->halfmove > 100 ? 100 : p->halfmove;
    float hmv = (float)hm / 100.0f;
    for (int s = 0; s < 64; s++) out[17 * 64 + s] = hmv;
    if (rep_count >= 1)
        for (int s = 0; s < 64; s++) out[18 * 64 + s] = 1.0f;
    if (rep_count >= 2)
        for (int s = 0; s < 64; s++) out[19 * 64 + s] = 1.0f;
    for (int s = 0; s < 64; s++) out[20 * 64 + s] = 1.0f;
}

/* uint8 replay-buffer form (encoding.pack_state): every plane is binary except
 * plane 17, which keeps the raw halfmove counter. */
static void enc_planes_u8(const Pos *p, int rep_count, uint8_t *out)
{
    memset(out, 0, ENC_SIZE);
    U64 oriented[12];
    enc_oriented(p, oriented);
    for (int i = 0; i < 12; i++) enc_scatter_u8(out + i * 64, oriented[i]);

    const int us = p->side, them = !p->side;
    int bits[4];
    bits[0] = enc_castle_bits(p, us, 1);
    bits[1] = enc_castle_bits(p, us, 0);
    bits[2] = enc_castle_bits(p, them, 1);
    bits[3] = enc_castle_bits(p, them, 0);
    for (int i = 0; i < 4; i++)
        if (bits[i]) memset(out + (12 + i) * 64, 1, 64);

    if (p->ep >= 0) {
        int sq = (us == WHITE) ? p->ep : (p->ep ^ 56);
        out[16 * 64 + sq] = 1;
    }

    int hm = p->halfmove > 100 ? 100 : p->halfmove;
    memset(out + 17 * 64, (uint8_t)hm, 64);
    if (rep_count >= 1) memset(out + 18 * 64, 1, 64);
    if (rep_count >= 2) memset(out + 19 * 64, 1, 64);
    memset(out + 20 * 64, 1, 64);
}

#endif /* ALPHACHESS_ENCODE_H */
