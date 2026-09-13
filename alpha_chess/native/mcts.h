/* Native PUCT self-play engine.
 *
 * Mirrors alpha_chess.batched_selfplay.SelfPlayEngine one-for-one -- same pool
 * of concurrent games, same one-leaf-per-game-per-step batching, same subtree
 * reuse, same resignation accounting -- but with the board, the tree and the
 * encoding all in C.  The Python side is left with exactly two jobs: hand the
 * collected leaf batch to the GPU, and hand the results back.
 *
 * Two search refinements the Python engine does not have are configurable
 * here, both off when their knobs are left at their neutral values:
 *
 *   * First-play urgency (``fpu_reduction``): an unvisited child inherits the
 *     parent's value minus a penalty scaled by the explored prior mass,
 *     instead of a flat 0.  0.0 reproduces the Python behaviour.
 *   * Playout-cap randomisation (``full_search_prob`` < 1): most plies are
 *     searched with a small budget and produce no training target, while a
 *     random subset gets the full budget, root noise, and a recorded policy
 *     target.  1.0 reproduces the Python behaviour.
 */
#ifndef ALPHACHESS_MCTS_H
#define ALPHACHESS_MCTS_H

#include <math.h>
#include <stdlib.h>
#include "encode.h"

#define MAX_POLICY_TARGETS 80

/* ------------------------------------------------------------------ */
/* RNG: xoshiro256** plus the gamma variates the Dirichlet noise needs  */
/* ------------------------------------------------------------------ */
typedef struct { uint64_t s[4]; } RNG;

static inline uint64_t rot64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static uint64_t rng_next(RNG *r)
{
    uint64_t *s = r->s;
    const uint64_t result = rot64(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rot64(s[3], 45);
    return result;
}

static void rng_seed(RNG *r, uint64_t seed)
{
    for (int i = 0; i < 4; i++) {
        seed += 0x9E3779B97F4A7C15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        r->s[i] = z ^ (z >> 31);
    }
}

/* Uniform in (0, 1). */
static inline double rng_double(RNG *r)
{
    uint64_t x = (rng_next(r) >> 11) + 1;
    return (double)x * (1.0 / 9007199254740993.0);
}

static double rng_normal(RNG *r)
{
    double u1 = rng_double(r), u2 = rng_double(r);
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

/* Marsaglia-Tsang, with the standard boost for alpha < 1. */
static double rng_gamma(RNG *r, double a)
{
    double boost = 1.0;
    if (a < 1.0) {
        boost = pow(rng_double(r), 1.0 / a);
        a += 1.0;
    }
    const double d = a - 1.0 / 3.0;
    const double c = 1.0 / sqrt(9.0 * d);
    for (;;) {
        double x, v;
        do {
            x = rng_normal(r);
            v = 1.0 + c * x;
        } while (v <= 0.0);
        v = v * v * v;
        double u = rng_double(r);
        if (u < 1.0 - 0.0331 * x * x * x * x) return boost * d * v;
        if (log(u) < 0.5 * x * x + d * (1.0 - v + log(v))) return boost * d * v;
    }
}

/* ------------------------------------------------------------------ */
/* Repetition table: open addressing, one per game                      */
/* ------------------------------------------------------------------ */
#define REP_CAP 2048
#define REP_MASK (REP_CAP - 1)

typedef struct {
    U64 keys[REP_CAP];
    uint16_t counts[REP_CAP];
} RepTable;

static void rep_reset(RepTable *t) { memset(t, 0, sizeof(*t)); }

static inline int rep_get(const RepTable *t, U64 key)
{
    unsigned i = (unsigned)(key >> 40) & REP_MASK;
    for (;;) {
        if (t->counts[i] == 0) return 0;
        if (t->keys[i] == key) return t->counts[i];
        i = (i + 1) & REP_MASK;
    }
}

static inline void rep_add(RepTable *t, U64 key)
{
    unsigned i = (unsigned)(key >> 40) & REP_MASK;
    for (;;) {
        if (t->counts[i] == 0) { t->keys[i] = key; t->counts[i] = 1; return; }
        if (t->keys[i] == key) { t->counts[i]++; return; }
        i = (i + 1) & REP_MASK;
    }
}

/* ------------------------------------------------------------------ */
/* Search tree                                                          */
/* ------------------------------------------------------------------ */
typedef struct {
    int32_t child_base;
    int32_t n_moves;
    int32_t n_total;
    float terminal_value;
    uint8_t expanded;
    uint8_t is_terminal;
} Node;

typedef struct {
    Node *nodes;
    int32_t n_nodes, cap_nodes;
    Move *cmove;
    int32_t *cidx;
    float *cP, *cW, *cQ;
    int32_t *cN, *cchild;
    int32_t n_child, cap_child;
} Arena;

static void arena_init(Arena *a)
{
    memset(a, 0, sizeof(*a));
}

static void arena_free(Arena *a)
{
    free(a->nodes); free(a->cmove); free(a->cidx);
    free(a->cP); free(a->cW); free(a->cQ); free(a->cN); free(a->cchild);
    memset(a, 0, sizeof(*a));
}

static void arena_reset(Arena *a) { a->n_nodes = 0; a->n_child = 0; }

static int arena_grow_nodes(Arena *a, int32_t need)
{
    if (need <= a->cap_nodes) return 1;
    int32_t cap = a->cap_nodes ? a->cap_nodes : 64;
    while (cap < need) cap *= 2;
    Node *p = (Node *)realloc(a->nodes, (size_t)cap * sizeof(Node));
    if (!p) return 0;
    a->nodes = p;
    a->cap_nodes = cap;
    return 1;
}

static int arena_grow_children(Arena *a, int32_t need)
{
    if (need <= a->cap_child) return 1;
    int32_t cap = a->cap_child ? a->cap_child : 512;
    while (cap < need) cap *= 2;
#define REGROW(field, type)                                                    \
    do {                                                                       \
        type *tmp = (type *)realloc(a->field, (size_t)cap * sizeof(type));     \
        if (!tmp) return 0;                                                    \
        a->field = tmp;                                                        \
    } while (0)
    REGROW(cmove, Move);
    REGROW(cidx, int32_t);
    REGROW(cP, float);
    REGROW(cW, float);
    REGROW(cQ, float);
    REGROW(cN, int32_t);
    REGROW(cchild, int32_t);
#undef REGROW
    a->cap_child = cap;
    return 1;
}

static int32_t arena_new_node(Arena *a)
{
    if (!arena_grow_nodes(a, a->n_nodes + 1)) return -1;
    int32_t i = a->n_nodes++;
    Node *n = &a->nodes[i];
    n->child_base = -1;
    n->n_moves = 0;
    n->n_total = 0;
    n->terminal_value = 0.0f;
    n->expanded = 0;
    n->is_terminal = 0;
    return i;
}

static int32_t arena_alloc_children(Arena *a, int32_t count)
{
    if (!arena_grow_children(a, a->n_child + count)) return -1;
    int32_t base = a->n_child;
    a->n_child += count;
    return base;
}

/* Copy the subtree rooted at ``node`` from ``src`` into a freshly reset
 * ``dst``.  Used for subtree reuse: the played move's child keeps its visits
 * while everything else is dropped, without a per-node free list. */
static int32_t arena_copy_subtree(Arena *dst, const Arena *src, int32_t node,
                                  int32_t **stack, int32_t *stack_cap)
{
    arena_reset(dst);
    if (node < 0) return -1;
    int32_t root = arena_new_node(dst);
    if (root < 0) return -1;

    int32_t sp = 0;
    if (*stack_cap < 64) {
        int32_t *tmp = (int32_t *)realloc(*stack, 64 * sizeof(int32_t) * 2);
        if (!tmp) return -1;
        *stack = tmp;
        *stack_cap = 64;
    }
    (*stack)[sp * 2] = node;
    (*stack)[sp * 2 + 1] = root;
    sp = 1;

    while (sp > 0) {
        sp--;
        int32_t s = (*stack)[sp * 2];
        int32_t d = (*stack)[sp * 2 + 1];
        const Node *sn = &src->nodes[s];
        int32_t nm = sn->n_moves;

        dst->nodes[d].n_total = sn->n_total;
        dst->nodes[d].terminal_value = sn->terminal_value;
        dst->nodes[d].expanded = sn->expanded;
        dst->nodes[d].is_terminal = sn->is_terminal;
        dst->nodes[d].n_moves = nm;
        dst->nodes[d].child_base = -1;
        if (!sn->expanded || nm == 0) continue;

        int32_t base = arena_alloc_children(dst, nm);
        if (base < 0) return -1;
        dst->nodes[d].child_base = base;
        int32_t sb = sn->child_base;
        for (int32_t j = 0; j < nm; j++) {
            dst->cmove[base + j] = src->cmove[sb + j];
            dst->cidx[base + j] = src->cidx[sb + j];
            dst->cP[base + j] = src->cP[sb + j];
            dst->cW[base + j] = src->cW[sb + j];
            dst->cQ[base + j] = src->cQ[sb + j];
            dst->cN[base + j] = src->cN[sb + j];
            dst->cchild[base + j] = -1;
        }
        for (int32_t j = 0; j < nm; j++) {
            int32_t sc = src->cchild[sb + j];
            if (sc < 0) continue;
            int32_t dc = arena_new_node(dst);
            if (dc < 0) return -1;
            dst->cchild[base + j] = dc;
            if (sp + 1 > *stack_cap) {
                int32_t cap = *stack_cap * 2;
                int32_t *tmp = (int32_t *)realloc(*stack, (size_t)cap * sizeof(int32_t) * 2);
                if (!tmp) return -1;
                *stack = tmp;
                *stack_cap = cap;
            }
            (*stack)[sp * 2] = sc;
            (*stack)[sp * 2 + 1] = dc;
            sp++;
        }
    }
    return root;
}

/* ------------------------------------------------------------------ */
/* Games                                                                */
/* ------------------------------------------------------------------ */
typedef struct {
    Pos pos;
    Arena arena, spare;
    int32_t *copy_stack;
    int32_t copy_stack_cap;
    int32_t root;
    RepTable rep;
    int cur_rep;
    int noise_pending;
    int sims_left;
    int move_count;
    int record_ply;      /* this ply gets a recorded training target */
    int allow_resign, resign_streak[2], resigned;
    int would_resign_at, would_resign_side;
    int active;

    /* Examples accumulated for this game, flushed when it ends. */
    uint8_t *ex_states;
    uint16_t *ex_pidx;
    float *ex_pval;
    uint8_t *ex_turn;
    int ex_n, ex_cap;
} Game;

static void game_free(Game *g)
{
    arena_free(&g->arena);
    arena_free(&g->spare);
    free(g->copy_stack);
    free(g->ex_states);
    free(g->ex_pidx);
    free(g->ex_pval);
    free(g->ex_turn);
    memset(g, 0, sizeof(*g));
}

static int game_grow_examples(Game *g)
{
    if (g->ex_n < g->ex_cap) return 1;
    int cap = g->ex_cap ? g->ex_cap * 2 : 128;
    uint8_t *s = (uint8_t *)realloc(g->ex_states, (size_t)cap * ENC_SIZE);
    if (!s) return 0;
    g->ex_states = s;
    uint16_t *pi = (uint16_t *)realloc(g->ex_pidx, (size_t)cap * MAX_POLICY_TARGETS * sizeof(uint16_t));
    if (!pi) return 0;
    g->ex_pidx = pi;
    float *pv = (float *)realloc(g->ex_pval, (size_t)cap * MAX_POLICY_TARGETS * sizeof(float));
    if (!pv) return 0;
    g->ex_pval = pv;
    uint8_t *t = (uint8_t *)realloc(g->ex_turn, (size_t)cap);
    if (!t) return 0;
    g->ex_turn = t;
    g->ex_cap = cap;
    return 1;
}

/* ------------------------------------------------------------------ */
/* Engine                                                               */
/* ------------------------------------------------------------------ */
typedef struct {
    Game *games;
    int n_games;

    int num_simulations;
    int fast_simulations;
    double full_search_prob;
    double c_puct;
    double dirichlet_alpha, dirichlet_epsilon;
    double fpu_reduction;
    int noise_all_plies;
    int temperature_moves;
    int max_moves;
    int use_resign;
    double resign_threshold;
    int resign_plies;
    double resign_disable_fraction;

    long games_target, games_started;

    /* Leaves awaiting evaluation. */
    int32_t *pend_game, *pend_node, *pend_nmoves;
    int32_t *pend_path_off, *pend_path_len, *pend_move_off;
    int32_t *path_node, *path_slot;
    U64 *path_key;
    int32_t path_used, path_cap;
    Move *pend_moves;
    int32_t *pend_idx;
    int32_t move_used, move_cap;
    int n_pend;

    /* Finished-game examples awaiting drain(). */
    uint8_t *out_states;
    uint16_t *out_pidx;
    float *out_pval;
    int16_t *out_plen;
    float *out_values;
    int out_n, out_cap;

    double st_games, st_plies, st_evals;
    double st_resigned, st_resign_checked, st_resign_fp;
    double st_full_plies, st_fast_plies, st_noise_plies;

    RNG rng;
    double *noise_buf;
    int noise_cap;
    int failed;
} Engine;

static int engine_grow_out(Engine *e, int need)
{
    if (need <= e->out_cap) return 1;
    int cap = e->out_cap ? e->out_cap : 4096;
    while (cap < need) cap *= 2;
    uint8_t *s = (uint8_t *)realloc(e->out_states, (size_t)cap * ENC_SIZE);
    if (!s) return 0;
    e->out_states = s;
    uint16_t *pi = (uint16_t *)realloc(e->out_pidx, (size_t)cap * MAX_POLICY_TARGETS * sizeof(uint16_t));
    if (!pi) return 0;
    e->out_pidx = pi;
    float *pv = (float *)realloc(e->out_pval, (size_t)cap * MAX_POLICY_TARGETS * sizeof(float));
    if (!pv) return 0;
    e->out_pval = pv;
    int16_t *pl = (int16_t *)realloc(e->out_plen, (size_t)cap * sizeof(int16_t));
    if (!pl) return 0;
    e->out_plen = pl;
    float *v = (float *)realloc(e->out_values, (size_t)cap * sizeof(float));
    if (!v) return 0;
    e->out_values = v;
    e->out_cap = cap;
    return 1;
}

static int engine_grow_path(Engine *e, int32_t need)
{
    if (need <= e->path_cap) return 1;
    int32_t cap = e->path_cap ? e->path_cap : 4096;
    while (cap < need) cap *= 2;
    int32_t *a = (int32_t *)realloc(e->path_node, (size_t)cap * sizeof(int32_t));
    if (!a) return 0;
    e->path_node = a;
    int32_t *b = (int32_t *)realloc(e->path_slot, (size_t)cap * sizeof(int32_t));
    if (!b) return 0;
    e->path_slot = b;
    U64 *c = (U64 *)realloc(e->path_key, (size_t)cap * sizeof(U64));
    if (!c) return 0;
    e->path_key = c;
    e->path_cap = cap;
    return 1;
}

static int engine_grow_moves(Engine *e, int32_t need)
{
    if (need <= e->move_cap) return 1;
    int32_t cap = e->move_cap ? e->move_cap : 8192;
    while (cap < need) cap *= 2;
    Move *m = (Move *)realloc(e->pend_moves, (size_t)cap * sizeof(Move));
    if (!m) return 0;
    e->pend_moves = m;
    int32_t *i = (int32_t *)realloc(e->pend_idx, (size_t)cap * sizeof(int32_t));
    if (!i) return 0;
    e->pend_idx = i;
    e->move_cap = cap;
    return 1;
}

/* PUCT child selection.  ``fpu`` is the value assigned to children with no
 * visits yet (see the file header). */
static inline int32_t select_child(const Engine *e, const Arena *a, const Node *n)
{
    const int32_t base = n->child_base;
    const int32_t nm = n->n_moves;
    const float *P = a->cP + base;
    const float *Q = a->cQ + base;
    const int32_t *N = a->cN + base;

    if (n->n_total == 0) {
        /* No visits anywhere below: sqrt(0) zeroes the exploration term and the
         * prior alone decides. */
        int32_t best = 0;
        float best_p = -1e30f;
        for (int32_t i = 0; i < nm; i++)
            if (P[i] > best_p) { best_p = P[i]; best = i; }
        return best;
    }

    const float coeff = (float)(e->c_puct * sqrt((double)n->n_total));
    float fpu = 0.0f;
    if (e->fpu_reduction != 0.0) {
        /* Leela-style: unvisited children inherit the node's own value,
         * discounted by how much prior mass has already been explored. */
        float explored = 0.0f;
        double sum_w = 0.0;
        int32_t sum_n = 0;
        for (int32_t i = 0; i < nm; i++)
            if (N[i] > 0) { explored += P[i]; sum_w += a->cW[base + i]; sum_n += N[i]; }
        float parent_q = sum_n > 0 ? (float)(sum_w / (double)sum_n) : 0.0f;
        fpu = parent_q - (float)e->fpu_reduction * sqrtf(explored);
    }

    int32_t best = 0;
    float best_score = -1e30f;
    for (int32_t i = 0; i < nm; i++) {
        float q = N[i] > 0 ? Q[i] : fpu;
        float score = q + coeff * P[i] / (float)(1 + N[i]);
        if (score > best_score) { best_score = score; best = i; }
    }
    return best;
}

static inline void backup(Arena *a, const int32_t *pn, const int32_t *ps,
                          int32_t len, float value)
{
    for (int32_t k = len - 1; k >= 0; k--) {
        value = -value;
        Node *parent = &a->nodes[pn[k]];
        int32_t c = parent->child_base + ps[k];
        int32_t n = a->cN[c] + 1;
        float w = a->cW[c] + value;
        a->cN[c] = n;
        a->cW[c] = w;
        a->cQ[c] = w / (float)n;
        parent->n_total++;
    }
}

/* Terminal value from the side-to-move's view, or 2.0f when still in play.
 * ``n_moves`` is the caller's already generated legal-move count. */
static inline float resolve_terminal(const Pos *p, int n_moves, int rep_count)
{
    if (n_moves == 0) return pos_in_check(p) ? -1.0f : 0.0f;
    if (rep_count >= 2) return 0.0f;
    if (p->halfmove >= 100) return 0.0f;
    if (pos_insufficient_material(p)) return 0.0f;
    return 2.0f;
}

static void add_root_noise(Engine *e, Game *g)
{
    Node *root = &g->arena.nodes[g->root];
    int32_t nm = root->n_moves;
    if (nm <= 0) return;
    if (nm > e->noise_cap) {
        double *tmp = (double *)realloc(e->noise_buf, (size_t)nm * sizeof(double));
        if (!tmp) return;
        e->noise_buf = tmp;
        e->noise_cap = nm;
    }
    double total = 0.0;
    for (int32_t i = 0; i < nm; i++) {
        double x = rng_gamma(&e->rng, e->dirichlet_alpha);
        e->noise_buf[i] = x;
        total += x;
    }
    if (total <= 0.0) return;
    const double eps = e->dirichlet_epsilon;
    float *P = g->arena.cP + root->child_base;
    for (int32_t i = 0; i < nm; i++)
        P[i] = (float)((1.0 - eps) * P[i] + eps * (e->noise_buf[i] / total));
}

static void start_ply(Engine *e, Game *g, int carried_visits)
{
    /* Playout-cap randomisation: a random subset of plies gets the full
     * budget and a training target; the rest run cheap. */
    g->record_ply = (e->full_search_prob >= 1.0)
                  || (rng_double(&e->rng) < e->full_search_prob);
    int target = g->record_ply ? e->num_simulations : e->fast_simulations;
    int left = target - carried_visits;
    g->sims_left = left > 1 ? left : 1;
    /* Root noise is what makes self-play games diverge, and it is free --- only
     * *recording* a ply costs anything. Bundling the two means that at
     * full_search_prob 0.25 three quarters of the moves actually played carry
     * no exploration noise at all, which lets the policy collapse onto one
     * opening. ``noise_all_plies`` unbundles them. */
    g->noise_pending = g->record_ply || e->noise_all_plies;
    if (g->record_ply) e->st_full_plies += 1.0; else e->st_fast_plies += 1.0;
}

static void new_game(Engine *e, Game *g)
{
    arena_reset(&g->arena);
    arena_reset(&g->spare);
    pos_set_start(&g->pos);
    rep_reset(&g->rep);
    rep_add(&g->rep, g->pos.key);
    g->cur_rep = 0;
    g->root = arena_new_node(&g->arena);
    g->move_count = 0;
    g->resign_streak[0] = g->resign_streak[1] = 0;
    g->resigned = 0;
    g->would_resign_at = -1;
    g->would_resign_side = -1;
    g->allow_resign = e->use_resign
        && (rng_double(&e->rng) >= e->resign_disable_fraction);
    g->ex_n = 0;
    g->active = 1;
    start_ply(e, g, 0);
}

/* ------------------------------------------------------------------ */
/* Collect: one descent per active game, batched for the network        */
/* ------------------------------------------------------------------ */
static int engine_collect(Engine *e, float *states, int32_t *idx,
                          int32_t *counts, int idx_stride)
{
    e->n_pend = 0;
    e->path_used = 0;
    e->move_used = 0;

    for (int gi = 0; gi < e->n_games; gi++) {
        Game *g = &e->games[gi];
        if (!g->active || g->sims_left <= 0) continue;

        Arena *a = &g->arena;
        Pos pos = g->pos;
        int32_t node = g->root;
        int32_t path_start = e->path_used;
        int32_t path_len = 0;
        int rep_count = rep_get(&g->rep, pos.key) - 1;
        if (rep_count < 0) rep_count = 0;

        /* Descend to a leaf, tracking repetitions of the search line as we go
         * rather than replaying a move stack. */
        while (a->nodes[node].expanded && !a->nodes[node].is_terminal) {
            int32_t slot = select_child(e, a, &a->nodes[node]);
            if (!engine_grow_path(e, e->path_used + 1)) { e->failed = 1; return -1; }
            e->path_node[e->path_used] = node;
            e->path_slot[e->path_used] = slot;
            e->path_used++;
            path_len++;

            int32_t base = a->nodes[node].child_base;
            pos_make(&pos, a->cmove[base + slot]);

            /* Occurrences along the real game line, plus any earlier in this
             * descent -- the search line can repeat a position too. */
            int seen = rep_get(&g->rep, pos.key);
            for (int32_t k = path_start; k < e->path_used - 1; k++)
                if (e->path_key[k] == pos.key) seen++;
            e->path_key[e->path_used - 1] = pos.key;
            rep_count = seen;

            int32_t child = a->cchild[base + slot];
            if (child < 0) {
                child = arena_new_node(a);
                if (child < 0) { e->failed = 1; return -1; }
                a->cchild[base + slot] = child;
            }
            node = child;
        }

        if (a->nodes[node].is_terminal) {
            backup(a, e->path_node + path_start, e->path_slot + path_start,
                   path_len, a->nodes[node].terminal_value);
            g->sims_left--;
            e->path_used = path_start;
            continue;
        }

        Move moves[MAX_MOVES];
        int n_moves = pos_gen_legal(&pos, moves);
        float term = resolve_terminal(&pos, n_moves, rep_count);
        if (term != 2.0f) {
            a->nodes[node].is_terminal = 1;
            a->nodes[node].terminal_value = term;
            backup(a, e->path_node + path_start, e->path_slot + path_start,
                   path_len, term);
            g->sims_left--;
            e->path_used = path_start;
            continue;
        }

        if (!engine_grow_moves(e, e->move_used + n_moves)) { e->failed = 1; return -1; }
        int32_t moff = e->move_used;
        int k = e->n_pend;
        float *dst = states + (size_t)k * ENC_SIZE;
        enc_planes_f32(&pos, rep_count, dst);
        int32_t *irow = idx + (size_t)k * idx_stride;
        for (int j = 0; j < n_moves; j++) {
            int mi = enc_move_index(&pos, moves[j]);
            e->pend_moves[moff + j] = moves[j];
            e->pend_idx[moff + j] = mi;
            irow[j] = mi;
        }
        if (idx_stride > n_moves)
            memset(irow + n_moves, 0, (size_t)(idx_stride - n_moves) * sizeof(int32_t));
        counts[k] = n_moves;
        e->move_used += n_moves;

        e->pend_game[k] = gi;
        e->pend_node[k] = node;
        e->pend_nmoves[k] = n_moves;
        e->pend_path_off[k] = path_start;
        e->pend_path_len[k] = path_len;
        e->pend_move_off[k] = moff;
        e->n_pend++;
    }
    return e->n_pend;
}

/* ------------------------------------------------------------------ */
/* Playing a move                                                       */
/* ------------------------------------------------------------------ */
static int record_example(Engine *e, Game *g)
{
    if (!game_grow_examples(g)) return 0;
    Arena *a = &g->arena;
    Node *root = &a->nodes[g->root];
    int32_t nm = root->n_moves;
    int32_t base = root->child_base;

    int32_t sel_idx[MAX_POLICY_TARGETS];
    double sel_n[MAX_POLICY_TARGETS];
    int k = 0;

    if (nm <= MAX_POLICY_TARGETS) {
        for (int32_t i = 0; i < nm; i++)
            if (a->cN[base + i] > 0) {
                sel_idx[k] = a->cidx[base + i];
                sel_n[k] = (double)a->cN[base + i];
                k++;
            }
        if (k == 0) {
            /* No simulation reached a child: fall back to uniform over the
             * moves the root already knows about. */
            int32_t lim = nm < MAX_POLICY_TARGETS ? nm : MAX_POLICY_TARGETS;
            for (int32_t i = 0; i < lim; i++) {
                sel_idx[k] = a->cidx[base + i];
                sel_n[k] = 1.0;
                k++;
            }
        }
    } else {
        /* More visited moves than target slots: keep the most visited. */
        for (int32_t i = 0; i < nm; i++) {
            int32_t n = a->cN[base + i];
            if (n <= 0) continue;
            if (k < MAX_POLICY_TARGETS) {
                sel_idx[k] = a->cidx[base + i];
                sel_n[k] = (double)n;
                k++;
            } else {
                int worst = 0;
                for (int j = 1; j < k; j++) if (sel_n[j] < sel_n[worst]) worst = j;
                if ((double)n > sel_n[worst]) {
                    sel_n[worst] = (double)n;
                    sel_idx[worst] = a->cidx[base + i];
                }
            }
        }
        if (k == 0) {
            for (int32_t i = 0; i < MAX_POLICY_TARGETS; i++) {
                sel_idx[k] = a->cidx[base + i];
                sel_n[k] = 1.0;
                k++;
            }
        }
    }

    double total = 0.0;
    for (int i = 0; i < k; i++) total += sel_n[i];
    if (total <= 0.0) total = 1.0;

    int slot = g->ex_n;
    uint16_t *pi = g->ex_pidx + (size_t)slot * MAX_POLICY_TARGETS;
    float *pv = g->ex_pval + (size_t)slot * MAX_POLICY_TARGETS;
    memset(pi, 0, MAX_POLICY_TARGETS * sizeof(uint16_t));
    memset(pv, 0, MAX_POLICY_TARGETS * sizeof(float));
    for (int i = 0; i < k; i++) {
        pi[i] = (uint16_t)sel_idx[i];
        pv[i] = (float)(sel_n[i] / total);
    }
    enc_planes_u8(&g->pos, g->cur_rep, g->ex_states + (size_t)slot * ENC_SIZE);
    g->ex_turn[slot] = (uint8_t)(g->pos.side == WHITE);
    g->ex_n++;
    return 1;
}

static int should_resign(Engine *e, Game *g)
{
    if (!e->use_resign) return 0;
    Arena *a = &g->arena;
    Node *root = &a->nodes[g->root];
    int32_t base = root->child_base;
    float best_q = -1.0f;
    int seen = 0;
    for (int32_t i = 0; i < root->n_moves; i++)
        if (a->cN[base + i] > 0) {
            seen = 1;
            if (a->cQ[base + i] > best_q) best_q = a->cQ[base + i];
        }
    /* The streak is kept PER SIDE. Consecutive plies alternate the mover, and
     * in a zero-sum game the two sides' root values are opposite: a lost side's
     * -0.95 is always followed by the winner's +0.95. A single shared counter
     * therefore resets every ply and can never exceed 1, which made any
     * resign_plies above 1 unsatisfiable and resignation silently dead. */
    const int side = (int)g->pos.side;
    if (!seen) { g->resign_streak[side] = 0; return 0; }

    if (best_q <= (float)e->resign_threshold) g->resign_streak[side]++;
    else g->resign_streak[side] = 0;

    if (g->resign_streak[side] < e->resign_plies) return 0;
    if (g->would_resign_at < 0) {
        g->would_resign_at = g->move_count;
        g->would_resign_side = g->pos.side;
    }
    return g->allow_resign;
}

static int32_t choose_move(Engine *e, Game *g)
{
    Arena *a = &g->arena;
    Node *root = &a->nodes[g->root];
    int32_t base = root->child_base;
    int32_t nm = root->n_moves;
    if (g->move_count < e->temperature_moves && root->n_total > 0) {
        double r = rng_double(&e->rng) * (double)root->n_total;
        double acc = 0.0;
        for (int32_t i = 0; i < nm; i++) {
            acc += (double)a->cN[base + i];
            if (r <= acc) return i;
        }
        return nm - 1;
    }
    int32_t best = 0;
    int32_t best_n = -1;
    for (int32_t i = 0; i < nm; i++)
        if (a->cN[base + i] > best_n) { best_n = a->cN[base + i]; best = i; }
    return best;
}

static void harvest(Engine *e, Game *g, float result)
{
    if (!engine_grow_out(e, e->out_n + g->ex_n)) { e->failed = 1; return; }
    for (int i = 0; i < g->ex_n; i++) {
        int o = e->out_n + i;
        memcpy(e->out_states + (size_t)o * ENC_SIZE,
               g->ex_states + (size_t)i * ENC_SIZE, ENC_SIZE);
        memcpy(e->out_pidx + (size_t)o * MAX_POLICY_TARGETS,
               g->ex_pidx + (size_t)i * MAX_POLICY_TARGETS,
               MAX_POLICY_TARGETS * sizeof(uint16_t));
        const float *pv = g->ex_pval + (size_t)i * MAX_POLICY_TARGETS;
        memcpy(e->out_pval + (size_t)o * MAX_POLICY_TARGETS, pv,
               MAX_POLICY_TARGETS * sizeof(float));
        int16_t len = 0;
        for (int j = 0; j < MAX_POLICY_TARGETS; j++) if (pv[j] > 0.0f) len++;
        e->out_plen[o] = len;
        e->out_values[o] = g->ex_turn[i] ? result : -result;
    }
    e->out_n += g->ex_n;
}

/* Play one move; returns 2.0f while the game continues, else the White-side
 * result. */
static float advance_game(Engine *e, Game *g)
{
    if (g->record_ply && !record_example(e, g)) { e->failed = 1; return 0.0f; }

    if (should_resign(e, g)) {
        g->resigned = 1;
        return g->pos.side == WHITE ? -1.0f : 1.0f;
    }

    Arena *a = &g->arena;
    int32_t slot = choose_move(e, g);
    int32_t base = a->nodes[g->root].child_base;
    Move move = a->cmove[base + slot];
    int32_t child = a->cchild[base + slot];

    pos_make(&g->pos, move);
    g->move_count++;

    int seen = rep_get(&g->rep, g->pos.key);
    rep_add(&g->rep, g->pos.key);
    g->cur_rep = seen;

    /* Subtree reuse: keep the played move's subtree, drop the siblings. */
    int32_t carried = 0;
    if (child >= 0) {
        int32_t new_root = arena_copy_subtree(&g->spare, &g->arena, child,
                                              &g->copy_stack, &g->copy_stack_cap);
        if (new_root < 0) { e->failed = 1; return 0.0f; }
        Arena tmp = g->arena;
        g->arena = g->spare;
        g->spare = tmp;
        g->root = new_root;
        carried = g->arena.nodes[new_root].n_total;
    } else {
        arena_reset(&g->spare);
        Arena tmp = g->arena;
        g->arena = g->spare;
        g->spare = tmp;
        g->root = arena_new_node(&g->arena);
        if (g->root < 0) { e->failed = 1; return 0.0f; }
    }

    Move moves[MAX_MOVES];
    int n_moves = pos_gen_legal(&g->pos, moves);
    float term = resolve_terminal(&g->pos, n_moves, g->cur_rep);
    if (term != 2.0f) {
        if (term == -1.0f) return g->pos.side == WHITE ? -1.0f : 1.0f;
        return 0.0f;
    }
    if (g->move_count >= e->max_moves) return 0.0f;

    start_ply(e, g, carried);
    if (g->arena.nodes[g->root].expanded && g->noise_pending) {
        add_root_noise(e, g);
        e->st_noise_plies += 1.0;
        g->noise_pending = 0;
    }
    return 2.0f;
}

/* ------------------------------------------------------------------ */
/* Apply: expand the evaluated leaves, back up, then play out any game  */
/* that has spent its budget.                                           */
/* ------------------------------------------------------------------ */
static void engine_apply(Engine *e, const float *priors, const float *values,
                         int prior_stride)
{
    for (int k = 0; k < e->n_pend; k++) {
        Game *g = &e->games[e->pend_game[k]];
        Arena *a = &g->arena;
        int32_t node = e->pend_node[k];
        int32_t nm = e->pend_nmoves[k];
        int32_t moff = e->pend_move_off[k];

        int32_t base = arena_alloc_children(a, nm);
        if (base < 0) { e->failed = 1; return; }
        const float *row = priors + (size_t)k * prior_stride;
        for (int32_t j = 0; j < nm; j++) {
            a->cmove[base + j] = e->pend_moves[moff + j];
            a->cidx[base + j] = e->pend_idx[moff + j];
            a->cP[base + j] = row[j];
            a->cW[base + j] = 0.0f;
            a->cQ[base + j] = 0.0f;
            a->cN[base + j] = 0;
            a->cchild[base + j] = -1;
        }
        a->nodes[node].child_base = base;
        a->nodes[node].n_moves = nm;
        a->nodes[node].expanded = 1;

        if (node == g->root && g->noise_pending) {
            add_root_noise(e, g);
            g->noise_pending = 0;
        }
        backup(a, e->path_node + e->pend_path_off[k],
               e->path_slot + e->pend_path_off[k], e->pend_path_len[k],
               values[k]);
        g->sims_left--;
    }
    e->st_evals += (double)e->n_pend;
    e->n_pend = 0;

    for (int gi = 0; gi < e->n_games; gi++) {
        Game *g = &e->games[gi];
        if (!g->active || g->sims_left > 0) continue;
        float result = advance_game(e, g);
        if (e->failed) return;
        if (result == 2.0f) continue;

        harvest(e, g, result);
        if (e->failed) return;
        e->st_games += 1.0;
        e->st_plies += (double)g->ex_n;
        if (g->resigned) e->st_resigned += 1.0;
        if (!g->allow_resign && g->would_resign_at >= 0) {
            e->st_resign_checked += 1.0;
            int lost = g->would_resign_side == WHITE ? (result < 0.0f) : (result > 0.0f);
            if (!lost) e->st_resign_fp += 1.0;
        }

        if (e->games_started < e->games_target) {
            new_game(e, g);
            e->games_started++;
        } else {
            g->active = 0;
        }
    }
}

static int engine_done(const Engine *e)
{
    for (int i = 0; i < e->n_games; i++) if (e->games[i].active) return 0;
    return 1;
}

#endif /* ALPHACHESS_MCTS_H */
