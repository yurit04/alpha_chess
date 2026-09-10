/* CPython bindings for the native self-play engine.
 *
 * Array arguments come in through the buffer protocol (so no NumPy headers are
 * needed to build) and the search itself runs with the GIL released, which is
 * what lets several engines overlap with each other and with the GPU.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "mcts.h"

/* ------------------------------------------------------------------ */
/* Buffer helpers                                                       */
/* ------------------------------------------------------------------ */
typedef struct { Py_buffer view; int ok; } Buf;

static int buf_get(Buf *b, PyObject *obj, const char *name, char fmt,
                   Py_ssize_t itemsize, int writable)
{
    b->ok = 0;
    int flags = writable ? (PyBUF_WRITABLE | PyBUF_FORMAT | PyBUF_ND)
                         : (PyBUF_FORMAT | PyBUF_ND);
    if (PyObject_GetBuffer(obj, &b->view, flags) < 0) {
        PyErr_Format(PyExc_TypeError, "%s must be a %s contiguous buffer",
                     name, writable ? "writable" : "readable");
        return 0;
    }
    if (!PyBuffer_IsContiguous(&b->view, 'C')) {
        PyBuffer_Release(&b->view);
        PyErr_Format(PyExc_ValueError, "%s must be C-contiguous", name);
        return 0;
    }
    if (b->view.itemsize != itemsize
        || !b->view.format || b->view.format[0] != fmt) {
        Py_ssize_t got = b->view.itemsize;
        const char *gf = b->view.format ? b->view.format : "?";
        PyBuffer_Release(&b->view);
        PyErr_Format(PyExc_ValueError,
                     "%s has dtype '%s' (itemsize %zd); expected '%c' "
                     "(itemsize %zd)", name, gf, got, fmt, itemsize);
        return 0;
    }
    b->ok = 1;
    return 1;
}

static void buf_release(Buf *b) { if (b->ok) { PyBuffer_Release(&b->view); b->ok = 0; } }

/* ------------------------------------------------------------------ */
/* Engine object                                                        */
/* ------------------------------------------------------------------ */
typedef struct {
    PyObject_HEAD
    Engine e;
} EngineObject;

static void engine_dealloc_arrays(Engine *e)
{
    if (e->games) {
        for (int i = 0; i < e->n_games; i++) game_free(&e->games[i]);
        free(e->games);
    }
    free(e->pend_game); free(e->pend_node); free(e->pend_nmoves);
    free(e->pend_path_off); free(e->pend_path_len); free(e->pend_move_off);
    free(e->path_node); free(e->path_slot); free(e->path_key);
    free(e->pend_moves); free(e->pend_idx);
    free(e->out_states); free(e->out_pidx); free(e->out_pval);
    free(e->out_plen); free(e->out_values);
    free(e->noise_buf);
    memset(e, 0, sizeof(*e));
}

static void Engine_dealloc(EngineObject *self)
{
    engine_dealloc_arrays(&self->e);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int Engine_init(EngineObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {
        "games_in_flight", "num_games", "num_simulations", "fast_simulations",
        "full_search_prob", "c_puct", "dirichlet_alpha", "dirichlet_epsilon",
        "fpu_reduction", "temperature_moves", "max_moves", "resign_threshold",
        "resign_plies", "resign_disable_fraction", "seed", NULL
    };
    int games_in_flight = 64;
    long num_games = 64;
    int num_simulations = 200, fast_simulations = 50;
    double full_search_prob = 1.0, c_puct = 1.5;
    double dirichlet_alpha = 0.3, dirichlet_epsilon = 0.25, fpu_reduction = 0.0;
    int temperature_moves = 30, max_moves = 400;
    PyObject *resign_obj = Py_None;
    int resign_plies = 2;
    double resign_disable_fraction = 0.10;
    unsigned long long seed = 0;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "|iliidddddiiOidK", kwlist,
            &games_in_flight, &num_games, &num_simulations, &fast_simulations,
            &full_search_prob, &c_puct, &dirichlet_alpha, &dirichlet_epsilon,
            &fpu_reduction, &temperature_moves, &max_moves, &resign_obj,
            &resign_plies, &resign_disable_fraction, &seed))
        return -1;

    if (games_in_flight < 1) games_in_flight = 1;
    if (num_games < 1) num_games = 1;
    if ((long)games_in_flight > num_games) games_in_flight = (int)num_games;

    pos_init();
    Engine *e = &self->e;
    engine_dealloc_arrays(e);

    e->n_games = games_in_flight;
    e->num_simulations = num_simulations < 1 ? 1 : num_simulations;
    e->fast_simulations = fast_simulations < 1 ? 1 : fast_simulations;
    e->full_search_prob = full_search_prob;
    e->c_puct = c_puct;
    e->dirichlet_alpha = dirichlet_alpha;
    e->dirichlet_epsilon = dirichlet_epsilon;
    e->fpu_reduction = fpu_reduction;
    e->temperature_moves = temperature_moves;
    e->max_moves = max_moves;
    if (resign_obj == Py_None) {
        e->use_resign = 0;
        e->resign_threshold = -1.0;
    } else {
        e->use_resign = 1;
        e->resign_threshold = PyFloat_AsDouble(resign_obj);
        if (PyErr_Occurred()) return -1;
    }
    e->resign_plies = resign_plies < 1 ? 1 : resign_plies;
    e->resign_disable_fraction = resign_disable_fraction;
    e->games_target = num_games;
    rng_seed(&e->rng, seed);

    e->games = (Game *)calloc((size_t)e->n_games, sizeof(Game));
    e->pend_game = (int32_t *)calloc((size_t)e->n_games, sizeof(int32_t));
    e->pend_node = (int32_t *)calloc((size_t)e->n_games, sizeof(int32_t));
    e->pend_nmoves = (int32_t *)calloc((size_t)e->n_games, sizeof(int32_t));
    e->pend_path_off = (int32_t *)calloc((size_t)e->n_games, sizeof(int32_t));
    e->pend_path_len = (int32_t *)calloc((size_t)e->n_games, sizeof(int32_t));
    e->pend_move_off = (int32_t *)calloc((size_t)e->n_games, sizeof(int32_t));
    if (!e->games || !e->pend_game || !e->pend_node || !e->pend_nmoves
        || !e->pend_path_off || !e->pend_path_len || !e->pend_move_off) {
        PyErr_NoMemory();
        return -1;
    }
    for (int i = 0; i < e->n_games; i++) {
        arena_init(&e->games[i].arena);
        arena_init(&e->games[i].spare);
        new_game(e, &e->games[i]);
        e->games_started++;
    }
    return 0;
}

static PyObject *Engine_collect(EngineObject *self, PyObject *args)
{
    PyObject *o_states, *o_idx, *o_counts;
    if (!PyArg_ParseTuple(args, "OOO", &o_states, &o_idx, &o_counts))
        return NULL;

    Buf bs = {0}, bi = {0}, bc = {0};
    if (!buf_get(&bs, o_states, "states", 'f', 4, 1)) return NULL;
    if (!buf_get(&bi, o_idx, "idx", 'i', 4, 1)) { buf_release(&bs); return NULL; }
    if (!buf_get(&bc, o_counts, "counts", 'i', 4, 1)) {
        buf_release(&bs); buf_release(&bi); return NULL;
    }

    Engine *e = &self->e;
    PyObject *err = NULL;
    if (bi.view.ndim != 2) {
        err = PyExc_ValueError;
    } else if (bi.view.shape[0] < e->n_games
               || bc.view.shape[0] < e->n_games
               || bs.view.len < (Py_ssize_t)e->n_games * ENC_SIZE * 4) {
        err = PyExc_ValueError;
    }
    if (err) {
        buf_release(&bs); buf_release(&bi); buf_release(&bc);
        PyErr_SetString(err, "collect() buffers are too small for this engine");
        return NULL;
    }

    int idx_stride = (int)bi.view.shape[1];
    int n;
    Py_BEGIN_ALLOW_THREADS
    n = engine_collect(e, (float *)bs.view.buf, (int32_t *)bi.view.buf,
                       (int32_t *)bc.view.buf, idx_stride);
    Py_END_ALLOW_THREADS

    buf_release(&bs); buf_release(&bi); buf_release(&bc);
    if (n < 0 || e->failed) {
        PyErr_SetString(PyExc_MemoryError, "native self-play engine ran out of memory");
        return NULL;
    }
    return PyLong_FromLong(n);
}

static PyObject *Engine_apply(EngineObject *self, PyObject *args)
{
    PyObject *o_priors, *o_values;
    if (!PyArg_ParseTuple(args, "OO", &o_priors, &o_values)) return NULL;

    Engine *e = &self->e;
    if (e->n_pend == 0) {
        Py_BEGIN_ALLOW_THREADS
        engine_apply(e, NULL, NULL, 0);
        Py_END_ALLOW_THREADS
        if (e->failed) {
            PyErr_SetString(PyExc_MemoryError, "native self-play engine ran out of memory");
            return NULL;
        }
        Py_RETURN_NONE;
    }

    Buf bp = {0}, bv = {0};
    if (!buf_get(&bp, o_priors, "priors", 'f', 4, 0)) return NULL;
    if (!buf_get(&bv, o_values, "values", 'f', 4, 0)) { buf_release(&bp); return NULL; }
    if (bp.view.ndim != 2 || bp.view.shape[0] < e->n_pend
        || bv.view.shape[0] < e->n_pend) {
        buf_release(&bp); buf_release(&bv);
        PyErr_SetString(PyExc_ValueError,
                        "apply() needs one prior row and one value per pending leaf");
        return NULL;
    }
    int stride = (int)bp.view.shape[1];

    Py_BEGIN_ALLOW_THREADS
    engine_apply(e, (const float *)bp.view.buf, (const float *)bv.view.buf, stride);
    Py_END_ALLOW_THREADS

    buf_release(&bp); buf_release(&bv);
    if (e->failed) {
        PyErr_SetString(PyExc_MemoryError, "native self-play engine ran out of memory");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *Engine_done(EngineObject *self, PyObject *Py_UNUSED(ignored))
{
    return PyBool_FromLong(engine_done(&self->e));
}

static PyObject *Engine_out_count(EngineObject *self, PyObject *Py_UNUSED(ignored))
{
    return PyLong_FromLong(self->e.out_n);
}

/* Copy the finished games' examples into caller-provided arrays and reset the
 * pending set; the caller sizes them from out_count(). */
static PyObject *Engine_drain_into(EngineObject *self, PyObject *args)
{
    PyObject *o_s, *o_pi, *o_pv, *o_pl, *o_v;
    if (!PyArg_ParseTuple(args, "OOOOO", &o_s, &o_pi, &o_pv, &o_pl, &o_v))
        return NULL;

    Engine *e = &self->e;
    Buf bs = {0}, bi = {0}, bv = {0}, bl = {0}, bo = {0};
    if (!buf_get(&bs, o_s, "states", 'B', 1, 1)) return NULL;
    if (!buf_get(&bi, o_pi, "pol_idx", 'H', 2, 1)) goto fail;
    if (!buf_get(&bv, o_pv, "pol_val", 'f', 4, 1)) goto fail;
    if (!buf_get(&bl, o_pl, "pol_len", 'h', 2, 1)) goto fail;
    if (!buf_get(&bo, o_v, "values", 'f', 4, 1)) goto fail;

    const Py_ssize_t n = e->out_n;
    if (bs.view.len < n * ENC_SIZE
        || bi.view.len < n * MAX_POLICY_TARGETS * 2
        || bv.view.len < n * MAX_POLICY_TARGETS * 4
        || bl.view.len < n * 2 || bo.view.len < n * 4) {
        PyErr_SetString(PyExc_ValueError, "drain_into() buffers are too small");
        goto fail;
    }

    memcpy(bs.view.buf, e->out_states, (size_t)n * ENC_SIZE);
    memcpy(bi.view.buf, e->out_pidx, (size_t)n * MAX_POLICY_TARGETS * sizeof(uint16_t));
    memcpy(bv.view.buf, e->out_pval, (size_t)n * MAX_POLICY_TARGETS * sizeof(float));
    memcpy(bl.view.buf, e->out_plen, (size_t)n * sizeof(int16_t));
    memcpy(bo.view.buf, e->out_values, (size_t)n * sizeof(float));
    e->out_n = 0;

    buf_release(&bs); buf_release(&bi); buf_release(&bv);
    buf_release(&bl); buf_release(&bo);
    return PyLong_FromSsize_t(n);

fail:
    buf_release(&bs); buf_release(&bi); buf_release(&bv);
    buf_release(&bl); buf_release(&bo);
    return NULL;
}

static PyObject *Engine_stats(EngineObject *self, PyObject *Py_UNUSED(ignored))
{
    Engine *e = &self->e;
    return Py_BuildValue(
        "{s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d}",
        "games", e->st_games,
        "plies", e->st_plies,
        "evals", e->st_evals,
        "resigned", e->st_resigned,
        "resign_checked", e->st_resign_checked,
        "resign_false_pos", e->st_resign_fp,
        "full_plies", e->st_full_plies,
        "fast_plies", e->st_fast_plies);
}

static PyObject *Engine_pending(EngineObject *self, PyObject *Py_UNUSED(ignored))
{
    return PyLong_FromLong(self->e.n_pend);
}

static PyObject *Engine_width(EngineObject *self, PyObject *Py_UNUSED(ignored))
{
    return PyLong_FromLong(self->e.n_games);
}

static PyMethodDef Engine_methods[] = {
    {"collect", (PyCFunction)Engine_collect, METH_VARARGS,
     "collect(states, idx, counts) -> number of leaves needing evaluation"},
    {"apply", (PyCFunction)Engine_apply, METH_VARARGS,
     "apply(priors, values) -- expand the leaves, back up, play finished plies"},
    {"done", (PyCFunction)Engine_done, METH_NOARGS, "True once every game is played"},
    {"out_count", (PyCFunction)Engine_out_count, METH_NOARGS,
     "Number of finished-game examples waiting to be drained"},
    {"drain_into", (PyCFunction)Engine_drain_into, METH_VARARGS,
     "drain_into(states, pol_idx, pol_val, pol_len, values) -> rows copied"},
    {"stats", (PyCFunction)Engine_stats, METH_NOARGS, "Cumulative run statistics"},
    {"pending", (PyCFunction)Engine_pending, METH_NOARGS, "Leaves awaiting apply()"},
    {"width", (PyCFunction)Engine_width, METH_NOARGS,
     "Games searched concurrently, i.e. the most rows collect() can return"},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject EngineType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "alpha_chess._fastchess.Engine",
    .tp_doc = "Native batched PUCT self-play engine.",
    .tp_basicsize = sizeof(EngineObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)Engine_init,
    .tp_dealloc = (destructor)Engine_dealloc,
    .tp_methods = Engine_methods,
};

/* ------------------------------------------------------------------ */
/* Module-level helpers, used by the test-suite to pin the C core to    */
/* python-chess's behaviour.                                            */
/* ------------------------------------------------------------------ */
static uint64_t perft_rec(Pos *p, int depth)
{
    Move moves[MAX_MOVES];
    int n = pos_gen_legal(p, moves);
    if (depth <= 1) return (uint64_t)n;
    uint64_t total = 0;
    for (int i = 0; i < n; i++) {
        Pos next = *p;
        pos_make(&next, moves[i]);
        total += perft_rec(&next, depth - 1);
    }
    return total;
}

static PyObject *fc_perft(PyObject *self, PyObject *args)
{
    const char *fen;
    int depth;
    if (!PyArg_ParseTuple(args, "si", &fen, &depth)) return NULL;
    pos_init();
    Pos p;
    if (!pos_from_fen(&p, fen)) {
        PyErr_SetString(PyExc_ValueError, "could not parse FEN");
        return NULL;
    }
    if (depth < 1) return PyLong_FromLong(1);
    uint64_t n;
    Py_BEGIN_ALLOW_THREADS
    n = perft_rec(&p, depth);
    Py_END_ALLOW_THREADS
    return PyLong_FromUnsignedLongLong(n);
}

static PyObject *fc_legal_moves(PyObject *self, PyObject *args)
{
    const char *fen;
    if (!PyArg_ParseTuple(args, "s", &fen)) return NULL;
    pos_init();
    Pos p;
    if (!pos_from_fen(&p, fen)) {
        PyErr_SetString(PyExc_ValueError, "could not parse FEN");
        return NULL;
    }
    Move moves[MAX_MOVES];
    int n = pos_gen_legal(&p, moves);
    PyObject *list = PyList_New(n);
    if (!list) return NULL;
    for (int i = 0; i < n; i++) {
        char uci[8];
        pos_move_uci(moves[i], uci);
        PyObject *item = Py_BuildValue("(si)", uci, enc_move_index(&p, moves[i]));
        if (!item) { Py_DECREF(list); return NULL; }
        PyList_SET_ITEM(list, i, item);
    }
    return list;
}

static PyObject *fc_encode(PyObject *self, PyObject *args)
{
    const char *fen;
    int rep = 0;
    PyObject *out;
    if (!PyArg_ParseTuple(args, "sOi", &fen, &out, &rep)) return NULL;
    pos_init();
    Pos p;
    if (!pos_from_fen(&p, fen)) {
        PyErr_SetString(PyExc_ValueError, "could not parse FEN");
        return NULL;
    }
    Buf b = {0};
    if (!buf_get(&b, out, "out", 'f', 4, 1)) return NULL;
    if (b.view.len < (Py_ssize_t)ENC_SIZE * 4) {
        buf_release(&b);
        PyErr_SetString(PyExc_ValueError, "out buffer too small");
        return NULL;
    }
    enc_planes_f32(&p, rep, (float *)b.view.buf);
    buf_release(&b);
    Py_RETURN_NONE;
}

static PyMethodDef module_methods[] = {
    {"perft", fc_perft, METH_VARARGS, "perft(fen, depth) -> leaf node count"},
    {"legal_moves", fc_legal_moves, METH_VARARGS,
     "legal_moves(fen) -> [(uci, policy_index), ...]"},
    {"encode", fc_encode, METH_VARARGS,
     "encode(fen, out_float32_buffer, rep_count) -- fill a (21, 8, 8) plane stack"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fastchess_module = {
    PyModuleDef_HEAD_INIT,
    "alpha_chess._fastchess",
    "Native chess move generation, encoding and PUCT self-play search.",
    -1,
    module_methods,
};

PyMODINIT_FUNC PyInit__fastchess(void)
{
    PyObject *m = PyModule_Create(&fastchess_module);
    if (!m) return NULL;
    if (PyType_Ready(&EngineType) < 0) { Py_DECREF(m); return NULL; }
    Py_INCREF(&EngineType);
    if (PyModule_AddObject(m, "Engine", (PyObject *)&EngineType) < 0) {
        Py_DECREF(&EngineType);
        Py_DECREF(m);
        return NULL;
    }
    PyModule_AddIntConstant(m, "NUM_PLANES", ENC_PLANES);
    PyModule_AddIntConstant(m, "POLICY_SIZE", POLICY_SIZE);
    PyModule_AddIntConstant(m, "MAX_POLICY_TARGETS", MAX_POLICY_TARGETS);
    PyModule_AddIntConstant(m, "MAX_LEGAL", MAX_MOVES);
    PyModule_AddIntConstant(m, "ABI_VERSION", 1);
    return m;
}
