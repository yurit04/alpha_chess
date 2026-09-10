from __future__ import annotations

"""Self-play driven by the native search core.

With the tree search in C (see :mod:`alpha_chess.native`) the shape of the
pipeline changes completely.  The Python engine had to spread a GIL-bound
search over a dozen worker *processes* and feed them through shared memory,
because one core could only produce ~12k leaves/s.  A native pool produces
leaves faster than an RTX 3090 can evaluate them, so this driver is a single
process with no IPC at all:

* a handful of independent **pools**, each holding its own games and its own
  pinned staging buffers;
* while one pool's batch is on the GPU, the next pool's descent runs on the CPU
  (the C code releases the GIL, and CUDA work is asynchronous), so neither side
  waits for the other;
* the whole run fits in one process, so nothing is serialised or copied
  between address spaces.

The output is a :class:`~alpha_chess.batched_selfplay.SelfPlayBatch`, identical
in layout to the pure-Python engine's, so the trainer and replay buffer do not
care which produced it.
"""

import contextlib
import time
from typing import List, Optional

import numpy as np

from alpha_chess import native
from alpha_chess.batched_selfplay import (
    MAX_POLICY_TARGETS,
    SelfPlayBatch,
    _bucket_batch,
)
from alpha_chess.encoding import NUM_PLANES

# Widest legal-move list the engine can hand back. It fixes the width of the
# staging buffers, so it is checked against the extension's own value at
# construction rather than trusted.
MAX_LEGAL = 256


def available() -> bool:
    """Whether the native core can be used in this process."""
    return native.available()


class _GraphedEvaluator:
    """One CUDA graph per batch shape, covering the *whole* evaluation.

    Not just the forward pass: the legal-move gather, the mask, the softmax and
    the dtype casts are all captured too.  Left eager, those ~10 small kernels
    plus their tensor bookkeeping cost ~2.8ms of host time per pass -- more
    than the forward pass itself at these batch sizes -- and the host, not the
    GPU, becomes the limit.  Captured, issuing a pass costs ~0.05ms.

    Inputs are copied into the graph's static buffers exactly as the engine
    wrote them (float32 states, int32 indices, int32 counts), so every host
    transfer is a plain contiguous DMA out of pinned memory.

    Graphs are captured lazily per shape and dropped at the end of the phase:
    the weights change between phases and a graph captured under autocast could
    otherwise hold a stale fp16 copy of them.
    """

    def __init__(self, torch, model, device, use_amp, max_legal):
        self.torch = torch
        self.model = model
        self.device = device
        self.use_amp = bool(use_amp) and device.type == "cuda"
        self.channels_last = device.type == "cuda"
        self.max_legal = max_legal
        self._graphs = {}
        self._pool = None
        self._disabled = device.type != "cuda"
        self._arange = torch.arange(max_legal, device=device)

    def _forward(self, x, idx_i32, cnt_i32):
        """Network + legal-move softmax; the body that gets captured."""
        torch = self.torch
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", torch.float16,
                            enabled=self.use_amp, cache_enabled=False):
            logits, value = self.model(x)
        # Gather in the network's own dtype and widen afterwards: upcasting the
        # full (B, 4672) logit matrix first would copy megabytes per pass to
        # read a few dozen entries per row.
        gathered = logits.gather(1, idx_i32.long()).float()
        mask = self._arange.unsqueeze(0) >= cnt_i32.long().unsqueeze(1)
        gathered = gathered.masked_fill(mask, float("-inf"))
        return torch.softmax(gathered, dim=1), value.float().view(-1)

    def _capture(self, size):
        torch = self.torch
        try:
            x = torch.zeros((size, NUM_PLANES, 8, 8), dtype=torch.float32,
                            device=self.device)
            idx = torch.zeros((size, self.max_legal), dtype=torch.int32,
                              device=self.device)
            cnt = torch.ones((size,), dtype=torch.int32, device=self.device)

            # Capture of an uninitialised cuDNN/cuBLAS workspace is not legal,
            # so warm the shape up on a side stream first.
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                with torch.no_grad():
                    for _ in range(3):
                        self._forward(x, idx, cnt)
            torch.cuda.current_stream().wait_stream(side)

            graph = torch.cuda.CUDAGraph()
            with torch.no_grad():
                with torch.cuda.graph(graph, pool=self._pool):
                    priors, value = self._forward(x, idx, cnt)
            if self._pool is None:
                self._pool = graph.pool()
            entry = (x, idx, cnt, priors, value, graph)
            self._graphs[size] = entry
            return entry
        except Exception:
            # Any capture failure (driver, memory, an uncapturable op) drops
            # this evaluator back to the eager path for good.
            self._graphs[size] = None
            self._disabled = True
            return None

    def entry(self, size):
        """Static buffers + graph for ``size``, or ``None`` to run eagerly."""
        if self._disabled:
            return None
        entry = self._graphs.get(size)
        if entry is None and size not in self._graphs:
            entry = self._capture(size)
        return entry

    def eager(self, x, idx_i32, cnt_i32):
        with self.torch.no_grad():
            return self._forward(x, idx_i32, cnt_i32)

    def close(self):
        self._graphs.clear()
        self._pool = None
        try:
            self.torch.cuda.synchronize()
            self.torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - best effort
            pass


class _Pool:
    """One native engine plus the pinned buffers its batches travel through."""

    def __init__(self, torch, mod, device, games_in_flight, num_games, cfg, seed):
        self.engine = mod.Engine(
            games_in_flight=games_in_flight, num_games=num_games, seed=seed, **cfg
        )
        self.n_games = self.engine.width()
        pin = device.type == "cuda"

        def _alloc(shape, dtype):
            t = torch.zeros(shape, dtype=dtype)
            if pin:
                t = t.pin_memory()
            return t

        self.states = _alloc((self.n_games, NUM_PLANES, 8, 8), torch.float32)
        self.idx = _alloc((self.n_games, MAX_LEGAL), torch.int32)
        self.counts = _alloc((self.n_games,), torch.int32)
        # Results come back into two contiguous buffers, one full-width row per
        # leaf; the engine reads each row's first ``n_legal`` priors.
        self.host_out = _alloc((self.n_games, MAX_LEGAL), torch.float32)
        self.host_val = _alloc((self.n_games,), torch.float32)

        self.states_np = self.states.numpy()
        self.idx_np = self.idx.numpy()
        self.counts_np = self.counts.numpy()
        self.host_np = self.host_out.numpy()
        self.val_np = self.host_val.numpy()

        self.event = torch.cuda.Event() if device.type == "cuda" else None
        self.pending = 0

    def collect(self) -> int:
        self.pending = self.engine.collect(
            self.states_np.reshape(-1), self.idx_np, self.counts_np
        )
        return self.pending

    def apply(self) -> None:
        n = self.pending
        if n <= 0:
            self.engine.apply(None, None)
            return
        self.engine.apply(self.host_np[:n], self.val_np[:n])
        self.pending = 0


class NativeRunner:
    """Runs ``num_games`` self-play games across a set of native pools."""

    def __init__(
        self,
        model,
        device,
        num_games: int,
        pools: int,
        games_in_flight: int,
        cfg: dict,
        seed: Optional[int],
        use_amp: bool,
    ) -> None:
        import torch

        self.torch = torch
        self.device = device
        self.model = model
        self.use_amp = bool(use_amp) and device.type == "cuda"
        self.channels_last = device.type == "cuda"
        self.num_games = int(num_games)
        model.eval()

        mod = native.load()
        if mod is None:
            raise RuntimeError("native core is not available")
        if mod.MAX_LEGAL != MAX_LEGAL:
            raise RuntimeError(
                "native core reports MAX_LEGAL={g} but this module sizes its "
                "buffers for {w}".format(g=mod.MAX_LEGAL, w=MAX_LEGAL)
            )

        pools = max(1, int(pools))
        base, extra = divmod(self.num_games, pools)
        self.pools: List[_Pool] = []
        for k in range(pools):
            share = base + (1 if k < extra else 0)
            if share <= 0:
                continue
            self.pools.append(
                _Pool(
                    torch, mod, device,
                    min(games_in_flight, share), share, cfg,
                    0 if seed is None else (seed * 7919 + k * 104729) & 0xFFFFFFFF,
                )
            )

        self._stream = torch.cuda.Stream() if device.type == "cuda" else None
        self._net = _GraphedEvaluator(torch, model, device, self.use_amp, MAX_LEGAL)

        self.batches = 0
        self.positions = 0
        self.gpu_time = 0.0
        self.search_time = 0.0

    # ------------------------------------------------------------------ #
    def _launch(self, pool: _Pool) -> None:
        """Upload one pool's leaves and start its evaluation; never blocks."""
        torch = self.torch
        n = pool.pending
        # Round up to a shape the evaluator already has a graph for: cuDNN and
        # CUDA-graph capture both key off the exact input shape.
        padded = _bucket_batch(n, pool.n_games)
        # Padding rows are fed to the network only to keep the shape stable; a
        # zero legal-move count would make their softmax all-NaN, so give them
        # one arbitrary legal slot instead.
        if padded > n:
            pool.counts_np[n:padded] = 1

        entry = self._net.entry(padded)
        stream_ctx = (
            torch.cuda.stream(self._stream) if self._stream is not None
            else contextlib.nullcontext()
        )
        with stream_ctx:
            if entry is not None:
                static_x, static_idx, static_cnt, priors, value, graph = entry
                static_x.copy_(pool.states[:padded], non_blocking=True)
                static_idx.copy_(pool.idx[:padded], non_blocking=True)
                static_cnt.copy_(pool.counts[:padded], non_blocking=True)
                graph.replay()
            else:
                x = pool.states[:padded].to(self.device, non_blocking=True)
                idx_t = pool.idx[:padded].to(self.device, non_blocking=True)
                cnt_t = pool.counts[:padded].to(self.device, non_blocking=True)
                priors, value = self._net.eager(x, idx_t, cnt_t)
            pool.host_out[:n].copy_(priors[:n], non_blocking=True)
            pool.host_val[:n].copy_(value[:n], non_blocking=True)
            if pool.event is not None:
                pool.event.record(self._stream)

        self.batches += 1
        self.positions += n

    def _sync(self, pool: _Pool) -> None:
        if pool.event is not None:
            pool.event.synchronize()

    def _cpu_step(self, pool: _Pool) -> int:
        """Finish the pool's previous batch and descend for the next one."""
        pool.apply()
        if pool.engine.done():
            return 0
        return pool.collect()

    # ------------------------------------------------------------------ #
    def run(self, verbose: bool = True, report_every: float = 30.0) -> SelfPlayBatch:
        pools = [p for p in self.pools if not p.engine.done()]
        if not pools:
            return SelfPlayBatch.empty()

        started = time.time()
        next_report = started + report_every
        # Progress is reported over the interval since the last line, not since
        # the start: the first minute is warm-up (CUDA-graph capture, and every
        # game still on its opening moves with an empty tree), so a cumulative
        # average takes many minutes to stop understating the steady state.
        mark = (started, 0, 0.0, 0.0, 0.0)
        m_batches = 0
        parts: List[SelfPlayBatch] = []

        # Prime every pool so the first forward pass has work waiting behind it.
        for pool in pools:
            if pool.collect() > 0:
                self._launch(pool)

        live = list(pools)
        while live:
            still: List[_Pool] = []
            for pool in live:
                if pool.pending > 0:
                    sync_start = time.time()
                    self._sync(pool)
                    self.gpu_time += time.time() - sync_start
                cpu_start = time.time()
                n = self._cpu_step(pool)
                self.search_time += time.time() - cpu_start
                if n > 0:
                    self._launch(pool)
                    still.append(pool)
                elif not pool.engine.done():
                    still.append(pool)
                else:
                    parts.append(self._drain(pool))

            live = still
            if verbose and time.time() >= next_report:
                now = time.time()
                elapsed = now - started
                done_games = sum(p.engine.stats()["games"] for p in pools)
                m_time, m_pos, m_games, m_search, m_gpu = mark
                window = max(now - m_time, 1e-9)
                print(
                    "  [self-play {e:.0f}s] {p:,.0f} positions/s | "
                    "{g:,.0f} games/h | mean batch {m:.0f} | search {s:.0f}% "
                    "waiting-on-GPU {w:.0f}% | {n} pools live".format(
                        e=elapsed, p=(self.positions - m_pos) / window,
                        g=(done_games - m_games) / window * 3600.0,
                        m=(self.positions - m_pos) / max(self.batches - m_batches, 1),
                        s=(self.search_time - m_search) / window * 100.0,
                        w=(self.gpu_time - m_gpu) / window * 100.0,
                        n=len(live),
                    ),
                    flush=True,
                )
                mark = (now, self.positions, done_games, self.search_time,
                        self.gpu_time)
                m_batches = self.batches
                next_report = now + report_every

        for pool in pools:
            if pool.engine.out_count():
                parts.append(self._drain(pool))
        self._net.close()

        batch = SelfPlayBatch.concat(parts)
        batch.stats["nn_batches"] = float(self.batches)
        batch.stats["nn_positions"] = float(self.positions)
        batch.stats["gpu_seconds"] = float(self.gpu_time)
        batch.stats["search_seconds"] = float(self.search_time)
        return batch

    def _drain(self, pool: _Pool) -> SelfPlayBatch:
        """Copy a pool's finished games out of the engine into NumPy arrays."""
        n = pool.engine.out_count()
        stats = {k: float(v) for k, v in pool.engine.stats().items()}
        if n == 0:
            batch = SelfPlayBatch.empty()
            batch.stats = stats
            return batch
        states = np.empty((n, NUM_PLANES, 8, 8), dtype=np.uint8)
        pol_idx = np.empty((n, MAX_POLICY_TARGETS), dtype=np.uint16)
        pol_val = np.empty((n, MAX_POLICY_TARGETS), dtype=np.float32)
        pol_len = np.empty((n,), dtype=np.int16)
        values = np.empty((n,), dtype=np.float32)
        pool.engine.drain_into(
            states.reshape(-1), pol_idx.reshape(-1), pol_val.reshape(-1),
            pol_len, values,
        )
        return SelfPlayBatch(states, pol_idx, pol_val, pol_len, values, stats)


def default_pools() -> int:
    """Pools to run when none is given.

    Two is enough to hide the CPU descent behind the GPU forward pass; a third
    covers the jitter from a pool that momentarily has few active games.
    """
    return 3


def generate_selfplay_data_native(
    model,
    device,
    num_games: int,
    games_in_flight: int = 512,
    pools: Optional[int] = None,
    use_amp: bool = True,
    seed: Optional[int] = None,
    verbose: bool = True,
    **cfg,
) -> SelfPlayBatch:
    """Native-core equivalent of ``batched_selfplay.generate_selfplay_data``."""
    runner = NativeRunner(
        model, device, num_games,
        default_pools() if pools is None else pools,
        games_in_flight, cfg, seed, use_amp,
    )
    return runner.run(verbose=verbose)
