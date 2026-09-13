from __future__ import annotations

"""GPU-efficient (but CPU-correct) self-play training loop for AlphaChess.

Each iteration:
  1. Generates ``games_per_iter`` self-play games and appends them to a
     :class:`~alpha_chess.self_play.ReplayBuffer`.  Two engines can do this:
     the **native** core (:mod:`alpha_chess.native_selfplay`), which runs the
     whole tree search in C in this process, and the pure-Python fallback
     (:func:`~alpha_chess.batched_selfplay.generate_selfplay_data`), which
     spreads its GIL-bound search over worker processes behind a GPU-owning
     inference server.  The native core is ~3.8x faster and is used whenever
     it can be built.
  2. Optimizes the network for ``epochs`` passes against the MCTS visit
     distributions (policy cross-entropy) and game outcomes (value MSE).  The
     number of gradient steps is sized from how much *new* data the iteration
     produced (``sample_reuse``), so the optimizer's share of the wall clock
     does not drift as the replay buffer fills.

States cross both boundaries uint8-packed and policy targets stay sparse; they
are expanded on the GPU, which keeps the replay buffer ~11x smaller in RAM and
the host->device transfers correspondingly cheaper.

CUDA-only fast paths (autocast fp16, GradScaler, cudnn.benchmark,
channels_last, pinned host memory, non-blocking copies) are all guarded so
the exact same code runs correctly in fp32 on CPU/MPS with no errors and
without hanging on tiny parameters.

Resumability: a full ``train_state.pt`` (model config+weights, optimizer
state, iteration counter and RNG states) is written every iteration so a run
can be continued from where it left off.
"""

import math
import os
import queue
import random
import threading
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from alpha_chess import native, native_selfplay
from alpha_chess.batched_selfplay import default_worker_count, generate_selfplay_data
from alpha_chess.encoding import LOAD_SCALE, NUM_PLANES
from alpha_chess.network import AlphaZeroNet, get_device, load_model, save_model
from alpha_chess.self_play import ReplayBuffer

__all__ = ["train"]


def _set_seeds(seed: Optional[int]) -> None:
    """Seed Python, NumPy and torch RNGs when a seed is provided."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cosine_lr(base_lr: float, final_lr: float, step: int, total: int) -> float:
    """Cosine-decayed learning rate for iteration ``step`` of ``total``.

    At ``step == 0`` this returns ``base_lr`` and it decays smoothly towards
    ``final_lr`` as ``step`` approaches ``total``.
    """
    if total <= 1:
        return final_lr
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(step, total) / total))
    return final_lr + (base_lr - final_lr) * cosine


def _make_grad_scaler(enabled: bool):
    """Build a GradScaler that is a no-op when ``enabled`` is False.

    Prefers the modern ``torch.amp.GradScaler`` API and falls back to the
    legacy ``torch.cuda.amp.GradScaler`` for older torch builds.
    """
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _resolve_resume(resume: Optional[str]) -> Optional[str]:
    """Resolve a resume argument to a concrete file path (or ``None``).

    A directory is resolved to ``train_state.pt`` inside it when present.
    """
    if not resume:
        return None
    if os.path.isdir(resume):
        candidate = os.path.join(resume, "train_state.pt")
        return candidate if os.path.exists(candidate) else None
    return resume if os.path.exists(resume) else None


def _torch_load_full(path: str, device):
    """Load a checkpoint including non-tensor objects (optimizer/RNG state).

    torch>=2.6 defaults ``weights_only=True``, which rejects the NumPy RNG
    state stored in a resumable ``train_state.pt``. We pass
    ``weights_only=False`` (the file is produced by this trainer). Older torch
    builds without the kwarg fall back to a plain load.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _is_train_state(checkpoint: dict) -> bool:
    """True when ``checkpoint`` looks like a resumable training state."""
    return isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint


def _restore_rng(checkpoint: dict) -> None:
    """Best-effort restoration of Python/NumPy/torch RNG states."""
    py_state = checkpoint.get("python_rng_state")
    if py_state is not None:
        try:
            random.setstate(py_state)
        except (TypeError, ValueError):
            pass
    np_state = checkpoint.get("numpy_rng_state")
    if np_state is not None:
        try:
            np.random.set_state(np_state)
        except (TypeError, ValueError):
            pass
    torch_state = checkpoint.get("torch_rng_state")
    if torch_state is not None:
        try:
            torch.set_rng_state(_as_byte_tensor(torch_state))
        except (TypeError, ValueError, RuntimeError):
            pass
    cuda_state = checkpoint.get("cuda_rng_state")
    if cuda_state is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(
                [_as_byte_tensor(s) for s in cuda_state]
            )
        except (TypeError, ValueError, RuntimeError):
            pass


def _as_byte_tensor(state) -> torch.Tensor:
    """Coerce a stored RNG state back into a ByteTensor on CPU."""
    if isinstance(state, torch.Tensor):
        return state.to(dtype=torch.uint8, device="cpu")
    return torch.as_tensor(state, dtype=torch.uint8)


def _resolve_engine(engine: str) -> bool:
    """True when self-play should run on the native core."""
    choice = (engine or "auto").lower()
    if choice == "python":
        return False
    if choice == "native":
        return native.available()
    if choice != "auto":
        raise ValueError(
            "unknown engine {e!r}; expected auto, native or python".format(e=engine)
        )
    return native.available()


def _train_steps(train_steps, sample_reuse, new_positions, buffer_size,
                 batch_size, epochs):
    """Gradient steps per epoch for one iteration.

    Drawing ``new_positions * sample_reuse`` samples per iteration means each
    position is sampled ``sample_reuse`` times over its life in the buffer: it
    survives ``buffer_size / new_positions`` iterations, and over that span the
    draws total ``buffer_size * sample_reuse`` across ``buffer_size`` positions.

    The point is that the work then scales with the *data*.  The old rule
    (``buffer_size // batch_size`` steps per epoch) scaled with the buffer
    instead: 7,812 steps an iteration at a 2M buffer and batch 1024, whether
    that iteration produced 50k new positions or 800k.
    """
    epochs = max(1, int(epochs))
    if train_steps and train_steps > 0:
        return max(1, int(train_steps) // epochs)
    if sample_reuse and sample_reuse > 0:
        total = new_positions * float(sample_reuse) / float(batch_size)
        return max(1, int(round(total / epochs)))
    return max(1, buffer_size // batch_size)


def train(
    iterations: int,
    games_per_iter: int,
    simulations: int,
    epochs: int,
    batch_size: int,
    lr: float,
    out_dir: str,
    channels: int,
    num_blocks: int,
    buffer_size: int,
    temperature_moves: int,
    max_moves: int,
    weight_decay: float = 1e-4,
    games_in_flight: int = 128,
    num_workers: Optional[int] = None,
    pipeline_stages: Optional[int] = None,
    engine: str = "auto",
    pools: Optional[int] = None,
    fast_simulations: int = 50,
    full_search_prob: float = 0.25,
    fpu_reduction: float = 0.0,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    noise_all_plies: bool = False,
    sample_reuse: float = 4.0,
    train_steps: int = 0,
    resign_threshold: Optional[float] = -0.90,
    resign_disable_fraction: float = 0.10,
    save_buffer: bool = True,
    lr_final: Optional[float] = None,
    grad_clip: float = 1.0,
    use_amp: bool = True,
    compile_model: bool = False,
    eval_every: int = 0,
    eval_games: int = 0,
    checkpoint_every: int = 1,
    resume: Optional[str] = None,
    device=None,
    seed: Optional[int] = None,
) -> str:
    """Run the batched self-play training loop; return the path to ``best.pt``.

    Args:
        iterations: Number of self-play + training iterations.
        games_per_iter: Self-play games generated per iteration.
        simulations: MCTS simulations per move during self-play.
        epochs: Optimization epochs per iteration.
        batch_size: Minibatch size for optimization.
        lr: Initial (peak) Adam learning rate.
        out_dir: Directory where checkpoints and ``train_state.pt`` are written.
        channels: Residual-tower width for a fresh network.
        num_blocks: Number of residual blocks for a fresh network.
        buffer_size: Replay-buffer capacity (in examples).
        temperature_moves: Plies of temperature-1 sampling during self-play.
        max_moves: Maximum plies before a game is cut off as a draw.
        weight_decay: L2 weight decay for Adam.
        games_in_flight: Concurrent games searched per native pool (or per
            Python worker process). This is the network's batch size.
        num_workers: Self-play search processes for the Python engine
            (default: CPU count - 2). Unused by the native engine.
        pipeline_stages: Sub-pools per worker, so several requests per worker
            are in flight at once (default 4). Python engine only.
        engine: ``auto`` (native when it builds, else Python), ``native`` or
            ``python``.
        pools: Independent native game pools. One pool's descent overlaps the
            next pool's forward pass, so 2 is the minimum useful value
            (default 3).
        fast_simulations: Simulation budget for non-recorded plies under
            playout-cap randomisation (native engine only).
        full_search_prob: Probability that a ply gets the full ``simulations``
            budget, root noise and a recorded training target. ``1.0`` searches
            every ply fully, which is the classic AlphaZero behaviour.
        fpu_reduction: First-play-urgency penalty for unvisited children
            (native engine only); ``0.0`` reproduces the Python engine.
        sample_reuse: Expected number of times each position is sampled over
            its lifetime in the replay buffer. Sets the gradient-step count;
            ``0`` restores the old "one pass over the whole replay buffer per
            epoch" rule.
        train_steps: Explicit gradient steps per iteration (0 = derive them
            from ``sample_reuse``).
        resign_threshold: Resign once the mover's best root value stays at or
            below this; ``None`` plays every game to the end.
        resign_disable_fraction: Fraction of games played out with resignation
            suppressed, to measure the resign false-positive rate.
        dirichlet_alpha: Shape of the root Dirichlet noise.
        dirichlet_epsilon: Weight of that noise in the root priors. Exploration
            has to compete with the policy's own confidence, so a policy that
            has sharpened needs more of it, not the same amount.
        noise_all_plies: Apply root noise on every ply rather than only on the
            plies that also record a training target. Noise is free; only
            recording costs anything, so bundling them leaves most of the moves
            actually played with no exploration at all.
        save_buffer: Persist the replay buffer alongside ``train_state.pt`` so a
            resumed run keeps its training history.
        lr_final: Final learning rate for cosine decay (defaults to ``lr*0.1``).
        grad_clip: Max gradient norm (``<= 0`` disables clipping).
        use_amp: Enable fp16 autocast + GradScaler (CUDA only).
        compile_model: Wrap the model with ``torch.compile`` (CUDA + torch>=2).
        eval_every: Run periodic evaluation every N iterations (0 disables).
        eval_games: Games per periodic evaluation match (0 disables).
        checkpoint_every: Save an iteration checkpoint every N iterations.
        resume: Path/dir to resume a ``train_state.pt`` or load plain weights.
        device: Torch device (defaults to :func:`get_device`).
        seed: Optional RNG seed for reproducibility.

    Returns:
        The filesystem path to ``out_dir/best.pt``.
    """
    _set_seeds(seed)
    os.makedirs(out_dir, exist_ok=True)

    if device is None:
        device = get_device()
    device = torch.device(device) if not isinstance(device, torch.device) else device
    use_cuda = device.type == "cuda"
    print("Training on device: {dev}".format(dev=device))

    if use_cuda:
        torch.backends.cudnn.benchmark = True

    # AMP configuration (fp16 only on CUDA; a safe no-op everywhere else).
    amp_enabled = bool(use_amp) and use_cuda
    amp_device_type = "cuda" if use_cuda else "cpu"
    memory_format = torch.channels_last if use_cuda else torch.contiguous_format

    # Pick the self-play engine. The native core's search saturates a 3090 from
    # a fraction of one core, so it needs neither worker processes nor shared
    # memory; the Python engine spreads its GIL-bound search over processes.
    use_native = _resolve_engine(engine)
    workers = default_worker_count() if num_workers is None else int(num_workers)
    if not use_cuda:
        workers = 1
    native_pools = native_selfplay.default_pools() if pools is None else int(pools)
    # A "fast" ply that costs more than a full one would be nonsense.
    fast_simulations = max(1, min(int(fast_simulations), int(simulations)))
    if use_native:
        print("Self-play: native core, {p} pool{s} x {g} games in flight "
              "= up to {b:,} positions per forward pass".format(
                  p=native_pools, s="" if native_pools == 1 else "s",
                  g=games_in_flight, b=native_pools * games_in_flight))
        if full_search_prob < 1.0:
            print("           playout-cap randomisation: {f:.0%} of plies get "
                  "{s} simulations and a training target, the rest {q}".format(
                      f=full_search_prob, s=simulations, q=fast_simulations))
    else:
        print("Self-play: Python engine, {w} search worker(s) x {g} games in "
              "flight = up to {b:,} positions per forward pass".format(
                  w=workers, g=games_in_flight, b=workers * games_in_flight))
        if engine == "native":
            print("           (native core requested but unavailable: {e})"
                  .format(e=native.last_error()))
        if full_search_prob < 1.0:
            print("           note: playout-cap randomisation and "
                  "--fpu-reduction are native-engine features and are not "
                  "applied here; every ply gets {s} simulations.".format(
                      s=simulations))

    # Replay states are stored uint8-packed; this rescales them back on device.
    load_scale = torch.from_numpy(LOAD_SCALE).to(device).view(1, NUM_PLANES, 1, 1)

    if lr_final is None:
        lr_final = lr * 0.1

    # ------------------------------------------------------------------ #
    # Model / optimizer construction (fresh or resumed).                  #
    # ------------------------------------------------------------------ #
    resume_path = _resolve_resume(resume)
    resumed_state: Optional[dict] = None
    start_iter = 0

    if resume_path is not None:
        # weights_only=False: a resumable train_state.pt carries optimizer state
        # and Python/NumPy RNG states (not plain tensors), which the torch>=2.6
        # weights-only unpickler rejects. This file is produced by us.
        checkpoint = _torch_load_full(resume_path, device)
        if _is_train_state(checkpoint):
            # Full resumable training state: rebuild everything.
            config = checkpoint.get("config") or dict(
                channels=channels, num_blocks=num_blocks
            )
            model = AlphaZeroNet(**config)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            resumed_state = checkpoint
            start_iter = int(checkpoint.get("iteration", -1)) + 1
            print(
                "Resuming training state from {p} at iteration {i}".format(
                    p=resume_path, i=start_iter
                )
            )
        else:
            # Plain model checkpoint: load weights only, fresh optimizer.
            model = load_model(resume_path, device=device)
            print(
                "Loaded weights from {p} (fresh optimizer)".format(p=resume_path)
            )
    else:
        if resume:
            print(
                "Resume target {r} not found; starting fresh.".format(r=resume)
            )
        model = AlphaZeroNet(channels=channels, num_blocks=num_blocks)
        model.to(device)

    if use_cuda:
        model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    if resumed_state is not None:
        try:
            optimizer.load_state_dict(resumed_state["optimizer_state_dict"])
        except (ValueError, KeyError, RuntimeError) as exc:  # pragma: no cover
            print("Could not restore optimizer state ({e}); continuing.".format(e=exc))
        _restore_rng(resumed_state)

    # Optional graph compilation (guarded; default off).
    if compile_model and use_cuda and hasattr(torch, "compile"):
        try:
            major = int(str(torch.__version__).split(".", 1)[0])
        except (ValueError, IndexError):
            major = 0
        if major >= 2:
            try:
                model = torch.compile(model)
                print("Compiled model with torch.compile.")
            except Exception as exc:  # pragma: no cover - backend dependent
                print("torch.compile failed ({e}); continuing uncompiled.".format(e=exc))

    scaler = _make_grad_scaler(amp_enabled)
    buffer = ReplayBuffer(capacity=buffer_size)
    best_path = os.path.join(out_dir, "best.pt")
    train_state_path = os.path.join(out_dir, "train_state.pt")
    buffer_path = _buffer_path(out_dir)

    # A resumed run that starts from an empty buffer would spend its first
    # iterations training on one iteration's worth of games, so the buffer is
    # restored alongside the model.
    if resumed_state is not None and _load_replay_buffer(buffer_path, buffer):
        print("Restored replay buffer with {n:,} positions".format(n=len(buffer)))

    if start_iter >= iterations:
        print(
            "Nothing to do: start iteration {s} >= iterations {i}.".format(
                s=start_iter, i=iterations
            )
        )
        # Ensure best.pt exists so callers get a usable path back.
        if not os.path.exists(best_path):
            save_model(model, best_path)
        return best_path

    for i in range(start_iter, iterations):
        # ---- Learning-rate schedule (cosine decay) ---------------------
        cur_lr = _cosine_lr(lr, lr_final, i, iterations)
        for group in optimizer.param_groups:
            group["lr"] = cur_lr

        # ---- Self-play phase (batched) ---------------------------------
        model.eval()
        selfplay_seed = None if seed is None else seed + 1 + i
        sp_start = time.time()
        common = dict(
            num_games=games_per_iter,
            games_in_flight=games_in_flight,
            use_amp=amp_enabled,
            num_simulations=simulations,
            c_puct=1.5,
            temperature_moves=temperature_moves,
            max_moves=max_moves,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            resign_threshold=resign_threshold,
            resign_disable_fraction=resign_disable_fraction,
            seed=selfplay_seed,
            # A self-play phase runs for minutes; print throughput as it goes
            # rather than leaving the run silent until it finishes.
            verbose=True,
        )
        if use_native:
            batch = native_selfplay.generate_selfplay_data_native(
                model, device, pools=native_pools,
                fast_simulations=fast_simulations,
                full_search_prob=full_search_prob,
                fpu_reduction=fpu_reduction,
                noise_all_plies=bool(noise_all_plies),
                **common
            )
        else:
            batch = generate_selfplay_data(
                model, device, num_workers=workers,
                pipeline_stages=pipeline_stages, **common
            )
        sp_time = max(time.time() - sp_start, 1e-9)

        buffer.append(batch)
        stats = batch.stats
        num_positions = len(batch)
        played = max(stats.get("games", 0.0), 1.0)
        evals = stats.get("evals", 0.0)
        nn_batches = stats.get("nn_batches", 0.0)

        print(
            "[iter {ii}/{it}] self-play: {g:.0f} games, {p} positions in {t:.1f}s "
            "| {gph:,.0f} games/h, {eps:,.0f} evals/s | mean game {ml:.0f} plies "
            "| lr={lr:.2e} buffer={bs:,}".format(
                ii=i + 1, it=iterations, g=played, p=num_positions, t=sp_time,
                gph=played / sp_time * 3600.0, eps=evals / sp_time,
                ml=(stats.get("full_plies", 0.0) + stats.get("fast_plies", 0.0)
                    or stats.get("plies", 0.0)) / played,
                lr=cur_lr, bs=len(buffer),
            )
        )
        if nn_batches:
            checked = stats.get("resign_checked", 0.0)
            # The native driver reports the host's own split (time spent in the
            # search vs. blocked on the device); the Python server reports how
            # much of the phase the device was busy for.
            if use_native:
                busy = "search {s:.0f}% waiting-on-GPU {g:.0f}%".format(
                    s=stats.get("search_seconds", 0.0) / sp_time * 100.0,
                    g=stats.get("gpu_seconds", 0.0) / sp_time * 100.0,
                )
            else:
                busy = "GPU busy {g:.0f}%".format(
                    g=stats.get("gpu_seconds", 0.0) / sp_time * 100.0)
            print(
                "[iter {ii}/{it}]   inference: mean batch {mb:.0f}, {b} | "
                "resigned {r:.0f} games, resign false-positive {fp}".format(
                    ii=i + 1, it=iterations,
                    mb=stats.get("nn_positions", 0.0) / nn_batches, b=busy,
                    r=stats.get("resigned", 0.0),
                    fp=("{0:.1%} of {1:.0f} checked".format(
                        stats.get("resign_false_pos", 0.0) / checked, checked)
                        if checked else "n/a"),
                )
            )

        # ---- Optimization phase ----------------------------------------
        model.train()
        steps = _train_steps(
            train_steps, sample_reuse, num_positions, len(buffer),
            batch_size, epochs,
        )
        for epoch in range(epochs):
            # Losses accumulate on the device: reading them per step with
            # .item() would synchronise on every iteration and serialise the
            # host against the GPU it is trying to keep fed.
            loss_totals = torch.zeros(2, device=device)
            train_start = time.time()

            # Sampling and pinning run on a background thread so the ~5ms host
            # gather overlaps the ~30ms GPU step instead of preceding it.
            for packed, pol_idx, pol_val, target_value in _prefetch(
                buffer, batch_size, steps, use_cuda
            ):
                states = packed.to(device, non_blocking=True).float().mul_(
                    load_scale
                )
                if use_cuda:
                    states = states.contiguous(memory_format=memory_format)
                pol_idx = pol_idx.to(device, non_blocking=True)
                pol_val = pol_val.to(device, non_blocking=True)
                target_value = target_value.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type=amp_device_type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    policy_logits, value_pred = model(states)
                    log_probs = F.log_softmax(policy_logits, dim=1)
                    # The visit-count target is sparse, so the cross-entropy is
                    # gathered at the visited moves rather than materializing a
                    # dense (B, 4672) target. Padded slots carry weight 0.
                    picked = log_probs.gather(1, pol_idx)
                    policy_loss = -(pol_val * picked).sum(dim=1).mean()
                    value_loss = F.mse_loss(value_pred, target_value)
                    loss = policy_loss + value_loss

                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip
                    )
                scaler.step(optimizer)
                scaler.update()

                loss_totals[0] += policy_loss.detach()
                loss_totals[1] += value_loss.detach()

            totals = loss_totals.tolist()
            train_time = max(time.time() - train_start, 1e-9)
            mean_policy = totals[0] / steps
            mean_value = totals[1] / steps
            steps_per_sec = steps / train_time
            print(
                "[iter {ii}/{it}] epoch {ep}/{eps}: "
                "policy_loss={pl:.4f} value_loss={vl:.4f} total={tot:.4f} "
                "| {sps:.2f} steps/s (steps={st}, buffer={bs})".format(
                    ii=i + 1,
                    it=iterations,
                    ep=epoch + 1,
                    eps=epochs,
                    pl=mean_policy,
                    vl=mean_value,
                    tot=mean_policy + mean_value,
                    sps=steps_per_sec,
                    st=steps,
                    bs=len(buffer),
                )
            )

        # ---- Checkpointing ---------------------------------------------
        # ``torch.compile`` wraps the module; unwrap so saved state_dict keys
        # are not prefixed with ``_orig_mod.`` (which load_model can't load).
        inner_model = getattr(model, "_orig_mod", model)
        if checkpoint_every > 0 and (i % checkpoint_every == 0):
            checkpoint_path = os.path.join(
                out_dir, "checkpoint_{i:03d}.pt".format(i=i)
            )
            save_model(inner_model, checkpoint_path)
            print("[iter {ii}/{it}] saved {cp}".format(
                ii=i + 1, it=iterations, cp=checkpoint_path))

        # Always mirror the latest weights to best.pt.
        save_model(inner_model, best_path)

        # Resumable training state (model + optimizer + iteration + RNG).
        _save_train_state(
            train_state_path, model, optimizer, i, cur_lr
        )
        if save_buffer:
            _save_replay_buffer(buffer_path, buffer)
        print("[iter {ii}/{it}] saved {bp} and {ts}".format(
            ii=i + 1, it=iterations, bp=best_path, ts=train_state_path))

        # ---- Optional periodic evaluation (never crashes training) -----
        if eval_every > 0 and eval_games > 0 and ((i + 1) % eval_every == 0):
            _run_eval_hook(
                best_path=best_path,
                eval_games=eval_games,
                simulations=simulations,
                device=device,
                max_moves=max_moves,
                seed=seed,
                iter_index=i,
                iterations=iterations,
            )

    return best_path


def _prefetch(buffer, batch_size: int, steps: int, use_cuda: bool, depth: int = 3):
    """Yield ``steps`` pinned minibatches, sampled on a background thread.

    States stay uint8-packed and policies stay sparse across the transfer; the
    trainer expands both on the GPU. That is ~11x fewer bytes over PCIe than
    sending dense float32 states and dense policy targets.
    """
    out: "queue.Queue" = queue.Queue(maxsize=depth)
    sentinel = object()

    def produce():
        try:
            for _ in range(steps):
                states, pol_idx, pol_val, values = buffer.sample(batch_size)
                tensors = (
                    torch.from_numpy(states),
                    torch.from_numpy(pol_idx),
                    torch.from_numpy(pol_val),
                    torch.from_numpy(values),
                )
                if use_cuda:
                    try:
                        tensors = tuple(t.pin_memory() for t in tensors)
                    except RuntimeError:  # pragma: no cover - unsupported host
                        pass
                out.put(tensors)
        except Exception as exc:  # pragma: no cover - surfaced to the consumer
            out.put(exc)
            return
        out.put(sentinel)

    thread = threading.Thread(target=produce, daemon=True)
    thread.start()
    while True:
        item = out.get()
        if item is sentinel:
            return
        if isinstance(item, Exception):
            raise item
        yield item


def _buffer_path(out_dir: str) -> str:
    return os.path.join(out_dir, "replay_buffer.npz")


def _save_replay_buffer(path: str, buffer) -> None:
    """Persist the replay buffer so a resumed run keeps its training history."""
    tmp = path + ".tmp.npz"
    np.savez(tmp, **buffer.state_dict())
    os.replace(tmp, path)


def _load_replay_buffer(path: str, buffer) -> bool:
    """Restore a persisted replay buffer; return whether anything was loaded."""
    if not os.path.exists(path):
        return False
    try:
        with np.load(path) as data:
            buffer.load_state_dict({k: data[k] for k in data.files})
    except (OSError, ValueError, KeyError) as exc:
        print("Could not restore replay buffer ({e}); starting empty.".format(e=exc))
        return False
    return len(buffer) > 0


def _save_train_state(path: str, model, optimizer, iteration: int, cur_lr: float) -> None:
    """Write a resumable training-state checkpoint atomically-ish."""
    # ``torch.compile`` wraps the module; unwrap to get a clean state_dict.
    inner = getattr(model, "_orig_mod", model)
    config = getattr(inner, "config", None)

    state = {
        "config": config,
        "model_state_dict": inner.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
        "lr": cur_lr,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        except RuntimeError:  # pragma: no cover
            pass

    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def _run_eval_hook(
    best_path: str,
    eval_games: int,
    simulations: int,
    device,
    max_moves: int,
    seed: Optional[int],
    iter_index: int,
    iterations: int,
) -> None:
    """Play the current model against a baseline; never raise on failure."""
    try:
        from alpha_chess.evaluate import evaluate_model  # lazy import

        result = evaluate_model(
            best_path,
            "material",
            games=eval_games,
            simulations=simulations,
            device=device,
            max_moves=max_moves,
            seed=seed,
        )
        print(
            "[iter {ii}/{it}] eval vs {opp}: "
            "W/D/L={w}/{d}/{l} score={s:.3f} elo_diff={e:+.0f}".format(
                ii=iter_index + 1,
                it=iterations,
                opp=result.get("opponent", "material"),
                w=result.get("wins", "?"),
                d=result.get("draws", "?"),
                l=result.get("losses", "?"),
                s=float(result.get("score", 0.0)),
                e=float(result.get("elo_diff", 0.0)),
            )
        )
    except Exception as exc:  # pragma: no cover - eval must never crash training
        print(
            "[iter {ii}/{it}] evaluation skipped ({e})".format(
                ii=iter_index + 1, it=iterations, e=exc
            )
        )
