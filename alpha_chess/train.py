from __future__ import annotations

"""GPU-efficient (but CPU-correct) self-play training loop for AlphaChess.

Each iteration:
  1. Generates ``games_per_iter`` self-play games with :class:`BatchedSelfPlay`
     (one batched forward pass per simulation step across many concurrent
     games) and appends the examples to a numpy ring :class:`ReplayBuffer`.
  2. Optimizes the network for ``epochs`` passes against the MCTS visit
     distributions (policy cross-entropy) and game outcomes (value MSE).

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
import random
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from alpha_chess.batched_selfplay import BatchedSelfPlay
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
    num_parallel_games: int = 64,
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
        num_parallel_games: Concurrent games per self-play wave (batch width).
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
        generator = BatchedSelfPlay(
            model,
            device,
            num_parallel_games=num_parallel_games,
            num_simulations=simulations,
            c_puct=1.5,
            temperature_moves=temperature_moves,
            max_moves=max_moves,
            seed=selfplay_seed,
        )
        examples = generator.generate(games_per_iter)
        sp_time = max(time.time() - sp_start, 1e-9)

        buffer.append(examples)
        num_positions = len(examples)
        games_per_sec = games_per_iter / sp_time
        positions_per_sec = num_positions / sp_time

        print(
            "[iter {ii}/{it}] self-play: {g} games, {p} positions in {t:.1f}s "
            "| {gps:.2f} games/s, {pps:.1f} pos/s | lr={lr:.2e} buffer={bs}".format(
                ii=i + 1,
                it=iterations,
                g=games_per_iter,
                p=num_positions,
                t=sp_time,
                gps=games_per_sec,
                pps=positions_per_sec,
                lr=cur_lr,
                bs=len(buffer),
            )
        )

        # ---- Optimization phase ----------------------------------------
        model.train()
        for epoch in range(epochs):
            steps = max(1, len(buffer) // batch_size)
            policy_loss_sum = 0.0
            value_loss_sum = 0.0
            train_start = time.time()

            for _ in range(steps):
                states_np, policies_np, values_np = buffer.sample(batch_size)

                states = _to_device(
                    states_np, device, use_cuda, memory_format=memory_format
                )
                target_policy = _to_device(policies_np, device, use_cuda)
                target_value = _to_device(values_np, device, use_cuda)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type=amp_device_type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    policy_logits, value_pred = model(states)
                    log_probs = F.log_softmax(policy_logits, dim=1)
                    policy_loss = -(target_policy * log_probs).sum(dim=1).mean()
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

                policy_loss_sum += float(policy_loss.item())
                value_loss_sum += float(value_loss.item())

            train_time = max(time.time() - train_start, 1e-9)
            mean_policy = policy_loss_sum / steps
            mean_value = value_loss_sum / steps
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


def _to_device(array: np.ndarray, device, use_cuda: bool, memory_format=None):
    """Move a numpy array to ``device`` using pinned/non-blocking on CUDA."""
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if use_cuda:
        try:
            tensor = tensor.pin_memory()
        except RuntimeError:  # pragma: no cover - already pinned / unsupported
            pass
        tensor = tensor.to(device, non_blocking=True)
        if memory_format is not None and tensor.dim() == 4:
            tensor = tensor.to(memory_format=memory_format)
    else:
        tensor = tensor.to(device)
    return tensor


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
