from __future__ import annotations

"""From-scratch self-play training loop for AlphaChess.

Alternates between generating self-play games (which populate a replay
buffer) and optimizing the network against the collected MCTS visit
distributions and game outcomes.  Designed to stay CPU-friendly and to
never hang: the network runs in ``eval()`` mode during self-play and in
``train()`` mode during optimization.
"""

import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from alpha_chess.network import AlphaZeroNet, get_device, load_model, save_model
from alpha_chess.self_play import ReplayBuffer, play_game

__all__ = ["train"]


def _set_seeds(seed: Optional[int]) -> None:
    """Seed Python, NumPy and torch RNGs when a seed is provided."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    resume: Optional[str] = None,
    device=None,
    seed: Optional[int] = None,
) -> str:
    """Run the self-play training loop and return the path to ``best.pt``.

    For each iteration we play ``games_per_iter`` self-play games (appending
    the resulting examples to a shared :class:`ReplayBuffer`), then optimize
    the network for ``epochs`` passes.  After every iteration the model is
    saved both as an iteration checkpoint and as ``best.pt``.

    Args:
        iterations: Number of self-play + training iterations.
        games_per_iter: Self-play games generated per iteration.
        simulations: MCTS simulations per move during self-play.
        epochs: Optimization epochs per iteration.
        batch_size: Minibatch size for optimization.
        lr: Adam learning rate.
        out_dir: Directory where checkpoints are written.
        channels: Residual-tower width for a fresh network.
        num_blocks: Number of residual blocks for a fresh network.
        buffer_size: Replay-buffer capacity (in examples).
        temperature_moves: Plies of temperature-1 sampling during self-play.
        max_moves: Maximum plies before a game is cut off as a draw.
        weight_decay: L2 weight decay for Adam.
        resume: Optional checkpoint path to resume/fine-tune from.
        device: Torch device (defaults to :func:`get_device`).
        seed: Optional RNG seed for reproducibility.

    Returns:
        The filesystem path to ``out_dir/best.pt``.
    """
    _set_seeds(seed)

    os.makedirs(out_dir, exist_ok=True)

    if device is None:
        device = get_device()
    print("Training on device: {dev}".format(dev=device))

    # Build a fresh network or resume from an existing checkpoint.
    if resume:
        model = load_model(resume, device=device)
    else:
        model = AlphaZeroNet(channels=channels, num_blocks=num_blocks)
        model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    buffer = ReplayBuffer(capacity=buffer_size)

    best_path = os.path.join(out_dir, "best.pt")

    # Running self-play outcome statistics (from White's perspective).
    total_games = 0
    white_wins = 0
    black_wins = 0
    draws = 0

    for i in range(iterations):
        # ---- Self-play phase -------------------------------------------
        model.eval()
        for g in range(games_per_iter):
            game_seed = None if seed is None else seed + i * games_per_iter + g
            examples = play_game(
                model,
                device=device,
                num_simulations=simulations,
                temperature_moves=temperature_moves,
                max_moves=max_moves,
                seed=game_seed,
            )

            # Determine the game result from White's perspective using the
            # value assigned to the first (White-to-move) example.
            game_len = len(examples)
            if game_len > 0:
                first_value = float(examples[0]["value"])
                if first_value > 0:
                    result_str = "1-0"
                    white_wins += 1
                elif first_value < 0:
                    result_str = "0-1"
                    black_wins += 1
                else:
                    result_str = "1/2-1/2"
                    draws += 1
            else:
                result_str = "1/2-1/2"
                draws += 1

            buffer.append(examples)
            total_games += 1

            print(
                "[iter {ii}/{it}] game {gg}/{gpi}: result={res} len={ln} "
                "| running W/D/B = {w}/{d}/{b} (of {tot})".format(
                    ii=i + 1,
                    it=iterations,
                    gg=g + 1,
                    gpi=games_per_iter,
                    res=result_str,
                    ln=game_len,
                    w=white_wins,
                    d=draws,
                    b=black_wins,
                    tot=total_games,
                )
            )

        # ---- Optimization phase ----------------------------------------
        model.train()
        for epoch in range(epochs):
            steps = max(1, len(buffer) // batch_size)
            policy_loss_sum = 0.0
            value_loss_sum = 0.0

            for _ in range(steps):
                states_np, policies_np, values_np = buffer.sample(batch_size)

                states = torch.as_tensor(
                    states_np, dtype=torch.float32, device=device
                )
                target_policy = torch.as_tensor(
                    policies_np, dtype=torch.float32, device=device
                )
                target_value = torch.as_tensor(
                    values_np, dtype=torch.float32, device=device
                )

                policy_logits, value_pred = model(states)

                # Cross-entropy against the MCTS visit distribution.
                log_probs = F.log_softmax(policy_logits, dim=1)
                policy_loss = -(target_policy * log_probs).sum(dim=1).mean()

                # Mean-squared error on the scalar value head.
                value_loss = F.mse_loss(value_pred, target_value)

                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                policy_loss_sum += float(policy_loss.item())
                value_loss_sum += float(value_loss.item())

            mean_policy = policy_loss_sum / steps
            mean_value = value_loss_sum / steps
            print(
                "[iter {ii}/{it}] epoch {ep}/{eps}: "
                "policy_loss={pl:.4f} value_loss={vl:.4f} "
                "total={tot:.4f} (steps={st}, buffer={bs})".format(
                    ii=i + 1,
                    it=iterations,
                    ep=epoch + 1,
                    eps=epochs,
                    pl=mean_policy,
                    vl=mean_value,
                    tot=mean_policy + mean_value,
                    st=steps,
                    bs=len(buffer),
                )
            )

        # ---- Checkpointing ---------------------------------------------
        checkpoint_path = os.path.join(
            out_dir, "checkpoint_{i:03d}.pt".format(i=i)
        )
        save_model(model, checkpoint_path)
        save_model(model, best_path)
        print(
            "[iter {ii}/{it}] saved {cp} and {bp}".format(
                ii=i + 1, it=iterations, cp=checkpoint_path, bp=best_path
            )
        )

    return best_path
