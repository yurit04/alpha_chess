from __future__ import annotations

"""AlphaZero-style residual network for chess.

Exposes :class:`AlphaZeroNet` (a policy/value network operating on the
side-to-move-relative board encoding) together with device selection and
checkpoint serialization helpers.
"""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from alpha_chess.encoding import NUM_PLANES, POLICY_SIZE


class _ResidualBlock(nn.Module):
    """A single pre-activation-free residual block (conv-BN-ReLU-conv-BN + skip)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + x
        return F.relu(out)


class AlphaZeroNet(nn.Module):
    """Residual policy/value network.

    ``forward`` takes a batch of encoded boards ``(B, num_planes, 8, 8)`` and
    returns ``(policy_logits, value)`` where ``policy_logits`` has shape
    ``(B, policy_size)`` (raw, unnormalized logits) and ``value`` has shape
    ``(B, 1)`` in ``[-1, 1]``.
    """

    def __init__(
        self,
        num_planes: int = NUM_PLANES,
        channels: int = 128,
        num_blocks: int = 10,
        policy_size: int = POLICY_SIZE,
    ) -> None:
        super().__init__()
        # Preserve the construction arguments so checkpoints can rebuild the net.
        self.config = dict(
            num_planes=num_planes,
            channels=channels,
            num_blocks=num_blocks,
            policy_size=policy_size,
        )

        # Convolutional stem: num_planes -> channels.
        self.stem_conv = nn.Conv2d(num_planes, channels, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)

        # Residual tower.
        self.res_blocks = nn.ModuleList(_ResidualBlock(channels) for _ in range(num_blocks))

        # Policy head: 1x1 conv -> BN -> ReLU -> flatten -> linear to policy_size.
        self.policy_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, policy_size)

        # Value head: 1x1 conv -> BN -> ReLU -> flatten -> linear(64) -> ReLU -> linear(1) -> tanh.
        self.value_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        """Run a forward pass; returns ``(policy_logits, value)``."""
        out = F.relu(self.stem_bn(self.stem_conv(x)))
        for block in self.res_blocks:
            out = block(out)

        # Policy head.
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.flatten(1)
        policy_logits = self.policy_fc(p)

        # Value head.
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value


def get_device(prefer: str = "auto") -> torch.device:
    """Select a torch device.

    ``prefer="auto"`` picks CUDA if available, else MPS if available, else CPU.
    Any other value is passed straight through to :class:`torch.device`.
    """
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(prefer)


def save_model(model: AlphaZeroNet, path: str) -> None:
    """Serialize ``model`` (config + weights) to ``path``, creating parent dirs."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save({"config": model.config, "state_dict": model.state_dict()}, path)


def load_model(path: str, device: Optional[torch.device] = None) -> AlphaZeroNet:
    """Load an :class:`AlphaZeroNet` checkpoint saved by :func:`save_model`.

    Rebuilds the network from the stored config, loads the weights, moves it to
    ``device`` (auto-selected when ``None``) and puts it in eval mode.
    """
    if device is None:
        device = get_device()
    checkpoint = torch.load(path, map_location=device)
    config = dict(checkpoint["config"])
    stored_planes = config.get("num_planes", NUM_PLANES)
    if stored_planes != NUM_PLANES:
        raise ValueError(
            "{p} was trained on a {s}-plane encoding but this build uses {n} "
            "planes, so its weights cannot be reused. Train a fresh model, or "
            "check out the revision that produced the checkpoint.".format(
                p=path, s=stored_planes, n=NUM_PLANES
            )
        )
    model = AlphaZeroNet(**config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model
