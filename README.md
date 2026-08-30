# alpha_chess

An **AlphaZero-style, self-play chess engine** built with PyTorch and
[python-chess](https://python-chess.readthedocs.io/). It learns entirely from
self-play — no human games, no opening book, no handcrafted evaluation — using a
single residual **policy + value** neural network guided by **PUCT Monte-Carlo
Tree Search (MCTS)**. It ships with a pygame GUI so you can play against the
trained agent or ask it for the best move in any position.

---

## Table of contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [1. Train from scratch](#1-train-from-scratch)
4. [2. Play in the GUI](#2-play-in-the-gui)
5. [3. Suggest the best move](#3-suggest-the-best-move)
6. [How it works](#how-it-works)
7. [Project layout](#project-layout)
8. [Testing](#testing)
9. [FAQ / troubleshooting](#faq--troubleshooting)

---

## Installation

Requires **Python 3.9+**. Everything runs from the repository root.

```bash
# from the repo root
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

If pip cannot find a suitable PyTorch build, install the CPU wheel explicitly
(on Apple Silicon this wheel still includes GPU/MPS support):

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Both `python -m alpha_chess` and `python -m alpha_chess.cli` invoke the same CLI.
The examples below use `.venv/bin/python`; if you've activated the venv
(`source .venv/bin/activate`) you can just write `python`.

---

## Quick start

```bash
# 1) Train a small model fast (a few minutes on CPU) — writes models/best.pt
.venv/bin/python -m alpha_chess.cli train \
  --iterations 5 --games-per-iter 10 --simulations 50 \
  --channels 32 --blocks 4 --out models --seed 0

# 2) Play against it in a window (H = hint, U = undo, N = new game, F = flip)
.venv/bin/python -m alpha_chess.cli play --model models/best.pt

# 3) Ask for the best move in any position (FEN)
.venv/bin/python -m alpha_chess.cli suggest \
  --fen "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3" \
  --model models/best.pt --simulations 200
```

> **Note:** A freshly/briefly trained network plays weakly and often near-random
> (the top moves show roughly uniform probabilities). Real strength needs a
> serious training run — see [Training cost & expectations](#training-cost--expectations).

---

## 1. Train from scratch

Training alternates **self-play** (generate games with the current network +
MCTS) and **learning** (train the network on those games), saving a checkpoint
after every iteration. Checkpoints are written to `--out` as
`checkpoint_{i:03d}.pt`, and the latest is always mirrored to `best.pt`.

```bash
.venv/bin/python -m alpha_chess.cli train \
  --iterations 10 \
  --games-per-iter 20 \
  --simulations 100 \
  --epochs 5 \
  --batch-size 64 \
  --lr 1e-3 \
  --channels 64 \
  --blocks 5 \
  --buffer-size 50000 \
  --temperature-moves 30 \
  --max-moves 400 \
  --weight-decay 1e-4 \
  --out models \
  --seed 0
```

### Training flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--iterations` | `10` | Number of self-play → train cycles. |
| `--games-per-iter` | `20` | Self-play games generated each iteration. |
| `--simulations` | `100` | MCTS simulations per move during self-play. Higher = stronger data, slower. |
| `--epochs` | `5` | Passes over the replay buffer per iteration during the learning phase. |
| `--batch-size` | `64` | Minibatch size for the gradient step. |
| `--lr` | `1e-3` | Adam learning rate. |
| `--channels` | `64` | Width of the residual tower (conv channels). Bigger = stronger, slower. |
| `--blocks` | `5` | Number of residual blocks (depth). Bigger = stronger, slower. |
| `--buffer-size` | `50000` | Replay buffer capacity (positions). Old positions are evicted. |
| `--temperature-moves` | `30` | For the first N plies, moves are **sampled** from MCTS visit counts (exploration); after that the **best** move is played. |
| `--max-moves` | `400` | Cap on plies per self-play game (cutoff scored as a draw). |
| `--weight-decay` | `1e-4` | L2 regularization for Adam. |
| `--resume` | `None` | Path to a checkpoint to continue/fine-tune from. |
| `--seed` | `None` | Seeds `torch`/`numpy`/`random` for reproducibility. |
| `--device` | `auto` | `auto` \| `cpu` \| `cuda` \| `mps`. See [Devices](#devices-cpu--gpu--apple-silicon). |

### Resuming training

```bash
.venv/bin/python -m alpha_chess.cli train --resume models/best.pt --iterations 10 --out models
```

### Training cost & expectations

- The wall-clock is dominated by **self-play**, which is
  `games-per-iter × (moves per game) × simulations` network evaluations. The
  quick-start config is intentionally tiny so it finishes fast; it will **not**
  produce a strong engine.
- To get meaningfully non-random play, scale up gradually — more `--iterations`,
  more `--games-per-iter`, more `--simulations`, and a larger network
  (`--channels 128 --blocks 10`). AlphaZero-level chess famously required
  enormous compute; this project is a faithful, runnable *implementation* of the
  method, not a from-scratch grandmaster on a laptop.

### Devices (CPU / GPU / Apple Silicon)

`--device auto` picks, in order:

1. **CUDA** (NVIDIA GPU) if available;
2. **MPS** (Apple Silicon GPU) if available;
3. **CPU** otherwise.

On an Apple Silicon Mac, `auto` selects **MPS**, so training uses the Mac GPU by
default. The GPU helps most during the batched learning step; self-play consists
of many tiny single-position evaluations where GPU speedups are smaller. Force a
device with `--device cpu` or `--device mps`. If you hit an unsupported-op error
on MPS, set `PYTORCH_ENABLE_MPS_FALLBACK=1` in your environment or use
`--device cpu`.

---

## 2. Play in the GUI

Launch the pygame board. If `--model` is omitted or missing, the GUI runs
**human vs human** with the agent and hints disabled.

```bash
.venv/bin/python -m alpha_chess.cli play \
  --model models/best.pt \
  --simulations 200 \
  --color white
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | `models/best.pt` | Checkpoint to load. Omit/missing → human vs human. |
| `--simulations` | `200` | MCTS simulations per agent move (higher = stronger, slower). |
| `--color` | `white` | Which side **you** play. |
| `--device` | `auto` | Inference device (see above). |

**Controls**

- **Click** a piece, then click its destination to move. Pawns auto-promote to a
  queen.
- **H** — hint: highlights the engine's suggested move and shows its evaluation
  and top candidates in the side panel.
- **U** — undo the last move (a full human+agent pair when playing the engine).
- **N** — new game.
- **F** — flip the board.
- **ESC / Q** — quit.

The side panel shows whose turn it is, check/checkmate/stalemate/draw status, a
"thinking…" indicator while the agent searches, and the latest evaluation.

---

## 3. Suggest the best move

Print the engine's recommended move, its value estimate, and the top 5
candidates for any position given as FEN (defaults to the standard start).

```bash
.venv/bin/python -m alpha_chess.cli suggest \
  --fen "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3" \
  --model models/best.pt \
  --simulations 200
```

Example output:

```
Best move: Nge7 (g8e7)
Eval (value, side-to-move perspective): -0.0081
Top moves:
  Nge7     12.50%
  ...
```

The **eval** is the network's value estimate in `[-1, +1]` from the perspective
of the side to move (`+1` = winning, `0` = balanced, `-1` = losing). The
percentages are the MCTS visit-count distribution over candidate moves.

| Flag | Default | Meaning |
|------|---------|---------|
| `--fen` | start position | Position to analyze. |
| `--model` | `models/best.pt` | Checkpoint to load. |
| `--simulations` | `200` | MCTS simulations for the search. |
| `--device` | `auto` | Inference device. |

---

## How it works

alpha_chess reimplements the **AlphaZero** algorithm: a single neural network
provides move priors and a position evaluation, and **MCTS** uses the network to
look ahead and produce a stronger policy. The network is then trained to imitate
that stronger MCTS policy and to predict the game outcome — a loop that
bootstraps from random play.

### The model (`network.py`)

A **dual-head residual convolutional network** (`AlphaZeroNet`), the AlphaZero
architecture in miniature:

- **Input:** a `(19, 8, 8)` tensor encoding the board (see *Encoding* below).
- **Body:** a 3×3 conv **stem** (→ `channels`) + BatchNorm + ReLU, followed by
  `num_blocks` **residual blocks** (each: conv3×3 → BN → ReLU → conv3×3 → BN,
  plus a skip connection, then ReLU).
- **Policy head:** 1×1 conv → BN → ReLU → flatten → linear → **4672 logits**,
  one per possible move (see *Move encoding* below). Raw logits are masked to the
  legal moves and soft-maxed at use time.
- **Value head:** 1×1 conv → BN → ReLU → flatten → linear(64) → ReLU → linear(1)
  → **tanh**, giving a scalar in `[-1, +1]` — the expected game result from the
  side-to-move's perspective.

Defaults are `channels=64`, `num_blocks=5` (both configurable from the CLI).
Checkpoints store the architecture config alongside the weights, so
`load_model` rebuilds the exact network automatically.

### Board & move encoding (`encoding.py`)

- **Board → tensor:** 19 planes of 8×8 — 6 white piece types, 6 black piece
  types, side to move, the four castling rights, the en-passant target square,
  and a halfmove-clock plane.
- **Move ↔ index:** the AlphaZero **8×8×73 = 4672** move representation. For each
  from-square, 73 planes encode 56 "queen" sliding moves (8 directions × 7
  distances), 8 knight moves, and 9 underpromotions (knight/bishop/rook × 3
  directions). Queen promotions reuse the sliding planes. This mapping is
  round-trip verified by the test suite for every legal move across many
  positions (including promotions, castling, and en passant).

### Search: PUCT MCTS (`mcts.py`)

For each move, MCTS runs `--simulations` iterations. Each simulation:

1. **Select** a path from the root by maximizing the PUCT score
   `Q + c_puct · P · √(ΣN) / (1 + N)`, balancing the network's prior `P` and the
   average value `Q` against how often a child has been visited `N`.
2. **Expand** the leaf: evaluate it with the network to get a value and priors
   over the leaf's legal moves.
3. **Back up** the value along the path, negating it each ply (chess is
   zero-sum). Terminal nodes use the true result: checkmate = `-1` for the side
   to move, stalemate/insufficient material/draw = `0`.

At the root, **Dirichlet noise** is mixed into the priors during self-play to
force exploration. The search returns the **visit-count distribution** over legal
moves, which is both the training target and (via its argmax or a temperature
sample) the move actually played.

### Self-play data generation (`self_play.py`)

The current network plays complete games against itself. For every position it
records `(board tensor, MCTS visit-count policy, side to move)`. Move selection
uses **temperature**: for the first `--temperature-moves` plies moves are sampled
from the visit counts (exploration/opening diversity); afterward the most-visited
move is chosen. When the game ends, the final result (`+1`/`0`/`-1`) is written
back onto every stored position as its **value target**, signed for the player
who was to move there. A **replay buffer** holds the most recent positions.

### The learning step (`train.py`)

Each iteration, after self-play, the network is trained with **Adam** on
minibatches sampled from the replay buffer, minimizing:

```
loss = policy_loss + value_loss
     = cross_entropy(policy_logits, MCTS_visit_policy)   # −Σ π·log softmax(logits)
     + MSE(value_pred, game_result)
```

plus L2 weight decay. In short: the network is pushed to (a) predict the
stronger, look-ahead MCTS policy and (b) predict who wins. The improved network
then generates better self-play data next iteration — the AlphaZero bootstrap.
Checkpoints are saved every iteration to `checkpoint_XXX.pt` and `best.pt`.

### The agent (`agent.py`)

`AlphaChessAgent` wraps a loaded model + MCTS for actual play: `play_move` picks
the most-visited move (no exploration noise), and `suggest_move` returns the best
move, its SAN/UCI, the value estimate, and the top candidates — this powers both
the `suggest` command and the GUI's **H** hint.

---

## Project layout

```
alpha_chess/
  encoding.py     board → (19,8,8) tensor and the 4672-way move ↔ index mapping
  network.py      AlphaZeroNet (residual policy/value net), device + save/load
  mcts.py         network-guided PUCT Monte-Carlo Tree Search
  self_play.py    self-play game generation and the replay buffer
  train.py        the from-scratch self-play training loop
  agent.py        high-level agent: play_move / suggest_move
  gui.py          pygame GUI to play against the agent
  cli.py          the train / play / suggest command line
tests/
  test_encoding.py   round-trip + shape tests for the encoding
requirements.txt
README.md
```

---

## Testing

```bash
.venv/bin/python -m pytest tests/ -q
```

The encoding tests verify constants, tensor shape/dtype, and that **every** legal
move round-trips through `move_to_index` / `index_to_move` across openings,
random playouts, promotions, underpromotions, castling, and en passant.

---

## FAQ / troubleshooting

- **The engine plays random/weak moves.** Expected for a small or briefly trained
  model — the value/policy are near-untrained. Train longer with a bigger network
  and more simulations.
- **GUI shows empty boxes instead of pieces.** The renderer auto-detects a font
  containing the Unicode chess glyphs (e.g. *Apple Symbols* on macOS,
  *DejaVu Sans* on Linux) and falls back to drawn lettered discs if none is
  found. This should "just work"; if not, ensure a symbol-capable system font is
  installed.
- **MPS / unsupported-op errors on Mac.** Run with `PYTORCH_ENABLE_MPS_FALLBACK=1`
  or use `--device cpu`.
- **`suggest`/`play` can't find the model.** Train first (creates
  `models/best.pt`) or pass an explicit `--model path/to/checkpoint.pt`.
