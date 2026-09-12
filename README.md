# alpha_chess

An **AlphaZero-style, self-play chess engine** built with PyTorch and
[python-chess](https://python-chess.readthedocs.io/). It learns entirely from
self-play — no human games, no opening book, no handcrafted evaluation — using a
single residual **policy + value** neural network guided by **PUCT Monte-Carlo
Tree Search (MCTS)**. It ships with a pygame GUI so you can play against the
trained agent or ask it for the best move in any position, and an `evaluate`
command to track playing strength as you train.

Self-play is the wall-clock bottleneck of the whole method, so it runs on a
**native search core**: the board, move generation, the encoding and the PUCT
tree are a small C extension (`alpha_chess/native/`) that is compiled on first
use, leaving Python with nothing to do but hand batches to the GPU. On an RTX
3090 that keeps the card at ~99% and turns self-play from a CPU problem back
into a GPU one — **~102k evaluated positions/s against ~27k** for the
pure-Python engine, which is still there as a fallback.

The defaults target a **single NVIDIA RTX 3090 (24 GB)** and a practical goal of
reaching roughly **1500–2000 Elo** with a small-but-capable network. Everything
also runs correctly on CPU (and Apple-Silicon MPS) for development and testing —
CUDA-only fast paths (fp16 autocast, GradScaler, cudnn.benchmark, channels-last,
pinned memory) are all guarded and simply fall back to fp32, and the native core
degrades to the Python engine when no compiler is available.

> **Deep dive:** [`docs/alpha_chess.pdf`](docs/alpha_chess.pdf) is an 18-page
> technical write-up of the method as implemented here — encodings, network,
> PUCT search, the self-play pipeline, the training objective, and how strength
> is measured. The [How it works](#how-it-works) section below is the shorter
> tour.

---

## Table of contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [The full workflow — zero → ~1500–2000 on an RTX 3090](#the-full-workflow--zero--15002000-on-an-rtx-3090)
4. [1. Train from scratch](#1-train-from-scratch)
5. [2. Evaluate strength](#2-evaluate-strength)
5. [3. Play in the GUI](#3-play-in-the-gui)
6. [3b. Advisor board (analyze)](#3b-advisor-board-analyze)
7. [4. Suggest the best move (CLI)](#4-suggest-the-best-move)
8. [How it works](#how-it-works)
9. [Project layout](#project-layout)
10. [Testing](#testing)
11. [FAQ / troubleshooting](#faq--troubleshooting)

---

## Installation

Requires **Python 3.9+**. Everything runs from the repository root.

```bash
# from the repo root
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Dependencies are **numpy**, **torch**, **python-chess**, **pygame** and
**ziglang**. `python-chess` already provides `chess.engine`, used for the
optional UCI-engine opponent in `evaluate`.

`ziglang` is only a **compiler**, not a runtime dependency: the native search
core is built the first time self-play needs it, and any of `cc`, `gcc`,
`clang` (or `$CC`) is used in preference. It is in `requirements.txt` because it
is pip-installable and needs no root, so a host with no system compiler still
gets the fast path. To build it ahead of time, or to find out why it did not
build:

```bash
.venv/bin/python -m alpha_chess.native          # build + import, prints the path
.venv/bin/python -m alpha_chess.native --force  # rebuild from scratch
```

Training prints which engine it is using and, when the native core is
unavailable, why. `--engine python` forces the fallback.

If pip cannot find a suitable PyTorch build on CPU-only hosts, install the CPU
wheel explicitly (on Apple Silicon this wheel still includes GPU/MPS support):

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

On an RTX 3090 host, install a **CUDA** build of PyTorch (see the
[PyTorch install matrix](https://pytorch.org/get-started/locally/)) so
`--device auto` can select the GPU and mixed precision can activate.

Both `python -m alpha_chess` and `python -m alpha_chess.cli` invoke the same CLI.
The examples below use `.venv/bin/python`; if you've activated the venv
(`source .venv/bin/activate`) you can just write `python`.

---

## Quick start

```bash
# 1) Train a TINY model fast (a few minutes on CPU) — writes models/best.pt
.venv/bin/python -m alpha_chess.cli train \
  --iterations 3 --games-per-iter 8 --simulations 40 --no-playout-cap \
  --pools 2 --games-in-flight 4 --batch-size 64 --buffer-size 5000 \
  --channels 32 --blocks 4 --out models --seed 0

# 2) See how it does against a trivial baseline
.venv/bin/python -m alpha_chess.cli evaluate \
  --model models/best.pt --opponent random --games 20 --simulations 40

# 3) Play in a window (H = hint, U/R = undo/redo, N = new game, F = flip)
.venv/bin/python -m alpha_chess.cli play --model models/best.pt

# 4) Ask for the best move in any position (FEN)
.venv/bin/python -m alpha_chess.cli suggest \
  --fen "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3" \
  --model models/best.pt --simulations 200
```

> **Note:** A freshly/briefly trained network plays weakly and often near-random
> (the top moves show roughly uniform probabilities). Real strength needs a
> serious training run — see the
> [recommended RTX 3090 recipe](#recommended-rtx-3090-recipe) and
> [training cost & expectations](#training-cost--expectations).

---

## The full workflow — zero → ~1500–2000 on an RTX 3090

Every command you need, in order, on the training machine. Run them from the
repo root with the venv active (or prefix `.venv/bin/`). Details for each step
are in the sections below; this is the copy-paste playbook.

**0 — Install (once), with a CUDA build of PyTorch for the 3090:**

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
# CUDA 12.1 wheel (matches most recent drivers); pick the build for your CUDA:
.venv/bin/pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available())"   # -> CUDA: True
```

**1 — Start training from scratch** (`--device auto` picks CUDA; the native
search core, AMP/fp16, cudnn.benchmark and channels-last all activate
automatically; periodic self-eval every 5 iterations):

```bash
.venv/bin/python -m alpha_chess.cli train \
  --device auto \
  --iterations 80 \
  --games-per-iter 16000 \
  --pools 3 --games-in-flight 1024 \
  --simulations 200 --fast-simulations 50 --full-search-prob 0.25 \
  --epochs 4 --batch-size 1024 --sample-reuse 4 \
  --lr 1e-3 --channels 128 --blocks 10 \
  --buffer-size 2000000 \
  --temperature-moves 30 --max-moves 400 \
  --eval-every 5 --eval-games 40 \
  --checkpoint-every 1 \
  --out models --seed 0
```

**2 — Resume/continue** any time (after a stop, crash, or to add more
iterations). Restores model + optimizer + iteration + RNG and continues:

```bash
.venv/bin/python -m alpha_chess.cli train --resume models --out models \
  --device auto --iterations 160 \
  --games-per-iter 16000 --pools 3 --games-in-flight 1024 \
  --channels 128 --blocks 10
```

**3 — Check strength as it trains** (in another shell; each is a standalone run):

```bash
# sanity: should crush a random mover once training has done anything
.venv/bin/python -m alpha_chess.cli evaluate --model models/best.pt \
  --opponent random --games 40 --simulations 200

# measure improvement vs an earlier checkpoint of your own model
.venv/bin/python -m alpha_chess.cli evaluate --model models/best.pt \
  --opponent model:models/checkpoint_020.pt --games 60 --simulations 400
```

**4 — Anchor to a REAL Elo** with a calibrated engine. Install Stockfish and
pass its path — `apt install stockfish` puts it at `/usr/games/stockfish`
(which is not on a non-root `PATH`), Homebrew at `/opt/homebrew/bin/stockfish`;
`which stockfish || ls /usr/games/stockfish` finds it. Bracket the level by
trying a few `--uci-elo` caps — the target is a ~50% score against
~1500–2000:

```bash
.venv/bin/python -m alpha_chess.cli evaluate --model models/best.pt \
  --opponent uci:/usr/games/stockfish --uci-elo 1500 \
  --games 60 --simulations 800

.venv/bin/python -m alpha_chess.cli evaluate --model models/best.pt \
  --opponent uci:/usr/games/stockfish --uci-elo 2000 \
  --games 60 --simulations 800
```

**5 — Use the finished model:** play it, or use the advisor board to get the
best move for a game you're playing elsewhere:

```bash
.venv/bin/python -m alpha_chess.cli play    --model models/best.pt   # play a game
.venv/bin/python -m alpha_chess.cli analyze --model models/best.pt   # advisor board
```

> Reaching 1500–2000 from scratch is a **long run (many hours to days)** — see
> [training cost & expectations](#training-cost--expectations). Keep step 1 (or
> its resume in step 2) running, and periodically run steps 3–4 to track
> progress. `--pools 3 --games-in-flight 1024` saturates a 3090; lower
> `--games-in-flight` if you hit RAM limits (each in-flight game holds its own
> search tree and its own accumulated training examples).

---

## 1. Train from scratch

Training alternates **batched self-play** (generate games with the current
network + MCTS) and **learning** (train the network on those games). Each
iteration writes a numbered checkpoint `checkpoint_{i:03d}.pt`, always mirrors
the latest weights to `best.pt`, and writes a **resumable** `train_state.pt`
(model + optimizer + iteration + RNG states).

The **defaults** below are tuned for an RTX 3090 aiming at ~1500–2000 Elo. They
build a `channels=128, blocks=10` residual network — small enough to run many
self-play games and large batches quickly on a 3090, but deep enough to learn
real chess. Scale the network **up** (`--channels 192 --blocks 15`) for more
ceiling at higher compute cost, or **down** (`--channels 64 --blocks 5`) for
faster iteration / CPU experiments; `agent`, `gui`, and `evaluate` read the
architecture from each checkpoint's stored config, so any size just works.

```bash
.venv/bin/python -m alpha_chess.cli train \
  --iterations 40 \
  --games-per-iter 16000 \
  --simulations 200 \
  --fast-simulations 50 \
  --full-search-prob 0.25 \
  --pools 3 \
  --games-in-flight 1024 \
  --epochs 4 \
  --batch-size 1024 \
  --sample-reuse 4 \
  --lr 1e-3 \
  --channels 128 \
  --blocks 10 \
  --buffer-size 2000000 \
  --temperature-moves 30 \
  --max-moves 400 \
  --out models \
  --seed 0
```

### Training flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--iterations` | `40` | Number of self-play -> train cycles. Also the length of the cosine LR schedule. |
| `--games-per-iter` | `4000` | Self-play games generated each iteration. |
| `--simulations` | `200` | MCTS simulations per move on **full-search** plies. Higher = stronger data, slower. |
| `--engine` | `auto` | `auto` \| `native` \| `python`. `auto` uses the native C core whenever it can be built, and says why if it cannot. |
| `--pools` | `3` | Independent native game pools. One pool's tree search overlaps the next pool's forward pass, so 2 is the minimum useful value. Native engine only. |
| `--games-in-flight` | `1024` | Concurrent games searched **per pool** (native) or **per worker** (Python). This is the network's batch size, and wider is much more efficient on the GPU. The default suits the native engine; with `--engine python` use ~128, since every worker stages a buffer of its own. |
| `--fast-simulations` | `50` | Simulation budget for plies that playout-cap randomisation does not search fully. Native engine only. |
| `--full-search-prob` | `0.25` | Probability a ply gets the full `--simulations` budget, root Dirichlet noise, and a recorded training target. See [playout-cap randomisation](#playout-cap-randomisation). Native engine only. |
| `--no-playout-cap` | off | Shorthand for `--full-search-prob 1.0`: search every ply fully (classic AlphaZero). |
| `--fpu-reduction` | `0.0` | First-play-urgency penalty for unvisited children. `0` treats them as drawn (AlphaZero); `0.2`-`0.3` makes the search commit to promising moves sooner. Native engine only. |
| `--num-workers` | ¾ of CPU count, capped at 24 | Search **processes** for the *Python* engine only; that search is GIL-bound, so it is that engine's main throughput lever. The native core ignores it. |
| `--pipeline-stages` | `4` | Sub-pools per worker (Python engine only). |
| `--resign-threshold` | `-0.90` | Resign once the mover's best root value stays at or below this for two plies. `--no-resign` plays every game out. |
| `--resign-disable-fraction` | `0.10` | Fraction of games played out with resignation suppressed, to measure the resign false-positive rate (printed each iteration). |
| `--no-save-buffer` | off | Skip persisting the replay buffer (it is otherwise written next to `train_state.pt` so resumes keep their history). |
| `--epochs` | `4` | Optimization passes per iteration. |
| `--sample-reuse` | `4.0` | How many times each position should be sampled over its **lifetime in the replay buffer**. This sets the gradient-step count, which then scales with the data rather than with the buffer; `0` restores the old rule (one pass over the whole replay buffer per epoch). |
| `--train-steps` | `0` | Explicit gradient steps per iteration; overrides `--sample-reuse`. |
| `--batch-size` | `1024` | Minibatch size for the gradient step. |
| `--lr` | `1e-3` | Adam learning rate (initial value, before cosine decay). |
| `--lr-final` | `None` -> `lr*0.1` | Final learning rate for cosine decay across `--iterations`. |
| `--grad-clip` | `1.0` | Max gradient norm (`clip_grad_norm_`). |
| `--channels` | `128` | Width of the residual tower (conv channels). Bigger = stronger, slower. |
| `--blocks` | `10` | Number of residual blocks (depth). Bigger = stronger, slower. |
| `--buffer-size` | `2000000` | Replay buffer capacity in **positions** (numpy ring buffer; oldest evicted). Positions cost ~1.9 KB each, so 2M is ~3.8 GB. |
| `--temperature-moves` | `30` | For the first N plies, moves are **sampled** from MCTS visit counts (exploration); after that the **best** move is played. |
| `--max-moves` | `400` | Cap on plies per self-play game (cutoff scored as a draw). |
| `--weight-decay` | `1e-4` | L2 regularization for Adam. |
| `--amp` / `--no-amp` | AMP **on** | Enable/disable fp16 automatic mixed precision. Active only on CUDA; a no-op (fp32) on CPU/MPS. |
| `--compile` | off | Wrap the model with `torch.compile` (CUDA + torch>=2 only; guarded). |
| `--eval-every` | `0` | Evaluate every N iterations (0 disables). Requires `--eval-games > 0`. |
| `--eval-games` | `0` | Games per periodic in-training evaluation (0 disables). |
| `--checkpoint-every` | `1` | Save a numbered checkpoint every N iterations (`best.pt` + `train_state.pt` are always written). |
| `--resume` | `None` | Path to a `train_state.pt` (or a dir containing one) to **continue**, or a plain model checkpoint to load **weights only** and start a fresh optimizer. |
| `--seed` | `None` | Seeds `torch`/`numpy`/`random` for reproducibility. |
| `--device` | `auto` | `auto` \| `cpu` \| `cuda` \| `mps`. See [Devices](#devices-cpu--gpu--apple-silicon). |

Each iteration prints mean policy/value losses plus **throughput**: self-play
games/hour and evaluated positions/s, the mean inference batch, the split
between search and waiting-on-GPU, and training steps/s.

### Playout-cap randomisation

Most of a self-play game's search budget buys very little: the position is not
close, and a 200-simulation search picks the move a 50-simulation one would.
Playout-cap randomisation (from KataGo - Wu, *Accelerating Self-Play Learning in
Go*, 2019) exploits that. With `--full-search-prob 0.25`:

- a random **25%** of plies get the full `--simulations` budget, root Dirichlet
  noise, **and** produce a recorded policy target;
- the other **75%** get only `--fast-simulations` and produce no training
  example at all - they exist to move the game along and reach a result.

Measured here over 400 complete games, that takes a ply from 179 evaluated
positions to 77, so the same GPU-hour finishes **2.2x as many games** - 2.2x as
many game outcomes to learn a value function from - while every policy target it
records still comes from a full-strength search. Subtree reuse makes it cheaper
still: a fast ply often already has more carried-over visits than its budget
calls for, and costs a single simulation.

Set `--no-playout-cap` to search every ply fully. Note that with playout caps on
an iteration records ~4x fewer positions per game, which the `--sample-reuse`
rule accounts for automatically.

### Recommended RTX 3090 recipe

On a 3090, `--device auto` selects **CUDA** automatically, and — because
`--amp` is on by default — fp16 mixed precision, `cudnn.benchmark` and
channels-last memory format all activate automatically, as does the native
search core. You do not need any extra flags to turn the GPU on.

```bash
.venv/bin/python -m alpha_chess.cli train \
  --device auto \
  --iterations 80 \
  --games-per-iter 16000 \
  --pools 3 --games-in-flight 1024 \
  --simulations 200 --fast-simulations 50 --full-search-prob 0.25 \
  --epochs 4 --batch-size 1024 --sample-reuse 4 \
  --lr 1e-3 \
  --channels 128 --blocks 10 \
  --buffer-size 2000000 \
  --temperature-moves 30 \
  --max-moves 400 \
  --eval-every 5 --eval-games 40 \
  --checkpoint-every 1 \
  --out models --seed 0
```

Notes on the knobs:

- **`--games-in-flight` is now the main throughput lever.** It is the network's
  batch size, and with the native core the GPU is the bottleneck, so wider is
  better: a 128×10 net runs at ~94k positions/s at batch 512, ~121k at 1024 and
  ~127k at 2048. 1024 per pool is the sweet spot on a 3090; it costs a few GB of
  host RAM for the in-flight games' trees and accumulated examples.
- **`--pools`** exists purely to overlap: while one pool's batch is on the GPU,
  the next pool's tree search runs on the CPU. Two suffice to hide the search
  entirely; three absorb the jitter from a pool that momentarily has few active
  games. More than that just fragments the batches.
- **Keep `--games-per-iter` well above `pools × games-in-flight`** (here
  16000 vs 3072). A pool refills a finished game with a fresh one until its
  quota is exhausted, after which the batch decays as the last games drain; the
  larger the ratio, the smaller the share of the iteration spent in that tail.
- **`--num-workers` no longer matters** unless you pass `--engine python`. The
  native search uses about a third of one core and leaves the rest of the
  machine idle, because the GPU cannot keep up with even that.
- **`--simulations`** trades data quality against speed; 200 is a solid default.
  Combined with `--full-search-prob 0.25` it costs about as much per ply as 90
  simulations would.
- **`--channels` / `--blocks` now set self-play speed directly**, since the GPU
  is the bottleneck: 128×10 runs at ~121k positions/s at batch 1024, 96×8 at
  ~207k and 160×12 at ~65k. A smaller net buys games/hour at the cost of
  ceiling; 128×10 is the balance point for the 1500–2000 target.
- **`--sample-reuse`** keeps the optimizer's share of the wall clock fixed. The
  old rule (one pass over the whole replay buffer per epoch) drew every position
  ~20 times per iteration once the buffer filled, which is both slow and more
  reuse than the data supports.
- **`--resign-threshold`** cuts decided games short, which is a large share of
  the win at long time controls. The iteration log prints the false-positive
  rate measured on the `--resign-disable-fraction` of games played out anyway;
  if it climbs above ~5%, lower the threshold (e.g. `-0.95`).
- **`--buffer-size`** of ~2M positions keeps a broad, recent history in ~3.8 GB
  of RAM. Positions are stored uint8-packed with sparse policy targets, ~1.9 KB
  each (see [replay buffer](#efficient-replay-buffer)).
- **`--eval-every 5 --eval-games 40`** plays the current model against a baseline
  every 5 iterations and prints the score + estimated Elo delta, so you can
  watch progress. Evaluation is wrapped in try/except and never crashes
  training.

**Resuming.** Training is resumable from the `train_state.pt` written into the
output directory. Point `--resume` at the file or its directory:

```bash
.venv/bin/python -m alpha_chess.cli train --resume models --out models \
  --iterations 160 --channels 128 --blocks 10 --games-in-flight 1024
```

This restores the model, optimizer, iteration counter, RNG states **and the
replay buffer** (`replay_buffer.npz`, written next to `train_state.pt` unless
`--no-save-buffer`) and **continues from the next iteration**. If you instead
point `--resume` at a plain model checkpoint (e.g. `checkpoint_012.pt`), it
loads the **weights only**
and starts a fresh optimizer — useful for fine-tuning. Missing optimizer state
never crashes the run.

### Training cost & expectations

Be realistic: reaching **1500–2000 Elo from scratch** on one 3090 still takes
**substantial wall-clock — realistically a couple of days** — and depends
heavily on how much self-play you generate (games × moves × simulations). What
changed is where the time goes.

Self-play *used* to be limited by CPU-bound legal-move generation and board
logic in python-chess. Profiling the pure-Python engine put the time here:

| | share of self-play |
|---|---|
| python-chess legal move generation | ~40% |
| Move -> policy index | ~14% |
| Board -> tensor encoding | ~10% |
| Tree descent (push, PUCT select, backup) | ~17% |

With all of that moved into the [native
core](#the-native-search-core-alpha_chessnative), the GPU is the bottleneck
again, and it runs at its power limit.

Measured on this repo's target host (RTX 3090, 8-core/16-thread i9-11900KF)
with a 128×10 net, 200 simulations, and an untrained network:

| | evaluated positions/s | evals per ply | self-play games/h |
|---|---|---|---|
| Python engine, 12 workers, batch ~470 | ~27,000 | 179 | ~2,800 |
| **Native core, 3 pools, batch 1024** | **~102,000** | 179 | **~10,600** |
| **Native core + playout-cap randomisation** | **~102,000** | **77** | **~24,000** |

positions/s is measured directly at steady state; games/h is derived from it and
from the measured cost of a game (194 plies at 179 evals/ply for a full search,
199 plies at 77 with playout caps — each measured over 400 complete games). The
last two rows share a positions/s because both saturate the GPU: playout-cap
randomisation does not make the card faster, it spends the same positions on
**2.2× as many finished games**.

End to end that is **~3.8× the raw search throughput and ~8.5× the games per
hour**, so a run that needed a week is now a long weekend.

Those absolute rates are pessimistic: with an untrained network the visit
distribution is nearly flat, so subtree reuse saves almost nothing (179 of 200
simulations per ply are new work) and no game ever resigns. Both improve
substantially once the network has learned anything — the ratios are what to
carry forward, not the absolute numbers.

For reference, the component rates behind all of this:

| | rate |
|---|---|
| python-chess `list(board.legal_moves)` | ~35,000 positions/s per core |
| Native legal move generation (bulk perft) | ~350,000,000 nodes/s |
| Network forward pass, 128×10 @ batch 1024 | ~121,000 positions/s |
| Network forward pass, 128×10 @ batch 2048 | ~127,000 positions/s |

The iteration log prints the mean batch size, games/hour and the split between
search and waiting-on-GPU, so you can see which side is short on your hardware.
Plan for a long run, checkpoint often, and track strength with `evaluate`. This
project is a faithful, runnable *implementation* of the AlphaZero method tuned
for a single GPU — not a promise of grandmaster strength on a laptop.

### Devices (CPU / GPU / Apple Silicon)

`--device auto` picks, in order:

1. **CUDA** (NVIDIA GPU) if available — the RTX 3090 case;
2. **MPS** (Apple Silicon GPU) if available;
3. **CPU** otherwise.

On CUDA, training enables `cudnn.benchmark`, moves the model to
`channels_last`, uses fp16 autocast + a `GradScaler`, and copies host→device
tensors from pinned memory with `non_blocking=True`. On **CPU/MPS** all of these
are disabled and everything runs in fp32 with no errors — this is exactly the
path used for the test suite. Force a device with `--device cpu` / `--device
mps` / `--device cuda`. If you hit an unsupported-op error on MPS, set
`PYTORCH_ENABLE_MPS_FALLBACK=1` or use `--device cpu`.

---

## 2. Evaluate strength

The `evaluate` command plays a trained model against an opponent and reports
**W/D/L**, a **score** in `[0, 1]`, and an **estimated Elo difference** relative
to that opponent. Colors alternate across games for fairness. Matches run on the
interactive [PUCT search](#interactive-search-puct-mcts-mctspy), which uses fp16
on CUDA like the rest of the pipeline; at 200 simulations a game costs roughly
8 seconds on a 3090, so a 40-game match is about 5 minutes.

```bash
# vs a random-mover baseline (sanity check that training did something)
.venv/bin/python -m alpha_chess.cli evaluate \
  --model models/best.pt --opponent random --games 40 --simulations 200

# vs a 1-ply greedy material grabber (a real, if weak, baseline)
.venv/bin/python -m alpha_chess.cli evaluate \
  --model models/best.pt --opponent material --games 40 --simulations 200

# vs a previous checkpoint of your own model (measure improvement)
.venv/bin/python -m alpha_chess.cli evaluate \
  --model models/best.pt --opponent model:models/checkpoint_010.pt --games 40

# vs an external UCI engine (e.g. Stockfish), capped to ~1600 Elo if supported
.venv/bin/python -m alpha_chess.cli evaluate \
  --model models/best.pt --opponent uci:/usr/games/stockfish \
  --uci-elo 1600 --games 40 --simulations 400
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | `models/best.pt` | Checkpoint to evaluate (played as player A). Exits with an error if missing. |
| `--opponent` | `material` | `random` \| `material` \| `model:PATH` \| `uci:PATH`. |
| `--games` | `40` | Number of games (colors alternate across games). |
| `--simulations` | `200` | MCTS simulations per move for the evaluated model. |
| `--uci-elo` | `None` | If the UCI opponent supports `UCI_LimitStrength`/`UCI_Elo`, cap it to this Elo. Engines enforce a **floor** — 1320 on Stockfish 16 — and a lower request is clamped **with a warning**; use `--uci-skill` below that. |
| `--uci-skill` | `None` | If the UCI opponent has a `Skill Level` option (0–20 on Stockfish), set it. Level 0 is far weaker than any `--uci-elo` can reach, which is what you want for a model that is not yet beating `material`. Mutually exclusive with `--uci-elo`. |
| `--max-moves` | `300` | Move cap per game (games hitting the cap are scored as draws). |
| `--opening-plies` | `4` | Random opening plies per game/color-swapped pair, so two deterministic players produce varied games (0 = always the start position). |
| `--seed` | `None` | Seed for stochastic opponents, opening randomization, and color alternation. |
| `--device` | `auto` | Inference device (see above). |

Example output:

```
Evaluation vs material:
  Games : 40
  W/D/L : 27 / 6 / 7
  Score : 0.750
  Elo   : +191 (relative to opponent)
Note: Elo is RELATIVE to this opponent, not an absolute rating; ...
```

**Limiting engine strength.** Stockfish exposes two independent throttles and
ignores its skill level whenever `UCI_LimitStrength` is on, so `evaluate` takes
one or the other and errors if you pass both:

```bash
# Elo-limited: cannot go below the engine's floor (1320 on Stockfish 16)
.venv/bin/python -m alpha_chess.cli evaluate --model models/best.pt \
  --opponent uci:/usr/games/stockfish --uci-elo 1600 --games 40

# Skill-limited: reaches genuinely weak play, for an early model
.venv/bin/python -m alpha_chess.cli evaluate --model models/best.pt \
  --opponent uci:/usr/games/stockfish --uci-skill 0 --games 40
```

The match header names the setting that was actually applied
(`uci:/usr/games/stockfish @ Skill Level 0`), so a clamped request cannot be
mistaken for the one you typed. Note also that a strength-limited Stockfish is
not a human of that rating: it plays weaker moves but still does not hang
pieces, so it tends to beat a human of the same nominal number.

**Honesty about Elo.** The reported Elo difference is computed from the match
score (`elo = -400·log10(1/score - 1)`, clamped at scores of 0 or 1) and is
**relative to the specific opponent you chose**. Beating the `random` or
`material` baselines by a large margin says little about absolute rating — those
opponents are far below 1500. To anchor a number you can compare to a real
rating, evaluate against a **calibrated UCI engine** (e.g. Stockfish with
`--uci-elo` set) or play the model online. Treat internal numbers as a relative
progress signal, not an official rating.

---

## 3. Play in the GUI

Launch the pygame board.

```bash
.venv/bin/python -m alpha_chess.cli play \
  --model models/best.pt \
  --simulations 200 \
  --color white
```

`--model` defaults to `models/best.pt`, so if you've trained with the default
settings you can just run `play` with no flags. If the model file **cannot be
found** (you haven't trained, or you point `--model` at a missing path), the GUI
falls back to **human vs human** with the agent and hints disabled.

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | `models/best.pt` | Checkpoint to load. If the file is missing → human vs human. |
| `--simulations` | `200` | MCTS simulations per agent move (higher = stronger, slower). |
| `--color` | `white` | Which side **you** play. |
| `--device` | `auto` | Inference device (see above). |
| `--setup` | *(off)* | Start in the board-editor / analysis (setup) mode. |

**Controls**

- **Click** a piece, then click its destination to move. Pawns auto-promote to a
  queen.
- **H** / **SPACE** / **A** — hint: highlights the engine's suggested move and
  shows its evaluation and top candidates in the side panel.
- **U** — undo the last move (a full human+agent pair when playing the engine).
- **R** — redo a move taken back with **U**, restoring the same pair. The
  agent's reply is replayed, not re-searched, so redo exactly reverses the undo
  and is instant. Playing a different move instead discards the redo history.
- **N** — new game.
- **F** — flip the board.
- **M** — mute / unmute the move sounds.
- **E** — enter the **board editor** (see below).
- **ESC / Q** — quit.

The piece that moved last is **boxed in green** on the square it now occupies
— so whenever it is your turn, the box is around your opponent's piece. It is
a hard-edged border rather than a tint, which stays legible on light and dark
squares alike. Each move also plays a short wooden
click, with a lower, fuller knock for a capture. The clicks are synthesised at
startup (no audio files ship with the package) and are silently skipped on a
machine with no sound device; **M** mutes them.

The side panel shows whose turn it is, check/checkmate/stalemate/draw status, a
"thinking…" indicator while the agent searches, and the latest evaluation.

---

## 3b. Advisor board (analyze)

The **advisor board** is for when you're playing a real game against another
person in a *separate* application and want AlphaChess open alongside as a
coach. **You** make every move for **both** colours (mirroring the external
game) and, at any moment, ask the engine for the best move for whoever is to
move. **The engine never moves a piece on its own** — it only recommends.

```bash
.venv/bin/python -m alpha_chess.cli analyze \
  --model models/best.pt \
  --simulations 400
```

`--model` defaults to `models/best.pt`. If no model is loaded the board still
works for making moves; asking for a suggestion just shows
"Load a model (--model) for suggestions."

**Controls (advisor board)**

- **Click** a piece then its destination to move — this works for **both**
  White and Black, alternating naturally with whoever is on move. Only legal
  moves are accepted; pawns auto-promote to a queen.
- **H** / **SPACE** — ask for the best move for the side to move: the from/to
  squares are highlighted and the panel shows the best move (SAN), its eval
  (`[-1, +1]`, side-to-move perspective), and the top 5 candidates with their
  visit-count percentages. Re-runnable after every move.
- **U** — undo the last move (a **single** ply — you made it).
- **R** — redo a ply taken back with **U**. Playing a different move instead
  discards the redo history.
- **E** — open the **board editor** (below) to set up a mid-game position when
  you join a game already in progress.
- **F** — flip the board.  **N** — new game (standard start position).
- **M** — mute / unmute the move sounds.
- **Q / ESC** — quit.

The piece that moved last is boxed in green, and every move plays a short
click — useful here for confirming that a move you mirrored from the other
application actually registered.

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | `models/best.pt` | Checkpoint to load. If the file is missing → suggestions disabled (the board still works). |
| `--simulations` | `400` | MCTS simulations per suggestion. |
| `--fen` | *(none)* | Optional FEN for the STARTING position to follow from. |
| `--setup` | *(off)* | Start in the board editor first (to join a game in progress); applying the position lands on the advisor board. |
| `--device` | `auto` | Inference device. |

### Board editor (set up a mid-game position)

Press **E** on the advisor board (or start with `analyze --setup`) to open the
palette editor and set up **any** position by hand — useful when you join a game
that's already underway. Entering the editor copies the current board (pieces,
side to move, castling rights) so you can tweak it or start fresh.

**Editing**

- The side panel shows a **piece palette** — a row of white pieces (K Q R B N P),
  a row of black pieces (k q r b n p), and an **eraser** cell. Click a cell to
  pick the current *brush* (highlighted).
- **Left-click** a board square to place the selected brush (overwriting any
  piece); with the eraser brush selected it clears the square.
- **Right-click** a square to erase it regardless of the brush.
- **T** — toggle the side to move (White/Black).
- **C** — clear the whole board.  **R** — reset to the standard start position.
- **X** — select the eraser brush.
- **K** — clear all castling rights. (Rights are otherwise auto-granted whenever
  the relevant king **and** rook are on their home squares.)
- **F** — flip the board.

**Analysis**

- **SPACE** or **A** — validate the position and, if legal, analyze it. The
  best move's from/to squares are highlighted on the board and the panel shows
  the best move (SAN + eval in `[-1, +1]`, side-to-move perspective) plus the
  top 5 candidates with their visit-count percentages. Re-run after edits to
  refresh. Analysis works for whichever side is to move.
- **P** — "use this position": if legal, adopt the edited position as a fresh
  game. Opened from the advisor board, this returns you to the advisor board
  (you keep moving both sides); from vs-agent `play` it returns to play mode.
- **E / ESC** — leave the editor (cancel) and return to the board you came from.
  **Q** — quit.

Invalid positions are rejected with a specific reason (missing/too many kings,
pawns on a back rank, the side *not* to move being in check, too many pieces,
etc.) and are never analyzed or adopted.

---

## 4. Suggest the best move

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
| `--model` | `models/best.pt` | Checkpoint to load. Exits with an error if the file is missing. |
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

- **Input:** a `(21, 8, 8)` tensor encoding the board (see *Encoding* below).
- **Body:** a 3×3 conv **stem** (→ `channels`) + BatchNorm + ReLU, followed by
  `num_blocks` **residual blocks** (each: conv3×3 → BN → ReLU → conv3×3 → BN,
  plus a skip connection, then ReLU).
- **Policy head:** 1×1 conv → BN → ReLU → flatten → linear → **4672 logits**,
  one per possible move (see *Move encoding* below). Raw logits are masked to the
  legal moves and soft-maxed at use time.
- **Value head:** 1×1 conv → BN → ReLU → flatten → linear(64) → ReLU → linear(1)
  → **tanh**, giving a scalar in `[-1, +1]` — the expected game result from the
  side-to-move's perspective.

Defaults are **`channels=128`, `num_blocks=10`** (both configurable from the
CLI) — a small-but-capable ResNet sized for the ~1500–2000 target and fast on a
3090. Checkpoints store the architecture config alongside the weights, so
`load_model` rebuilds the exact network automatically; `agent`, `gui`, and
`evaluate` therefore work with any size you trained.

### Board & move encoding (`encoding.py`)

- **Board → tensor:** 21 planes of 8×8, all **side-to-move relative** — the
  board is vertically flipped when Black is to move, so the network always sees
  the player to move at the bottom moving "up". That halves what it has to
  learn: a motif never has to be represented twice, once per colour. The planes
  are 6 *our* piece types, 6 *their* piece types, our two and their two castling
  rights, the en-passant target square, a halfmove-clock plane, two repetition
  planes (has this position occurred once / twice before?), and a constant
  all-ones plane that lets padded convolutions locate the board edge.

  The flip is internal to `encoding.py`: `encode_board`, `move_to_index` and
  `index_to_move` all take the board and apply (or undo) the orientation
  themselves, so callers work in ordinary absolute `chess` coordinates.
- **Move ↔ index:** the AlphaZero **8×8×73 = 4672** move representation. For each
  from-square, 73 planes encode 56 "queen" sliding moves (8 directions × 7
  distances), 8 knight moves, and 9 underpromotions (knight/bishop/rook × 3
  directions). Queen promotions reuse the sliding planes. This mapping is
  round-trip verified by the test suite for every legal move across many
  positions (including promotions, castling, and en passant).

### Interactive search: PUCT MCTS (`mcts.py`)

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
sample) the move actually played. `mcts.py` powers **interactive** single-position
search for the GUI, `suggest`, and `evaluate`; self-play uses the batched engines
instead.

Searching one position at a time is a **latency** problem, not a throughput one,
and two things dominated it:

- **A batch-of-one forward pass is entirely host-launch bound.** ~23 tiny
  convolutions cost ~3.1 ms to *issue* and microseconds to run. The same CUDA-graph
  replay the self-play driver uses issues the whole pass in one call: 0.67 ms per
  simulation including reading the result back. The legal-move softmax also
  happens on the device, with the value concatenated onto it, so one small
  transfer — and so one synchronisation — serves the whole call.
- **`is_game_over(claim_draw=True)` was called at every node of every descent**
  (~164 µs each; the expensive part is `can_claim_threefold_repetition`, which
  tries every legal move). Terminal status is now resolved once per node, at the
  leaf, from the legal-move list the leaf needs anyway, and cached on the node so
  later descents stop there for free.

Together those took a 40-game / 200-simulation evaluation match from ~23 minutes
to ~5 (measured: 8 games in 4 m 40 s before, 1 m 04 s after). Anchoring against a
UCI engine at `--games 60 --simulations 800` drops from most of a day to about an
hour and a half.

### The native search core (`alpha_chess/native/`)

Self-play is the wall-clock bottleneck of AlphaZero, and in a Python
implementation it is a **CPU** bottleneck, not a GPU one: ~80% of the pure-Python
engine's time went into four routines (see [training
cost](#training-cost--expectations)), none of which the GPU can help with, and it
left an RTX 3090 about a quarter busy. So all four live in C now:

- **`bitboard.h`** — knight/king/pawn attack tables and fancy **magic
  bitboards** for the sliders. The magic multipliers are searched for at
  start-up with a fixed PRNG seed (~30 ms, deterministic) rather than shipped as
  a table of hand-copied constants.
- **`position.h`** — the position, Zobrist hashing, make-move, and **fully
  legal** move generation: pins, check evasions and castling rights are resolved
  up front instead of generating pseudo-legal moves and filtering them. It runs
  at ~350M nodes/s in bulk perft, against python-chess's ~35k positions/s.
- **`encode.h`** — the 21-plane encoding and the 4672-way move index. The
  side-to-move mirror that costs a NumPy copy in Python is a single `bswap64`
  here.
- **`mcts.h`** — the tree itself: an arena-allocated node pool per game, PUCT
  selection, Dirichlet noise, subtree reuse (by compacting the retained subtree
  into a spare arena), incremental repetition tracking, temperature sampling,
  resignation, and the training-example accumulator.
- **`fastchess.c`** — the CPython bindings. Arrays cross the boundary through
  the buffer protocol, so no NumPy headers are needed to build, and both
  `collect()` and `apply()` release the GIL.

Two search refinements the Python engine does not have are configurable here,
both neutral at their default values: **first-play urgency**
(`--fpu-reduction`), which gives an unvisited child the parent's value minus a
penalty scaled by the explored prior mass instead of a flat 0, and
[playout-cap randomisation](#playout-cap-randomisation).

**Correctness.** A hand-written move generator is exactly the kind of code that
is 99% right and therefore wrong, so it is pinned down from both ends:
`tests/test_native.py` runs **perft** on the six standard positions (~194M nodes
between them, covering castling, en passant, promotion, discovered check, pins
and stalemate) and diffs the native legal-move set, move indices and encoded
planes against `python-chess` and `alpha_chess.encoding` across thousands of
positions from random games. The two encoders must agree exactly, because the
GUI, `suggest` and `evaluate` all feed the network through the *Python* one.

Rules and encoding being right does not prove the *search* is, so the two
engines were also run head to head on the same network and the same settings
(600 games each, 100 simulations, 64×4 net). They agree on everything that would
move if the search had a perspective or backup bug:

| | native | Python |
|---|---|---|
| mean game length | 114.2 plies | 116.1 plies |
| win / draw / loss | .234 / .535 / .231 | .220 / .563 / .217 |
| mean policy-target entropy | 3.180 | 3.171 |
| wall clock | **29 s** | 235 s |

**Building.** `alpha_chess/native.py` compiles the extension on first use and
caches it next to the package, rebuilding whenever a source file is newer. It
tries `$CC`, then `cc`/`gcc`/`clang`, then `python -m ziglang cc` — so a host
with no system compiler and no root still gets the fast path. If none of them
works it says so, and training transparently falls back to the Python engine.

### Native self-play (`native_selfplay.py`)

With the search in C the pipeline gets much simpler. One core produces leaves
faster than a 3090 can evaluate them, so there are no worker processes, no
shared memory and no serialization — just a few **pools** in one process.

**Pools overlap the CPU and the GPU.** Each pool owns its own games and its own
pinned staging buffers. While pool *k*'s batch is on the GPU, pool *k+1*'s
descent runs on the CPU; CUDA work is asynchronous and the C search releases the
GIL, so neither side waits for the other.

**One CUDA graph covers the whole evaluation.** Not just the forward pass: the
legal-move gather, the mask, the softmax and the dtype casts are captured too.
Left eager, those ~10 small kernels plus their tensor bookkeeping cost ~2.8 ms
of host time per pass — more than the forward pass itself at batch 1024 — and
the *host* becomes the limit again (measured: 65k positions/s eager against
~102k captured). Captured, issuing a pass costs ~0.05 ms. Graphs are captured
fresh each self-play phase, so one can never serve stale weights, and batch
sizes are rounded up to a multiple of 128 to keep the number of captured shapes
small.

**Every host transfer is a plain contiguous DMA.** The graph's static inputs
take the engine's own dtypes (float32 states, int32 indices, int32 counts) and
the casts happen inside the graph, so nothing on the host path is a strided or
dtype-converting copy — those quietly fall off the async DMA path and serialize.

**Padding rows are free.** A batch is padded up to its captured shape with
whatever rows are already in the buffer; eval-mode BatchNorm uses running
statistics, so they cannot influence the real rows, and their legal-move count
is set to 1 so their (discarded) softmax is not all-NaN.

### The Python self-play engine (`batched_selfplay.py`)

Still here, still correct, and used automatically when the native core cannot be
built (and on demand via `--engine python`). It closes the CPU/GPU gap as far as
Python allows.

**Many games in flight.** Every in-flight game contributes at most one leaf per
simulation step, so a step is a single batched forward pass. Games share no
search state, so **no virtual loss is needed** and the search semantics match a
plain sequential PUCT search.

**Worker processes behind one inference server.** Tree search holds the GIL, so
it runs in worker *processes* that exchange positions with a single GPU-owning
server through shared memory. The server concatenates every worker's pending
request into one large forward pass, which matters a lot: a 128×10 net runs at
~56k positions/s at batch 512 and ~80k at batch 1536.

**A pipelined worker.** Each worker splits its games into `pipeline_stages`
sub-pools and keeps one request per sub-pool outstanding, so it keeps searching
while earlier requests are in flight instead of blocking on every round trip.

**A double-buffered server.** The server starts a batch's forward pass, then
finishes the *previous* batch — so while the GPU runs pass *N* the host is
staging pass *N+1* and scattering pass *N−1*.

**O(1) draw detection.** `board.is_game_over(claim_draw=True)` costs ~159 µs
because it replays the move stack looking for repetitions, and it used to run at
every node of every descent. Repetition counts and the halfmove clock are
tracked incrementally along the descent, and terminal status is resolved only at
a leaf, reusing the legal-move list the leaf needs anyway — 3.8 µs instead.

### What both engines do, per simulation

Both engines **reuse the subtree** under the played move rather than discarding
it each ply, **refill finished games** immediately so the batch never decays to
a handful of stragglers, and **resign** decided games (keeping
`resign_disable_fraction` of them going to measure the false-positive rate).

Per simulation, each active game descends its own tree by PUCT to a leaf;
terminal leaves are scored directly (checkmate `-1`, draw `0`, no network call);
non-terminal leaves are batch-encoded and evaluated in one `torch.no_grad`
forward pass (fp16 autocast on CUDA). Priors are soft-maxed over each leaf's
legal moves **on the device**, so only the gathered legal-move rows come back to
the host rather than the full `(B, 4672)` logit matrix. Children are expanded and
the value is backed up along the path, negating per ply. Root priors get
per-game Dirichlet noise. Once a move's simulation budget is spent (visits
carried over by subtree reuse count towards it) the game picks a move from its
root visit counts — **sampled** at temperature 1 for the first
`temperature_moves` plies, **argmax** thereafter — records the training example
and pushes the move. When a game ends, its stored examples get the final result
written back as the value target, signed for the mover at each position.

Under [playout-cap randomisation](#playout-cap-randomisation) the budget, the
root noise and the "record a training example" step are all conditional on the
ply having been drawn as a full search.

`self_play.py` retains the original single-game `play_game` (used by tests and
available for reference) with unchanged behavior.

### Efficient replay buffer

`self_play.ReplayBuffer` is a **numpy ring buffer**, stored compactly because
buffer size is one of the levers on final strength and a dense buffer runs out
of RAM long before it runs out of usefulness. States are **uint8-packed** (1.3 KB
per position instead of 5.4 KB) and policy targets are **sparse** — only the
at-most-80 moves that actually received visits, rather than a 4672-wide float32
row. That is ~1.9 KB per position against ~21 KB dense, so 500k positions cost
~0.9 GB rather than ~11.7 GB. Both are expanded on the GPU by the trainer, which
also keeps the host→device transfer ~11× smaller. It preallocates
`states (capacity, 21, 8, 8)`, `pol_idx/pol_val (capacity, 80)`, and
`values (capacity, 1)`, with a write cursor and size that wrap around. This
gives O(1) appends and random access and a flat memory footprint (no per-sample
Python objects). The API is `__init__(capacity)`, `append(SelfPlayBatch)`,
`__len__`, `sample(batch_size)` returning `(states, pol_idx, pol_val, values)`
sampled uniformly with replacement, and `state_dict()` / `load_state_dict()` so
a resumed run keeps its history (a buffer saved at a different `--buffer-size`
is truncated to the most recent positions that fit).

### The learning step (`train.py`)

Each iteration, after batched self-play appends fresh examples to the buffer, the
network trains with **Adam** on minibatches sampled from the ring buffer,
minimizing:

```
loss = policy_loss + value_loss
     = cross_entropy(policy_logits, MCTS_visit_policy)   # −Σ π·log softmax(logits)
     + MSE(value_pred, game_result)
```

plus L2 weight decay. The policy target is sparse, so the cross-entropy is
gathered at the visited moves rather than materializing a dense `(B, 4672)`
target; padded slots carry weight zero, which makes it exactly equal to the
dense form.

**How much to train.** The step count is sized from the **new** data: the
iteration draws `new_positions × --sample-reuse` samples, which works out to
each position being sampled `--sample-reuse` times over its life in the buffer
(it survives `buffer_size / new_positions` iterations, and over that span the
draws total `buffer_size × sample_reuse` across `buffer_size` positions). Sizing
it off the buffer instead — the old rule, one pass over the whole buffer per
epoch — made the work independent of how much data arrived: 7,812 steps an
iteration at a 2M buffer and batch 1024, whether that iteration produced 50k new
positions or 800k. `--train-steps` sets an explicit count, and
`--sample-reuse 0` restores the old behaviour.

GPU-efficiency details, all guarded to be no-ops on CPU/MPS:

- **AMP / fp16:** forward + loss run under `torch.autocast(..., float16)` on
  CUDA; a `GradScaler` handles the optimizer step. On CPU/MPS this is fp32.
- **cudnn.benchmark**, **channels-last** model memory format, **pinned** host
  tensors, and **non-blocking** host→device copies on CUDA.
- **Background prefetch:** minibatches are sampled and pinned on a worker
  thread, so the host-side gather overlaps the GPU step instead of preceding it.
  States cross the bus uint8-packed and policies sparse, and are expanded on the
  device.
- **No per-step synchronisation:** the running losses accumulate in a device
  tensor and are read once per epoch. Calling `.item()` every step would
  synchronise on every iteration and serialise the host against the GPU it is
  trying to keep fed.
- **Cosine LR decay** from `--lr` to `--lr-final` (default `lr*0.1`) across
  `--iterations`.
- **Gradient clipping** to `--grad-clip`.
- Optional **`torch.compile`** via `--compile` (CUDA + torch≥2).

**Checkpoints & resume.** Every iteration mirrors the latest weights to
`best.pt` and writes a resumable `train_state.pt` (model config + weights,
optimizer state, iteration, and numpy/torch RNG states) plus
`replay_buffer.npz` (unless `--no-save-buffer`); numbered
`checkpoint_{i:03d}.pt` files are written every `--checkpoint-every`. `--resume`
on a `train_state.pt` (or its directory) restores everything and continues from
the next iteration; on a plain model checkpoint it loads weights only and starts
a fresh optimizer.

### Evaluation (`evaluate.py`)

`evaluate.py` provides baseline opponents (`RandomOpponent`, `MaterialOpponent`),
a `play_match(...)` that alternates colors and reports `wins_a/draws/wins_b` and
`score_a`, an `estimate_elo_diff(score, games)` helper, and the high-level
`evaluate_model(...)` used by the `evaluate` CLI command. It can pit the model
against random/material baselines, another checkpoint (`model:PATH`), or an
external UCI engine (`uci:PATH`, optional). See [Evaluate strength](#2-evaluate-strength).

### The agent (`agent.py`)

`AlphaChessAgent` wraps a loaded model + MCTS for actual play: `play_move` picks
the most-visited move (no exploration noise), and `suggest_move` returns the best
move, its SAN/UCI, the value estimate, and the top candidates — this powers the
`suggest` command, the GUI's **H** hint, and the evaluated player in `evaluate`.

---

## Project layout

```
alpha_chess/
  encoding.py          board → (21,8,8) side-to-move-relative tensor; 4672-way move ↔ index
  network.py           AlphaZeroNet (residual policy/value net), device + save/load
  mcts.py              network-guided PUCT MCTS (interactive single-position search)
  self_play.py         reference single-game self-play + compact ReplayBuffer
  native/              the C search core (bitboards, movegen, encoding, PUCT tree)
    bitboard.h           attack tables + runtime-generated magic bitboards
    position.h           position, Zobrist hashing, make-move, legal move generation
    encode.h             21-plane encoding + 4672-way move index (matches encoding.py)
    mcts.h               arena-allocated PUCT trees and the self-play game driver
    fastchess.c          CPython bindings (buffer protocol, GIL released)
  native.py            builds the extension on demand and loads it (or falls back)
  native_selfplay.py   pooled self-play on the native core, one CUDA graph per shape
  batched_selfplay.py  pure-Python fallback: multi-process self-play behind one server
  train.py             the self-play + AMP training loop (resumable)
  evaluate.py          strength evaluation: baselines, matches, Elo, UCI opponent
  agent.py             high-level agent: play_move / suggest_move
  gui.py               pygame GUI (play + advisor board + board editor)
  cli.py               train / evaluate / suggest / play / analyze command line
docs/
  alpha_chess.typ      Typst source for the technical write-up
  alpha_chess.pdf      built PDF (committed; see docs/README.md to rebuild)
tests/
  test_encoding.py     round-trip, mirror-invariance and packing tests for the encoding
  test_selfplay.py     search/terminal-detection correctness and replay-buffer tests
  test_mcts.py         interactive search: terminal handling, mate-in-one, caching
  test_native.py       perft + native-vs-Python parity for movegen, indices and planes
  test_evaluate.py     Elo estimation and UCI strength limiting (clamping, skill level)
  test_gui.py          GUI undo/redo, last-move box, move sounds (headless)
requirements.txt
README.md
```

---

## Testing

```bash
.venv/bin/python -m pytest tests/ -q
```

The tests run on CPU in fp32 (all CUDA-only fast paths are guarded off).

`test_encoding.py` verifies constants and tensor shape/dtype, that **every**
legal move round-trips through `move_to_index` / `index_to_move` across
openings, random playouts, promotions, underpromotions, castling and en
passant, that a position and its colour mirror encode identically and map
corresponding moves to the same policy index, and that uint8 packing is lossless.

`test_selfplay.py` covers the search: that the O(1) terminal detection agrees
with python-chess's `claim_draw` semantics (checkmate, stalemate, insufficient
material, fifty-move, threefold) and that games the engine calls finished really
are over; that subtree reuse carries the played move's visits into the next ply;
that search finds a mate in one from uniform priors; the resignation rules and
their bookkeeping; the emitted batch's invariants; and the replay buffer's ring
wrap, oversize append and save/restore-at-a-different-capacity. It also runs the
multi-process pipeline end to end with two workers.

`test_mcts.py` covers the interactive search: that its leaf-resolved terminal
detection agrees with python-chess on checkmate, stalemate, insufficient
material, the fifty-move rule and threefold repetition (and never calls a
position finished that python-chess would not), that the search finds a mate in
one, leaves the caller's board untouched, returns a normalised distribution over
exactly the legal moves, and that a terminal leaf keeps its value on the node.

`test_native.py` pins the C core to the rules and to the Python encoding:
**perft** on the six standard positions (~194M nodes between them, covering
castling, en passant, promotion, discovered check, pins and stalemate), and a
direct diff of the native legal-move set, move-to-index mapping and encoded
planes against `python-chess` / `alpha_chess.encoding` over thousands of
positions drawn from random games — including en-passant positions, the one
feature the two encoders could plausibly disagree on. It then runs a short
native self-play to completion and checks the emitted training data (normalised
policy targets, in-range indices, valid packed states, results in `{-1, 0, +1}`)
and that playout-cap randomisation records exactly the full-search plies. The
whole file skips itself when no compiler is available, which is exactly when
training falls back to the Python engine.

`test_evaluate.py` covers the reporting: that the Elo estimate is zero at an
even score, is symmetric, and stays finite on a whitewash (tightening as the
match lengthens); and that UCI strength limiting does what it says — a request
below the engine's `UCI_Elo` floor is clamped **and warned about**, an in-range
request is applied untouched and silently, `Skill Level` reaches below that
floor, an out-of-range skill is clamped, the two throttles are refused
together, and the match label names the setting actually applied rather than
the one requested. The engine-backed tests skip when no UCI engine is
installed.

`test_gui.py` pins the undo/redo history, headless against an offscreen
surface: that redo restores exactly the plies undo took back (the human+agent
pair when playing the engine, a single ply in advisor and human-vs-human
modes), that it replays the agent's *recorded* reply rather than re-searching
— so redo is a true inverse even for a stochastic agent — and that the redo
history is discarded whenever the game moves onto a different line: a
different move played by hand, a new game, or a position adopted from the
editor. It also checks `R` is redo in play mode while still resetting the
editor in setup mode.

It also covers the last-move box and the move sounds: that the destination
square is boxed and the origin and untouched squares are not, that it is a
border rather than a fill (the square colour still shows through), that it is
legible on both square colours, that it follows undo/redo (it is read off the
move stack, so it cannot go stale) and is absent on a fresh board; and that
the synthesised clicks are int16 stereo,
decay to silence rather than ending in a pop, are identical every time, and
that captures and quiet moves get different ones. A headless app builds no
sound bank at all, so neither the tests nor an offscreen render touch an audio
device.

---

## FAQ / troubleshooting

- **"Self-play: Python engine ... (native core requested but unavailable)".**
  No C compiler was found. `pip install -r requirements.txt` includes `ziglang`,
  which is one and needs no root; `python -m alpha_chess.native` prints the
  build error. Training still works, at roughly a quarter of the speed.
- **Where did `--num-workers` go?** It still exists, but only the Python engine
  uses it. The native search needs about a third of one core to saturate a 3090,
  so worker processes would have nothing to do. `--pools` and
  `--games-in-flight` are the knobs that matter now.
- **My old checkpoint won't load: "trained on a 19-plane encoding".** The board
  encoding is now side-to-move relative with repetition planes (19 → 21 planes),
  so weights from before that change cannot be reused and `load_model` says so
  rather than failing obscurely. Train a fresh model, or check out the revision
  that produced the checkpoint.
- **The engine plays random/weak moves.** Expected for a small or briefly trained
  model — the value/policy are near-untrained. Train longer with more
  `--iterations`, `--games-per-iter`, and `--simulations`; measure progress with
  `evaluate`. Reaching 1500–2000 from scratch is a run of days on a 3090.
- **How do I speed up self-play?** With the native core the GPU is the
  bottleneck, so the lever is batch width: raise `--games-in-flight` (it *is*
  the network's batch size) and keep `--amp` on (default). `--pools` only needs
  to be 2–3 — enough to overlap one pool's tree search with the next pool's
  forward pass. Watch the per-iteration log: if mean batch is well under
  `--games-in-flight`, raise `--games-per-iter` so fewer iterations are spent
  in the drain tail; if "GPU busy" is low, widen the batch or shrink the net
  (`--channels` / `--blocks`). `--num-workers` is a *Python*-engine knob and
  does nothing here — see the two bullets above.
- **Is mixed precision automatic?** Yes. On CUDA, `--device auto` selects the GPU
  and fp16 AMP + cudnn.benchmark + channels-last activate automatically. On
  CPU/MPS everything runs fp32. Disable AMP with `--no-amp`.
- **How do I resume a run?** Point `--resume` at your output dir (or its
  `train_state.pt`) — model, optimizer, iteration, RNG **and the replay buffer**
  resume, and training continues from the next iteration. Pointing it at a plain
  checkpoint loads weights only.
- **What does the Elo number mean?** It is **relative to the chosen opponent**,
  not an absolute rating. Anchor a comparable number with a calibrated UCI engine
  (`uci:PATH` + `--uci-elo`) or online play.
- **UCI opponent errors.** The UCI path is optional and imported lazily; you need
  a real engine binary (e.g. Stockfish) at the path you pass — `FileNotFoundError:
  UCI engine not found` means that path is wrong, not that the engine is missing.
  Find it with `which stockfish || ls /usr/games/stockfish`: the Debian/Ubuntu
  package installs to `/usr/games/stockfish`, which is not on a non-root `PATH`,
  and Homebrew to `/opt/homebrew/bin/stockfish`. `--uci-elo` only takes effect if
  the engine advertises `UCI_LimitStrength`/`UCI_Elo`.
- **`--uci-elo 600` gave me a crushing loss — is my model below 600?** Probably
  not: engines refuse to go that low. Stockfish 16 advertises
  `UCI_Elo min 1320`, so anything below that is clamped to 1320 (now with a
  warning, and the match header names the applied setting). For an opponent
  weaker than that floor use `--uci-skill 0`. Below `material`-baseline
  strength, `random` and `material` are the informative anchors.
- **I hear no move sounds.** They need a working audio device; when the mixer
  cannot open one the GUI stays silent rather than failing, and the rest of the
  window works normally. Check **M** has not muted them (the panel lists it),
  and on WSL that WSLg's PulseAudio server is up (`pactl info`). No audio files
  are involved — the clicks are synthesised at startup — so there is nothing
  missing to reinstall.
- **GUI shows empty boxes instead of pieces.** The renderer auto-detects a font
  containing the Unicode chess glyphs (e.g. *Apple Symbols* on macOS,
  *DejaVu Sans* on Linux) and falls back to drawn lettered discs if none is
  found. This should "just work"; if not, ensure a symbol-capable system font is
  installed.
- **MPS / unsupported-op errors on Mac.** Run with `PYTORCH_ENABLE_MPS_FALLBACK=1`
  or use `--device cpu`.
- **No model found at `models/best.pt`.** Most commands default `--model` to
  `models/best.pt`. If it doesn't exist: `suggest` and `evaluate` exit with an
  error, `play` falls back to human-vs-human, and `analyze` opens the advisor
  board with suggestions disabled (you can still move both sides). Train first
  (creates `models/best.pt`) or pass an explicit `--model path/to/checkpoint.pt`.
  Paths are relative to your current directory, so run from the repo root.
