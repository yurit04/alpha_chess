# alpha_chess

An **AlphaZero-style, self-play chess engine** built with PyTorch and
[python-chess](https://python-chess.readthedocs.io/). It learns entirely from
self-play — no human games, no opening book, no handcrafted evaluation — using a
single residual **policy + value** neural network guided by **PUCT Monte-Carlo
Tree Search (MCTS)**. It ships with a pygame GUI so you can play against the
trained agent or ask it for the best move in any position, plus a
**multi-process self-play pipeline** — worker processes running the tree search
behind a single GPU-owning inference server — and an `evaluate` command to track
playing strength as you train.

The defaults target a **single NVIDIA RTX 3090 (24 GB)** and a practical goal of
reaching roughly **1500–2000 Elo** with a small-but-capable network. Everything
also runs correctly on CPU (and Apple-Silicon MPS) for development and testing —
CUDA-only fast paths (fp16 autocast, GradScaler, cudnn.benchmark, channels-last,
pinned memory) are all guarded and simply fall back to fp32.

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

Dependencies are only **numpy**, **torch**, **python-chess**, and **pygame** —
no extra packages. `python-chess` already provides `chess.engine`, used for the
optional UCI-engine opponent in `evaluate`.

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
  --iterations 3 --games-per-iter 8 --simulations 40 \
  --num-workers 2 --games-in-flight 4 --batch-size 64 --buffer-size 5000 \
  --channels 32 --blocks 4 --out models --seed 0

# 2) See how it does against a trivial baseline
.venv/bin/python -m alpha_chess.cli evaluate \
  --model models/best.pt --opponent random --games 20 --simulations 40

# 3) Play against it in a window (H = hint, U = undo, N = new game, F = flip)
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

**1 — Start training from scratch** (`--device auto` picks CUDA; AMP/fp16,
cudnn.benchmark and channels-last activate automatically; periodic self-eval
every 5 iterations):

```bash
.venv/bin/python -m alpha_chess.cli train \
  --device auto \
  --iterations 80 \
  --games-per-iter 4000 \
  --num-workers 12 --games-in-flight 128 \
  --simulations 200 \
  --epochs 4 --batch-size 1024 \
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
  --games-per-iter 4000 --num-workers 12 --games-in-flight 128 \
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

**4 — Anchor to a REAL Elo** with a calibrated engine (install Stockfish; adjust
the path). Bracket the level by trying a few `--uci-elo` caps — the target is a
~50% score against ~1500–2000:

```bash
.venv/bin/python -m alpha_chess.cli evaluate --model models/best.pt \
  --opponent uci:/opt/homebrew/bin/stockfish --uci-elo 1500 \
  --games 60 --simulations 800

.venv/bin/python -m alpha_chess.cli evaluate --model models/best.pt \
  --opponent uci:/opt/homebrew/bin/stockfish --uci-elo 2000 \
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
> progress. `--num-workers 12 --games-in-flight 128` fills a 3090 paired with an
> 8-core CPU; lower `--games-in-flight` if you hit RAM limits.

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
  --games-per-iter 2000 \
  --simulations 200 \
  --num-workers 12 \
  --games-in-flight 128 \
  --epochs 4 \
  --batch-size 1024 \
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
| `--iterations` | `40` | Number of self-play → train cycles. Also the length of the cosine LR schedule. |
| `--games-per-iter` | `2000` | Self-play games generated each iteration. |
| `--simulations` | `200` | MCTS simulations per move during self-play. Higher = stronger data, slower. |
| `--num-workers` | CPU count − 2 | Self-play **search processes**. Tree search is pure Python and GIL-bound, so this is the primary throughput lever. |
| `--games-in-flight` | `128` | Concurrent games searched **per worker**. The network sees up to `num-workers × games-in-flight` positions per forward pass. |
| `--resign-threshold` | `-0.90` | Resign once the mover's best root value stays at or below this for two plies. `--no-resign` plays every game out. |
| `--resign-disable-fraction` | `0.10` | Fraction of games played out with resignation suppressed, to measure the resign false-positive rate (printed each iteration). |
| `--no-save-buffer` | off | Skip persisting the replay buffer (it is otherwise written next to `train_state.pt` so resumes keep their history). |
| `--epochs` | `4` | Passes over the replay buffer per iteration during the learning phase. |
| `--batch-size` | `1024` | Minibatch size for the gradient step. |
| `--lr` | `1e-3` | Adam learning rate (initial value, before cosine decay). |
| `--lr-final` | `None` → `lr*0.1` | Final learning rate for cosine decay across `--iterations`. |
| `--grad-clip` | `1.0` | Max gradient norm (`clip_grad_norm_`). |
| `--channels` | `128` | Width of the residual tower (conv channels). Bigger = stronger, slower. |
| `--blocks` | `10` | Number of residual blocks (depth). Bigger = stronger, slower. |
| `--buffer-size` | `2000000` | Replay buffer capacity in **positions** (numpy ring buffer; oldest evicted). Positions cost ~1.9 KB each, so 2M is ~3.8 GB. |
| `--temperature-moves` | `30` | For the first N plies, moves are **sampled** from MCTS visit counts (exploration); after that the **best** move is played. |
| `--max-moves` | `400` | Cap on plies per self-play game (cutoff scored as a draw). |
| `--weight-decay` | `1e-4` | L2 regularization for Adam. |
| `--amp` / `--no-amp` | AMP **on** | Enable/disable fp16 automatic mixed precision. Active only on CUDA; a no-op (fp32) on CPU/MPS. |
| `--compile` | off | Wrap the model with `torch.compile` (CUDA + torch≥2 only; guarded). |
| `--eval-every` | `0` | Evaluate every N iterations (0 disables). Requires `--eval-games > 0`. |
| `--eval-games` | `0` | Games per periodic in-training evaluation (0 disables). |
| `--checkpoint-every` | `1` | Save a numbered checkpoint every N iterations (`best.pt` + `train_state.pt` are always written). |
| `--resume` | `None` | Path to a `train_state.pt` (or a dir containing one) to **continue**, or a plain model checkpoint to load **weights only** and start a fresh optimizer. |
| `--seed` | `None` | Seeds `torch`/`numpy`/`random` for reproducibility. |
| `--device` | `auto` | `auto` \| `cpu` \| `cuda` \| `mps`. See [Devices](#devices-cpu--gpu--apple-silicon). |

Each iteration prints mean policy/value losses plus **throughput**: self-play
games/sec and positions/sec, and training steps/sec.

### Recommended RTX 3090 recipe

On a 3090, `--device auto` selects **CUDA** automatically, and — because
`--amp` is on by default — fp16 mixed precision, `cudnn.benchmark`, and
channels-last memory format all activate automatically. You do not need any
extra flags to turn the GPU on.

```bash
.venv/bin/python -m alpha_chess.cli train \
  --device auto \
  --iterations 80 \
  --games-per-iter 4000 \
  --simulations 200 \
  --num-workers 12 --games-in-flight 128 \
  --epochs 4 \
  --batch-size 1024 \
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

- **`--num-workers`** is the biggest throughput lever. MCTS is pure-Python tree
  search and holds the GIL, so it is spread over worker *processes* that feed a
  single GPU-owning inference server. Use roughly your core count; there is
  little to gain past ~1.5× the number of physical cores.
- **`--games-in-flight`** sets how many games each worker searches at once, and
  so how wide the batched forward pass gets (`num-workers × games-in-flight`
  positions). Wider batches are much more efficient on the GPU — a 128×10 net
  runs at ~56k positions/s at batch 512 against ~80k at batch 1536 — but large
  pools cost RAM and hurt cache locality, so **128** is a good middle. Neither
  flag changes *what* is learned, only how fast games are generated.
- **`--simulations`** trades data quality against speed; 200 is a solid default.
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
  --iterations 60 --channels 128 --blocks 10 --games-in-flight 128
```

This restores the model, optimizer, iteration counter, and RNG states and
**continues from the next iteration**. If you instead point `--resume` at a
plain model checkpoint (e.g. `checkpoint_012.pt`), it loads the **weights only**
and starts a fresh optimizer — useful for fine-tuning. Missing optimizer state
never crashes the run.

### Training cost & expectations

Be realistic: reaching **1500–2000 Elo from scratch** on one 3090 takes
**substantial wall-clock — realistically many hours to days** — and depends
heavily on how much self-play you generate (games × moves × simulations).

Self-play is ultimately limited by **CPU-bound legal-move generation and board
logic in python-chess** (running the MCTS tree), which the GPU does not
accelerate; the pipeline exists to spread that work over every core and keep the
GPU fed while it runs. On an RTX 3090 with an 8-core/16-thread i9 it sustains
roughly **32,000–36,000 evaluated positions/s**. At 200 simulations per move
that is on the order of **5,000–6,000 self-play games/hour**, so a few hundred
thousand games — the rough order needed for the 1500–2000 band — is a run of
**days, not months**.

Where the time goes, measured on that machine with a 128×10 net:

| | positions/s |
|---|---|
| Pure tree search, all workers, network stubbed out | ~80,000 |
| Network forward pass at batch 1536 | ~80,000 |
| Network forward pass at batch 512 | ~56,000 |
| **End-to-end self-play** | **~35,000** |

Both halves are within ~2× of the achieved rate, so the two are reasonably
balanced: adding GPU without adding cores (or vice versa) buys little. The
iteration log prints mean batch size and GPU-busy percentage so you can see
which side is short on your hardware. Plan for a long run, checkpoint often, and
track strength with `evaluate`. This project is a faithful, runnable
*implementation* of the AlphaZero method tuned for a single GPU — not a promise
of grandmaster strength on a laptop.

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
to that opponent. Colors alternate across games for fairness.

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
  --model models/best.pt --opponent uci:/usr/local/bin/stockfish \
  --uci-elo 1600 --games 40 --simulations 400
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | `models/best.pt` | Checkpoint to evaluate (played as player A). Exits with an error if missing. |
| `--opponent` | `material` | `random` \| `material` \| `model:PATH` \| `uci:PATH`. |
| `--games` | `40` | Number of games (colors alternate across games). |
| `--simulations` | `200` | MCTS simulations per move for the evaluated model. |
| `--uci-elo` | `None` | If the UCI opponent supports `UCI_LimitStrength`/`UCI_Elo`, cap it to this Elo. |
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
- **N** — new game.
- **F** — flip the board.
- **E** — enter the **board editor** (see below).
- **ESC / Q** — quit.

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
- **E** — open the **board editor** (below) to set up a mid-game position when
  you join a game already in progress.
- **F** — flip the board.  **N** — new game (standard start position).
- **Q / ESC** — quit.

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
sample) the move actually played. `mcts.py` powers **interactive** single-position
search for the GUI, `suggest`, and `evaluate`.

### Batched self-play (`batched_selfplay.py`)

Self-play is the wall-clock bottleneck. The tree search is pure Python and the
network is on the GPU, so the naive arrangement leaves the GPU almost completely
idle: one process descends a tree on one core while the device waits. Six things
close that gap.

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

**CUDA-graph replay.** Inference here is *host-launch bound*, not GPU bound: at
batch 384 a 128×10 net needs ~3.2 ms of host time to issue ~3.8 ms of device
work, and the server competes with the search workers for cores. Capturing the
forward pass into a CUDA graph drops the issue cost to ~0.03 ms. Graphs are
captured fresh each self-play phase, so a graph can never serve stale weights,
and batch sizes are rounded up to a multiple of 128 to keep the number of
captured shapes (and cuDNN's own algorithm cache) small.

**O(1) draw detection.** `board.is_game_over(claim_draw=True)` costs ~159 µs
because it replays the move stack looking for repetitions, and it used to run at
every node of every descent. Repetition counts and the halfmove clock are now
tracked incrementally along the descent, and terminal status is resolved only at
a leaf, reusing the legal-move list the leaf needs anyway — 3.8 µs instead.

On top of that the search **reuses the subtree** under the played move rather
than discarding it each ply, **refills finished games** immediately so the batch
never decays to a handful of stragglers, and **resigns** decided games (keeping
`resign_disable_fraction` of them going to measure the false-positive rate).

Per simulation, each active game descends its own tree by PUCT to a leaf;
terminal leaves are scored directly (checkmate `-1`, draw `0`, no network call);
non-terminal leaves are batch-encoded and evaluated in one `torch.no_grad`
forward pass (fp16 autocast on CUDA). Priors are soft-maxed over each leaf's
legal moves **on the device**, so only the `(B, max_legal)` gathered rows come
back to the host rather than the full `(B, 4672)` logit matrix. Children are
expanded and the value is backed up along the path, negating per ply. Root
priors get per-game Dirichlet noise. Once a move's simulation budget is spent
(visits carried over by subtree reuse count towards it) the game picks a move
from its root visit counts — **sampled** at temperature 1 for the first
`temperature_moves` plies, **argmax** thereafter — records the training example
and pushes the move. When a game ends, its stored examples get the final result
written back as the value target, signed for the mover at each position.

`generate_selfplay_data(model, device, num_games, **kwargs)` is the entry point;
`num_workers=1` runs everything in-process, which is what CPU-only machines and
the tests use.

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
`values (capacity, 1)` as float32, with a write cursor and size that wrap
around. This gives O(1) random access and a flat memory footprint (no per-sample
Python objects). The public API is unchanged: `__init__(capacity)`,
`append(list_of_examples)`, `__len__`, and `sample(batch_size)` returning
`(states, policies, values)` numpy arrays sampled uniformly with replacement.

### The learning step (`train.py`)

Each iteration, after batched self-play appends fresh examples to the buffer, the
network trains with **Adam** on minibatches sampled from the ring buffer,
minimizing:

```
loss = policy_loss + value_loss
     = cross_entropy(policy_logits, MCTS_visit_policy)   # −Σ π·log softmax(logits)
     + MSE(value_pred, game_result)
```

plus L2 weight decay. GPU-efficiency details, all guarded to be no-ops on
CPU/MPS:

- **AMP / fp16:** forward + loss run under `torch.autocast(..., float16)` on
  CUDA; a `GradScaler` handles the optimizer step. On CPU/MPS this is fp32.
- **cudnn.benchmark**, **channels-last** model memory format, **pinned** host
  tensors, and **non-blocking** host→device copies on CUDA.
- **Cosine LR decay** from `--lr` to `--lr-final` (default `lr*0.1`) across
  `--iterations`.
- **Gradient clipping** to `--grad-clip`.
- Optional **`torch.compile`** via `--compile` (CUDA + torch≥2).

**Checkpoints & resume.** Every iteration mirrors the latest weights to
`best.pt` and writes a resumable `train_state.pt` (model config + weights,
optimizer state, iteration, and numpy/torch RNG states); numbered
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
  batched_selfplay.py  multi-process self-play behind one batched GPU inference server
  train.py             the batched self-play + AMP training loop (resumable)
  evaluate.py          strength evaluation: baselines, matches, Elo, UCI opponent
  agent.py             high-level agent: play_move / suggest_move
  gui.py               pygame GUI (play + advisor board + board editor)
  cli.py               train / evaluate / suggest / play / analyze command line
tests/
  test_encoding.py     round-trip, mirror-invariance and packing tests for the encoding
  test_selfplay.py     search/terminal-detection correctness and replay-buffer tests
requirements.txt
README.md
```

---

## Testing

```bash
.venv/bin/python -m pytest tests/ -q
```

The tests run on CPU in fp32 (all CUDA-only fast paths are guarded off). The
encoding tests verify constants, tensor shape/dtype, and that **every** legal
move round-trips through `move_to_index` / `index_to_move` across openings,
random playouts, promotions, underpromotions, castling, and en passant.

---

## FAQ / troubleshooting

- **The engine plays random/weak moves.** Expected for a small or briefly trained
  model — the value/policy are near-untrained. Train longer with more
  `--iterations`, `--games-per-iter`, and `--simulations`; measure progress with
  `evaluate`. Reaching 1500–2000 from scratch takes many hours to days on a 3090.
- **How do I speed up self-play?** Raise `--num-workers` toward your core count
  — self-play is **CPU-bound** in python-chess, so cores matter more than the
  GPU. Then raise `--games-in-flight` to widen the batched forward pass, and
  keep `--amp` on (default). Watch the per-iteration log: if "GPU busy" is low,
  add workers; if mean batch is small, raise `--games-in-flight`.
- **Is mixed precision automatic?** Yes. On CUDA, `--device auto` selects the GPU
  and fp16 AMP + cudnn.benchmark + channels-last activate automatically. On
  CPU/MPS everything runs fp32. Disable AMP with `--no-amp`.
- **How do I resume a run?** Point `--resume` at your output dir (or its
  `train_state.pt`) — model, optimizer, iteration, and RNG resume and training
  continues from the next iteration. Pointing it at a plain checkpoint loads
  weights only.
- **What does the Elo number mean?** It is **relative to the chosen opponent**,
  not an absolute rating. Anchor a comparable number with a calibrated UCI engine
  (`uci:PATH` + `--uci-elo`) or online play.
- **UCI opponent errors.** The UCI path is optional and imported lazily; you need
  a real engine binary (e.g. Stockfish) at the path you pass. `--uci-elo` only
  takes effect if the engine advertises `UCI_LimitStrength`/`UCI_Elo`.
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
```
