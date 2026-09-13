"""Command-line interface for alpha_chess.

Exposes an argparse-based CLI with these subcommands:

* ``train``    -- run the from-scratch self-play training loop.
* ``suggest``  -- load a trained model and print the best move for a position.
* ``play``     -- launch the pygame GUI to play against the agent.
* ``analyze``  -- launch the ADVISOR board: you make every move for both
  colours (mirroring a game played elsewhere) and ask the engine for the best
  move on demand; the engine never moves on its own.
* ``evaluate`` -- play a trained model against a baseline / another model /
  a UCI engine and report W/D/L, score, and an estimated Elo difference.

Both ``python -m alpha_chess.cli`` and ``python -m alpha_chess`` route here via
:func:`main`.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import chess

# Standard chess starting position, used as the default FEN for ``suggest``.
START_FEN = chess.STARTING_FEN


def _add_train_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``train`` subcommand and its flags."""
    p = subparsers.add_parser(
        "train",
        help="Run the self-play training loop from scratch (or resume).",
    )
    p.add_argument("--iterations", type=int, default=40,
                   help="Number of self-play/training iterations.")
    p.add_argument("--games-per-iter", type=int, default=4000,
                   help="Self-play games generated per iteration.")
    p.add_argument("--simulations", type=int, default=200,
                   help="MCTS simulations per move during self-play.")
    p.add_argument("--engine", choices=("auto", "native", "python"),
                   default="auto",
                   help="Self-play engine: the native C core (~4x faster, "
                        "built on demand) or the pure-Python fallback. "
                        "'auto' uses native whenever it can be built.")
    p.add_argument("--pools", type=int, default=None,
                   help="Independent native game pools. One pool's tree "
                        "search overlaps the next pool's forward pass "
                        "(default 3). Native engine only.")
    p.add_argument("--num-workers", type=int, default=None,
                   help="Self-play search processes for the PYTHON engine. "
                        "Its tree search is GIL-bound, so this is that "
                        "engine's main throughput lever (default: three "
                        "quarters of the CPU count, capped at 24). Unused by "
                        "the native engine.")
    p.add_argument("--games-in-flight", type=int, default=1024,
                   help="Concurrent games searched per pool (native) or per "
                        "worker (Python). The network batch is that many "
                        "positions per pool/worker. The default suits the "
                        "native engine; with --engine python use ~128, since "
                        "each worker stages its own buffer.")
    p.add_argument("--pipeline-stages", type=int, default=None,
                   help="Sub-pools per worker, so several inference requests "
                        "per worker are in flight at once (default 4). "
                        "Python engine only.")
    p.add_argument("--fast-simulations", type=int, default=50,
                   help="Simulation budget for plies that playout-cap "
                        "randomisation does not search fully (native engine "
                        "only).")
    p.add_argument("--full-search-prob", type=float, default=0.25,
                   help="Probability that a ply gets the full --simulations "
                        "budget, root noise and a recorded training target. "
                        "Lower spends the budget on more games instead of "
                        "deeper searches (native engine only).")
    p.add_argument("--no-playout-cap", dest="full_search_prob",
                   action="store_const", const=1.0,
                   help="Search every ply fully (classic AlphaZero); "
                        "equivalent to --full-search-prob 1.0.")
    p.add_argument("--dirichlet-epsilon", type=float, default=0.25,
                   help="Weight of the root Dirichlet noise in the priors. "
                        "Exploration competes with the policy's own "
                        "confidence, so a policy that has sharpened needs "
                        "MORE than one that has not.")
    p.add_argument("--dirichlet-alpha", type=float, default=0.3,
                   help="Shape of the root Dirichlet noise (lower = spikier, "
                        "concentrating the noise on fewer moves).")
    p.add_argument("--noise-all-plies", dest="noise_all_plies",
                   action="store_true", default=False,
                   help="Apply root noise on EVERY ply, not only on the plies "
                        "that also record a training target. With "
                        "--full-search-prob 0.25 the default leaves three "
                        "quarters of the moves actually played with no "
                        "exploration noise. Native engine only.")
    p.add_argument("--fpu-reduction", type=float, default=0.0,
                   help="First-play-urgency penalty for unvisited children. "
                        "0 treats them as drawn (AlphaZero); 0.2-0.3 makes "
                        "the search commit sooner (native engine only).")
    p.add_argument("--resign-threshold", type=float, default=-0.90,
                   help="Resign once the mover's best root value stays at or "
                        "below this for two plies (raises games/hour by "
                        "cutting off decided games).")
    p.add_argument("--no-resign", dest="resign", action="store_false",
                   default=True,
                   help="Play every self-play game to the end.")
    p.add_argument("--resign-disable-fraction", type=float, default=0.10,
                   help="Fraction of games played out with resignation "
                        "suppressed, to measure resign false positives.")
    p.add_argument("--no-save-buffer", dest="save_buffer",
                   action="store_false", default=True,
                   help="Do not persist the replay buffer for resuming.")
    p.add_argument("--epochs", type=int, default=4,
                   help="Training epochs per iteration.")
    p.add_argument("--sample-reuse", type=float, default=4.0,
                   help="Expected number of times each position is sampled "
                        "over its lifetime in the replay buffer; this sets the "
                        "gradient-step count, which then scales with the data "
                        "rather than with the buffer. 0 restores the old rule "
                        "(one pass over the whole replay buffer per epoch).")
    p.add_argument("--train-steps", type=int, default=0,
                   help="Explicit gradient steps per iteration "
                        "(0 = derive from --sample-reuse).")
    p.add_argument("--batch-size", type=int, default=1024,
                   help="Minibatch size for optimization.")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Adam learning rate (initial, before cosine decay).")
    p.add_argument("--lr-final", type=float, default=None,
                   help="Final learning rate for cosine decay "
                        "(default: lr * 0.1).")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Max gradient norm for clipping.")
    p.add_argument("--out", type=str, default="models",
                   help="Output directory for checkpoints and best.pt.")
    p.add_argument("--channels", type=int, default=128,
                   help="Residual block channel width.")
    p.add_argument("--blocks", type=int, default=10,
                   help="Number of residual blocks.")
    p.add_argument("--buffer-size", type=int, default=2000000,
                   help="Replay buffer capacity.")
    p.add_argument("--temperature-moves", type=int, default=30,
                   help="Plies for which moves are sampled at temperature 1.")
    p.add_argument("--max-moves", type=int, default=400,
                   help="Maximum plies before a game is cut off as a draw.")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="Adam weight decay (L2 regularization).")
    # AMP (fp16) is on by default; --no-amp disables it. On CPU/MPS the
    # training loop keeps fp32 regardless (autocast is guarded to CUDA).
    p.add_argument("--amp", dest="amp", action="store_true", default=True,
                   help="Enable automatic mixed precision on CUDA (default).")
    p.add_argument("--no-amp", dest="amp", action="store_false",
                   help="Disable automatic mixed precision (force fp32).")
    p.add_argument("--compile", dest="compile", action="store_true",
                   default=False,
                   help="Compile the model with torch.compile (CUDA + "
                        "torch>=2 only; default off).")
    p.add_argument("--eval-every", type=int, default=0,
                   help="Evaluate every N iterations (0 disables).")
    p.add_argument("--eval-games", type=int, default=0,
                   help="Games per periodic evaluation (0 disables).")
    p.add_argument("--checkpoint-every", type=int, default=1,
                   help="Save a numbered checkpoint every N iterations.")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a train_state.pt (or a directory containing "
                        "one) to resume training, or a plain model checkpoint "
                        "to load weights only.")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility.")
    p.add_argument("--device", type=str, default="auto",
                   help="Device preference: auto|cpu|cuda|mps.")


def _add_suggest_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``suggest`` subcommand and its flags."""
    p = subparsers.add_parser(
        "suggest",
        help="Print the best move and top candidates for a position.",
    )
    p.add_argument("--fen", type=str, default=START_FEN,
                   help="FEN of the position to analyze.")
    p.add_argument("--model", type=str, default="models/best.pt",
                   help="Path to the trained model checkpoint.")
    p.add_argument("--simulations", type=int, default=200,
                   help="MCTS simulations to run.")
    p.add_argument("--device", type=str, default="auto",
                   help="Device preference: auto|cpu|cuda|mps.")


def _add_play_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``play`` subcommand and its flags."""
    p = subparsers.add_parser(
        "play",
        help="Launch the pygame GUI to play against the agent.",
    )
    p.add_argument("--model", type=str, default="models/best.pt",
                   help="Path to the trained model (optional; enables agent).")
    p.add_argument("--simulations", type=int, default=200,
                   help="MCTS simulations per agent move.")
    p.add_argument("--color", type=str, default="white",
                   help="Human color: white|black.")
    p.add_argument("--device", type=str, default="auto",
                   help="Device preference: auto|cpu|cuda|mps.")
    p.add_argument("--setup", action="store_true",
                   help="Start in the board-editor / analysis (setup) mode.")


def _add_analyze_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``analyze`` subcommand (the ADVISOR board)."""
    p = subparsers.add_parser(
        "analyze",
        help="Open the ADVISOR board: you move both sides, ask the engine for "
             "the best move on demand (it never moves on its own).",
    )
    p.add_argument("--model", type=str, default="models/best.pt",
                   help="Path to the trained model (optional; enables suggestions).")
    p.add_argument("--simulations", type=int, default=400,
                   help="MCTS simulations per suggestion.")
    p.add_argument("--fen", type=str, default=None,
                   help="Optional FEN for the STARTING position to follow from.")
    p.add_argument("--setup", action="store_true",
                   help="Start in the board editor first (to join a game in "
                        "progress); applying the position lands on the advisor board.")
    p.add_argument("--device", type=str, default="auto",
                   help="Device preference: auto|cpu|cuda|mps.")


def _add_evaluate_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``evaluate`` subcommand and its flags."""
    p = subparsers.add_parser(
        "evaluate",
        help="Play a trained model against a baseline / another model / a UCI "
             "engine and report W/D/L, score, and estimated Elo difference.",
    )
    p.add_argument("--model", type=str, default="models/best.pt",
                   help="Path to the trained model checkpoint to evaluate.")
    p.add_argument("--opponent", type=str, default="material",
                   help="Opponent spec: random | material | model:PATH | "
                        "uci:PATH.")
    p.add_argument("--games", type=int, default=40,
                   help="Number of games to play.")
    p.add_argument("--simulations", type=int, default=200,
                   help="MCTS simulations per model move.")
    p.add_argument("--uci-elo", type=int, default=None,
                   help="If the opponent is a UCI engine that supports "
                        "UCI_LimitStrength/UCI_Elo, cap it to this Elo. "
                        "Engines enforce a floor (1320 on Stockfish 16) and a "
                        "lower request is clamped with a warning; use "
                        "--uci-skill for weaker opponents.")
    p.add_argument("--uci-skill", type=int, default=None,
                   help="If the opponent is a UCI engine with a Skill Level "
                        "option, set it (0-20 on Stockfish, 0 being much "
                        "weaker than any --uci-elo can reach). Mutually "
                        "exclusive with --uci-elo.")
    p.add_argument("--max-moves", type=int, default=300,
                   help="Maximum plies before a game is scored a draw.")
    p.add_argument("--opening-plies", type=int, default=4,
                   help="Random opening plies per game/pair to decorrelate "
                        "games between deterministic players (0 = start "
                        "position every game).")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility.")
    p.add_argument("--device", type=str, default="auto",
                   help="Device preference: auto|cpu|cuda|mps.")


def _run_train(args: argparse.Namespace) -> None:
    """Dispatch the ``train`` subcommand to :func:`alpha_chess.train.train`."""
    from alpha_chess.train import train
    from alpha_chess.network import get_device

    device = get_device(args.device)
    best_path = train(
        iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        simulations=args.simulations,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_dir=args.out,
        channels=args.channels,
        num_blocks=args.blocks,
        buffer_size=args.buffer_size,
        temperature_moves=args.temperature_moves,
        max_moves=args.max_moves,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        games_in_flight=args.games_in_flight,
        pipeline_stages=args.pipeline_stages,
        engine=args.engine,
        pools=args.pools,
        fast_simulations=args.fast_simulations,
        full_search_prob=args.full_search_prob,
        fpu_reduction=args.fpu_reduction,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        noise_all_plies=args.noise_all_plies,
        sample_reuse=args.sample_reuse,
        train_steps=args.train_steps,
        resign_threshold=args.resign_threshold if args.resign else None,
        resign_disable_fraction=args.resign_disable_fraction,
        save_buffer=args.save_buffer,
        lr_final=args.lr_final,
        grad_clip=args.grad_clip,
        use_amp=args.amp,
        compile_model=args.compile,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        device=device,
        seed=args.seed,
    )
    print("Training complete. Best model saved to: {}".format(best_path))


def _run_suggest(args: argparse.Namespace) -> None:
    """Dispatch the ``suggest`` subcommand and pretty-print the result."""
    from alpha_chess.agent import AlphaChessAgent
    from alpha_chess.network import get_device

    if not os.path.isfile(args.model):
        print("Error: model file not found: {}".format(args.model),
              file=sys.stderr)
        sys.exit(1)

    board = chess.Board(args.fen)
    agent = AlphaChessAgent(
        model_path=args.model,
        device=get_device(args.device),
        simulations=args.simulations,
    )
    result = agent.suggest_move(board)

    if result.get("move") is None:
        print("Game is over; no move to suggest.")
        print("Result: {}".format(board.result()))
        return

    print("Best move: {} ({})".format(result["san"], result["uci"]))
    print("Eval (value, side-to-move perspective): {:+.4f}".format(
        result["value"]))
    print("Top moves:")
    for san, prob in result["top_moves"]:
        print("  {:<8} {:6.2%}".format(san, prob))


def _run_play(args: argparse.Namespace) -> None:
    """Dispatch the ``play`` subcommand to :func:`alpha_chess.gui.launch_gui`."""
    from alpha_chess.gui import launch_gui
    from alpha_chess.network import get_device

    model_path = args.model if args.model and os.path.isfile(args.model) else None
    if args.model and model_path is None:
        print("Warning: model '{}' not found; running human-vs-human."
              .format(args.model), file=sys.stderr)

    launch_gui(
        model_path=model_path,
        simulations=args.simulations,
        human_color=args.color,
        device=get_device(args.device),
        start_in_setup=args.setup,
    )


def _run_analyze(args: argparse.Namespace) -> None:
    """Dispatch the ``analyze`` subcommand: open the ADVISOR board."""
    from alpha_chess.gui import launch_gui
    from alpha_chess.network import get_device

    model_path = args.model if args.model and os.path.isfile(args.model) else None
    if args.model and model_path is None:
        print("Warning: model '{}' not found; suggestions disabled until a "
              "model is loaded.".format(args.model), file=sys.stderr)

    launch_gui(
        model_path=model_path,
        simulations=args.simulations,
        device=get_device(args.device),
        advisor=True,
        start_in_setup=args.setup,
        initial_fen=args.fen,
    )


def _run_evaluate(args: argparse.Namespace) -> None:
    """Dispatch ``evaluate`` to :func:`alpha_chess.evaluate.evaluate_model`."""
    from alpha_chess.evaluate import evaluate_model
    from alpha_chess.network import get_device

    if not os.path.isfile(args.model):
        print("Error: model file not found: {}".format(args.model),
              file=sys.stderr)
        sys.exit(1)

    # An engine limiting strength by Elo ignores its skill level, so refuse the
    # combination rather than quietly honouring one of them.
    if args.uci_elo is not None and args.uci_skill is not None:
        print("Error: pass only one of --uci-elo and --uci-skill.",
              file=sys.stderr)
        sys.exit(1)

    result = evaluate_model(
        model_path=args.model,
        opponent_spec=args.opponent,
        games=args.games,
        simulations=args.simulations,
        device=get_device(args.device),
        max_moves=args.max_moves,
        seed=args.seed,
        uci_elo=args.uci_elo,
        uci_skill=args.uci_skill,
        opening_random_plies=args.opening_plies,
    )

    wins = result["wins"]
    draws = result["draws"]
    losses = result["losses"]
    score = result["score"]
    elo_diff = result["elo_diff"]
    opponent = result["opponent"]

    print("Evaluation vs {}:".format(opponent))
    print("  Games : {}".format(args.games))
    print("  W/D/L : {} / {} / {}".format(wins, draws, losses))
    print("  Score : {:.3f}".format(score))
    print("  Elo   : {:+.0f} (relative to opponent)".format(elo_diff))
    print("Note: Elo is RELATIVE to this opponent, not an absolute rating; "
          "anchor to real Elo only via a calibrated UCI engine.")


def main(argv: Optional[List[str]] = None) -> None:
    """Parse arguments and dispatch to the selected subcommand.

    Parameters
    ----------
    argv:
        Optional list of argument strings (defaults to ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(
        prog="alpha_chess",
        description="AlphaZero-style chess engine: train, suggest, and play.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_train_parser(subparsers)
    _add_suggest_parser(subparsers)
    _add_play_parser(subparsers)
    _add_analyze_parser(subparsers)
    _add_evaluate_parser(subparsers)

    args = parser.parse_args(argv)

    if args.command == "train":
        _run_train(args)
    elif args.command == "suggest":
        _run_suggest(args)
    elif args.command == "play":
        _run_play(args)
    elif args.command == "analyze":
        _run_analyze(args)
    elif args.command == "evaluate":
        _run_evaluate(args)
    else:  # pragma: no cover - argparse enforces a valid command.
        parser.error("Unknown command: {}".format(args.command))


if __name__ == "__main__":
    main()
