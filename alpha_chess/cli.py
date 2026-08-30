"""Command-line interface for alpha_chess.

Exposes an argparse-based CLI with these subcommands:

* ``train``   -- run the from-scratch self-play training loop.
* ``suggest`` -- load a trained model and print the best move for a position.
* ``play``    -- launch the pygame GUI to play against the agent.
* ``analyze`` -- launch the GUI directly in the board-editor / analysis mode.

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
    p.add_argument("--iterations", type=int, default=10,
                   help="Number of self-play/training iterations.")
    p.add_argument("--games-per-iter", type=int, default=20,
                   help="Self-play games generated per iteration.")
    p.add_argument("--simulations", type=int, default=100,
                   help="MCTS simulations per move during self-play.")
    p.add_argument("--epochs", type=int, default=5,
                   help="Training epochs per iteration.")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Minibatch size for optimization.")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Adam learning rate.")
    p.add_argument("--out", type=str, default="models",
                   help="Output directory for checkpoints and best.pt.")
    p.add_argument("--channels", type=int, default=64,
                   help="Residual block channel width.")
    p.add_argument("--blocks", type=int, default=5,
                   help="Number of residual blocks.")
    p.add_argument("--buffer-size", type=int, default=50000,
                   help="Replay buffer capacity.")
    p.add_argument("--temperature-moves", type=int, default=30,
                   help="Plies for which moves are sampled at temperature 1.")
    p.add_argument("--max-moves", type=int, default=400,
                   help="Maximum plies before a game is cut off as a draw.")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="Adam weight decay (L2 regularization).")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint to resume training from.")
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
    """Register the ``analyze`` subcommand (GUI board editor + analysis)."""
    p = subparsers.add_parser(
        "analyze",
        help="Open the GUI directly in the setup / analysis (board editor) mode.",
    )
    p.add_argument("--model", type=str, default="models/best.pt",
                   help="Path to the trained model (optional; enables analysis).")
    p.add_argument("--simulations", type=int, default=400,
                   help="MCTS simulations per analysis.")
    p.add_argument("--fen", type=str, default=None,
                   help="Optional FEN to prefill the editor.")
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
    """Dispatch the ``analyze`` subcommand: open the GUI in setup mode."""
    from alpha_chess.gui import launch_gui
    from alpha_chess.network import get_device

    model_path = args.model if args.model and os.path.isfile(args.model) else None
    if args.model and model_path is None:
        print("Warning: model '{}' not found; analysis disabled until a model "
              "is loaded.".format(args.model), file=sys.stderr)

    launch_gui(
        model_path=model_path,
        simulations=args.simulations,
        device=get_device(args.device),
        start_in_setup=True,
        initial_fen=args.fen,
    )


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

    args = parser.parse_args(argv)

    if args.command == "train":
        _run_train(args)
    elif args.command == "suggest":
        _run_suggest(args)
    elif args.command == "play":
        _run_play(args)
    elif args.command == "analyze":
        _run_analyze(args)
    else:  # pragma: no cover - argparse enforces a valid command.
        parser.error("Unknown command: {}".format(args.command))


if __name__ == "__main__":
    main()
