"""OMRChecker - Main CLI entry point.

Thin wrapper that delegates to the CLI module for cleaner separation of concerns.
"""

from src.cli import parse_args, run_cli

if __name__ == "__main__":
    args = parse_args()
    run_cli(args)
