"""CLI interface for OMRChecker - Command-line specific functionality.

This module separates CLI concerns from core processing logic.
"""

from src.cli.argument_parser import parse_args
from src.cli.cli_runner import run_cli

__all__ = ["parse_args", "run_cli"]
