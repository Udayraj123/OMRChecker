#!/usr/bin/env python3
"""
Command-line script to calculate and print file checksums.

Usage:
    python scripts/local/checksum.py <file_path> [--algorithm <algorithm>]

Examples:
    python scripts/local/checksum.py README.md
    python scripts/local/checksum.py README.md --algorithm md5
    python scripts/local/checksum.py README.md --algorithm sha1
    python scripts/local/checksum.py README.md --algorithm sha512
"""

import argparse
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.file import calculate_file_checksum, print_file_checksum


def main():
    parser = argparse.ArgumentParser(
        description="Calculate and print file checksums",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s README.md
  %(prog)s README.md --algorithm md5
  %(prog)s README.md --algorithm sha1
  %(prog)s README.md --algorithm sha512

Supported algorithms: md5, sha1, sha256 (default), sha512
        """
    )

    parser.add_argument(
        "file_path",
        help="Path to the file to checksum"
    )

    parser.add_argument(
        "--algorithm", "-a",
        default="sha256",
        choices=["md5", "sha1", "sha256", "sha512"],
        help="Hash algorithm to use (default: sha256)"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only output the checksum value (no filename or algorithm label)"
    )

    args = parser.parse_args()

    try:
        if args.quiet:
            checksum = calculate_file_checksum(args.file_path, args.algorithm)
            print(checksum)
        else:
            print_file_checksum(args.file_path, args.algorithm)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
