#!/usr/bin/env python3
"""
Generate test fixtures for Python-TypeScript parity testing.

This script runs Python implementations of key utilities and generates
JSON fixtures that TypeScript tests can use to verify identical behavior.

Usage:
    python scripts/generate_parity_fixtures.py

Output:
    omrchecker-js/packages/core/tests/parity/fixtures.json
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import math as math_utils
from src.utils import stats as stats_utils


def generate_math_fixtures():
    """Generate fixtures for math utility functions."""
    fixtures = {
        "distance": [],
        "add_points": [],
        "subtract_points": [],
        "angle": [],
        "check_collinear_points": [],
    }

    # Distance tests
    distance_cases = [
        {"p1": [0, 0], "p2": [3, 4]},
        {"p1": [0, 0], "p2": [0, 0]},
        {"p1": [1, 1], "p2": [1, 1]},
        {"p1": [-1, -1], "p2": [1, 1]},
        {"p1": [10, 20], "p2": [13, 24]},
    ]

    for case in distance_cases:
        result = math_utils.MathUtils.distance(case["p1"], case["p2"])
        fixtures["distance"].append(
            {"input": {"p1": case["p1"], "p2": case["p2"]}, "output": result}
        )

    # Add points tests
    add_cases = [
        {"p1": [0, 0], "p2": [1, 1]},
        {"p1": [5, 10], "p2": [3, 4]},
        {"p1": [-1, -2], "p2": [3, 4]},
    ]

    for case in add_cases:
        result = math_utils.MathUtils.add_points(case["p1"], case["p2"])
        fixtures["add_points"].append({"input": case, "output": result})

    # Subtract points tests
    sub_cases = [
        {"p1": [5, 5], "p2": [2, 3]},
        {"p1": [10, 20], "p2": [5, 10]},
        {"p1": [0, 0], "p2": [1, 1]},
    ]

    for case in sub_cases:
        result = math_utils.MathUtils.subtract_points(case["p1"], case["p2"])
        fixtures["subtract_points"].append({"input": case, "output": result})

    # Angle tests (cosine between three points)
    angle_cases = [
        {"p1": [1, 0], "p2": [0, 1], "p0": [0, 0]},
        {"p1": [1, 0], "p2": [1, 0], "p0": [0, 0]},
        {"p1": [2, 0], "p2": [0, 2], "p0": [0, 0]},
    ]

    for case in angle_cases:
        result = math_utils.MathUtils.angle(case["p1"], case["p2"], case["p0"])
        fixtures["angle"].append({"input": case, "output": result})

    # Check collinear points
    collinear_cases = [
        {"p1": [0, 0], "p2": [1, 1], "p3": [2, 2]},  # collinear
        {"p1": [0, 0], "p2": [1, 0], "p3": [2, 0]},  # collinear
        {"p1": [0, 0], "p2": [1, 1], "p3": [1, 0]},  # not collinear
    ]

    for case in collinear_cases:
        result = math_utils.MathUtils.check_collinear_points(
            case["p1"], case["p2"], case["p3"]
        )
        fixtures["check_collinear_points"].append({"input": case, "output": result})

    return fixtures


def generate_stats_fixtures():
    """Generate fixtures for stats utility functions."""
    fixtures = {
        "mean": [],
        "median": [],
        "standard_deviation": [],
        "mode": [],
    }

    # Mean tests
    mean_cases = [
        [1, 2, 3, 4, 5],
        [10, 20, 30],
        [5],
        [1.5, 2.5, 3.5],
        [-1, 0, 1],
    ]

    for case in mean_cases:
        result = stats_utils.mean(case)
        fixtures["mean"].append({"input": case, "output": result})

    # Median tests
    median_cases = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4],
        [5],
        [1, 3, 2],
        [10, 1, 5, 3],
    ]

    for case in median_cases:
        result = stats_utils.median(case)
        fixtures["median"].append({"input": case, "output": result})

    # Standard deviation tests
    std_cases = [
        [1, 2, 3, 4, 5],
        [10, 10, 10],
        [1, 5],
        [2, 4, 6, 8],
    ]

    for case in std_cases:
        result = stats_utils.standard_deviation(case)
        fixtures["standard_deviation"].append({"input": case, "output": result})

    # Mode tests (if available)
    if hasattr(stats_utils, "mode"):
        mode_cases = [
            [1, 2, 2, 3],
            [1, 1, 2, 2, 3],
            [5, 5, 5],
        ]

        for case in mode_cases:
            result = stats_utils.mode(case)
            fixtures["mode"].append({"input": case, "output": result})

    return fixtures


def generate_checksum_fixtures():
    """Generate fixtures for checksum utility functions."""
    import hashlib

    fixtures = {
        "calculate_checksum": [],
    }

    # Checksum tests - compute directly with hashlib
    checksum_cases = [
        {"data": "hello world", "algorithm": "md5"},
        {"data": "test data", "algorithm": "md5"},
        {"data": "", "algorithm": "md5"},
        {"data": "The quick brown fox", "algorithm": "sha256"},
        {"data": "test", "algorithm": "sha256"},
    ]

    for case in checksum_cases:
        hasher = hashlib.new(case["algorithm"])
        hasher.update(case["data"].encode("utf-8"))
        result = hasher.hexdigest()

        fixtures["calculate_checksum"].append(
            {
                "input": {"data": case["data"], "algorithm": case["algorithm"]},
                "output": result,
            }
        )

    return fixtures


def main():
    """Generate all fixtures and save to JSON."""
    print("Generating Python-TypeScript parity fixtures...")

    fixtures = {
        "math": generate_math_fixtures(),
        "checksum": generate_checksum_fixtures(),
        "metadata": {
            "generated_by": "scripts/generate_parity_fixtures.py",
            "python_version": sys.version,
            "description": "Test fixtures for verifying Python-TypeScript parity",
        },
    }

    # Create output directory
    output_dir = (
        Path(__file__).parent.parent
        / "omrchecker-js"
        / "packages"
        / "core"
        / "tests"
        / "parity"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write fixtures
    output_file = output_dir / "fixtures.json"
    with open(output_file, "w") as f:
        json.dump(fixtures, f, indent=2)

    print(f"✓ Fixtures written to {output_file}")
    print(f"  - Math: {len(fixtures['math'])} function groups")
    print(f"  - Checksum: {len(fixtures['checksum'])} function groups")


if __name__ == "__main__":
    main()
