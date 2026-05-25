#!/usr/bin/env python3
"""
Generates a summary spreadsheet from OMR Results CSV.
Creates an accumulated score table per question, plus survey count and mean statistics.
"""

import csv
import sys
from pathlib import Path


def parse_score(value: str) -> int | None:
    """
    Extracts score 0-10 from cell value (OMR stores concatenated marked digits).
    Single digit or "10" → that score. Multi-mark → highest marked value. All 11 marked → None.
    """
    val = str(value).strip()
    if not val:
        return None
    if val == "10":
        return 10
    if len(val) == 1 and val in "0123456789":
        return int(val)
    if len(val) >= 10:
        return None
    if "10" in val:
        return 10
    digits = [int(ch) for ch in val if ch in "0123456789"]
    return max(digits) if digits else None


def load_results(csv_path: Path) -> tuple[list[str], list[dict]]:
    """Loads results CSV and returns question columns and list of row dicts."""
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        questions = [c for c in reader.fieldnames if c.startswith("q") and c[1:].isdigit()]
        rows = list(reader)
    return questions, rows


def build_accumulation_table(questions: list[str], rows: list[dict]) -> dict[str, dict[int, int]]:
    """Builds per-question score accumulation (count per score 0-10)."""
    table = {q: {i: 0 for i in range(11)} for q in questions}
    for row in rows:
        for q in questions:
            score = parse_score(row.get(q, ""))
            if score is not None:
                table[q][score] += 1
    return table


def compute_means(
    table: dict[str, dict[int, int]], questions: list[str]
) -> tuple[float, dict[str, float]]:
    """Computes overall mean and per-question mean."""
    total_sum = 0
    total_count = 0
    per_question = {}
    for q in questions:
        count = sum(table[q].values())
        s = sum(score * table[q][score] for score in range(11))
        total_sum += s
        total_count += count
        per_question[q] = s / count if count else 0.0
    overall = total_sum / total_count if total_count else 0.0
    return overall, per_question


def write_summary(
    out_path: Path,
    table: dict[str, dict[int, int]],
    questions: list[str],
    n_surveys: int,
    overall_mean: float,
    per_question_mean: dict[str, float],
    question_labels: dict[str, str] | None = None,
):
    """Writes summary CSV in resultado format. Stats are in label/value columns to the right of the table with two blank columns in between."""
    labels = question_labels or {q: q for q in questions}
    scores = list(range(11))
    table_width = 1 + len(scores)
    sep = 2

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        empty = [""] * table_width
        w.writerow(empty)
        w.writerow(empty)
        w.writerow(empty)
        w.writerow(["PREGUNTA"] + scores)

        for q in questions:
            row = [labels.get(q, q)] + [table[q][s] for s in scores]
            w.writerow(row)

        total_row = [
            "TOTAL",
        ] + [sum(table[q][s] for q in questions) for s in scores]
        w.writerow(total_row)

        w.writerow(empty)
        w.writerow(empty)
        w.writerow(empty)
        pad = [""] * table_width + [""] * sep
        w.writerow(pad + ["Número de encuestas:", n_surveys])
        w.writerow(pad + ["Media todas las encuestas:", f"{overall_mean:.2f}"])
        for q in questions:
            w.writerow(pad + [f"Media {labels.get(q, q)}:", f"{per_question_mean[q]:.2f}"])


HELP_TEXT = """
Usage: summarize_results.py [INPUT_CSV] [OUTPUT_CSV]

Generates a summary spreadsheet from OMR Results CSV.
Creates an accumulated score table per question, plus survey count and mean statistics.

Arguments:
  INPUT_CSV   Path to the results CSV (default: outputs/Results/Results_01PM.csv)
  OUTPUT_CSV  Path for the output summary CSV (default: resultado.csv)

Options:
  -h, --help  Show this help and exit.
"""


def main() -> int:
    if any(arg in ("-h", "--help") for arg in sys.argv[1:3]):
        print(HELP_TEXT.strip())
        return 0

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    input_path = project_root / "outputs" / "Results" / "Results_01PM.csv"
    output_path = project_root / "resultado.csv"

    args = [a for a in sys.argv[1:] if a not in ("-h", "--help")]
    if len(args) >= 1:
        input_path = Path(args[0])
    if len(args) >= 2:
        output_path = Path(args[1])

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return 1

    question_labels = {
        "q1": "1 (global)",
        "q2": "2 (persona)",
        "q3": "3 (duración)",
        "q4": "4 (repetir)",
        "q5": "5 (material)",
        "q6": "6 (sala)",
    }

    questions, rows = load_results(input_path)
    n_surveys = len(rows)
    table = build_accumulation_table(questions, rows)
    overall_mean, per_question_mean = compute_means(table, questions)

    write_summary(
        output_path,
        table,
        questions,
        n_surveys,
        overall_mean,
        per_question_mean,
        question_labels,
    )
    print(f"Summary written to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
