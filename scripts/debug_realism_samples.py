"""Generate prefilled sheets at every realism preset and report pixel deltas.

Run from the repo root:
    python scripts/debug_realism_samples.py

Writes:
    debug_realism/<preset>.png       full sheet at preset
    debug_realism/<preset>_diff.png  per-pixel diff vs none (amplified 4x)
    debug_realism/report.txt         numerical comparison
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from webui.services import prefill as prefill_service  # noqa: E402

OUT_DIR = ROOT / "debug_realism"
OUT_DIR.mkdir(exist_ok=True)

ROW = dict(
    student_name="Jane Doe",
    school_name="Sample School",
    exam_name="Sample Exam",
    candidate_number="9010690012",
)

PRESETS = ("none", "subtle", "moderate", "adversarial")


def _save_png(name: str, data: bytes) -> Image.Image:
    path = OUT_DIR / f"{name}.png"
    path.write_bytes(data)
    return Image.open(path).convert("RGB")


def _amplified_diff(base: np.ndarray, other: np.ndarray, factor: int = 4) -> np.ndarray:
    diff = np.abs(base.astype(np.int16) - other.astype(np.int16))
    amplified = np.clip(diff * factor, 0, 255).astype(np.uint8)
    return amplified


def main() -> None:
    images: dict[str, Image.Image] = {}
    for preset in PRESETS:
        print(f"Rendering {preset}...", flush=True)
        png = prefill_service.generate_single_png(**ROW, realism_preset=preset)
        images[preset] = _save_png(preset, png)

    base = np.array(images["none"], dtype=np.uint8)
    h, w, _ = base.shape

    # ROI: candidate-number bubble grid (right-hand block, top half).
    roi_x0, roi_y0 = int(w * 0.60), int(h * 0.18)
    roi_x1, roi_y1 = int(w * 0.98), int(h * 0.50)

    report_lines: list[str] = []
    report_lines.append(f"Image size: {w}x{h}")
    report_lines.append(f"Candidate-number ROI: ({roi_x0},{roi_y0})-({roi_x1},{roi_y1})")
    report_lines.append("")
    report_lines.append(
        f"{'preset':<14} {'PNG bytes':>12} {'mean_diff':>10} {'roi_mean':>9} {'max_diff':>9} {'changed_%':>10}"
    )

    for preset in PRESETS:
        arr = np.array(images[preset], dtype=np.uint8)
        if preset != "none":
            diff = _amplified_diff(base, arr, factor=4)
            Image.fromarray(diff).save(OUT_DIR / f"{preset}_diff.png")
        raw = np.abs(arr.astype(np.int16) - base.astype(np.int16))
        roi_diff = raw[roi_y0:roi_y1, roi_x0:roi_x1].mean()
        changed_pct = float((raw.max(axis=-1) > 5).mean()) * 100.0
        size = (OUT_DIR / f"{preset}.png").stat().st_size
        report_lines.append(
            f"{preset:<14} {size:>12} {raw.mean():>10.3f} {roi_diff:>9.3f} {raw.max():>9} {changed_pct:>9.2f}%"
        )

    (OUT_DIR / "report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print()
    print("\n".join(report_lines))
    print()
    print(f"Samples written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
