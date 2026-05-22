"""Tests for per-worker config isolation in the in-memory OMR pipeline.

Fix #1: ``entry_point_for_image`` now builds ``TuningConfig`` in-memory via
``_build_tuning_config_inmemory``, eliminating the shared ``config.json``
race that caused ``JSONDecodeError`` / silent ``preprocess_failures`` under
parallel load on Windows.

Test structure
--------------
* ``test_baseline_correctness`` – single-call sanity check.
* ``test_parallel_config_isolation`` – 20 tasks × 8 workers; each task
  carries a deterministically distinct ``processing_height``/``processing_width``
  so that any cross-contamination between workers would produce mismatched
  dimensions or an outright crash.
"""

from __future__ import annotations

import json
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Repo / fixture paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
CUSTOM_DIR = REPO_ROOT / "custom_25_definitive_final"
SAMPLE_TEMPLATE = CUSTOM_DIR / "template.json"

# Use the baseline stress-test image — it's a clean scan the CropOnMarkers
# preprocessor can process without errors.
STRESS_ARTIFACTS = REPO_ROOT / "omr_stress_artifacts"
SAMPLE_IMAGE = STRESS_ARTIFACTS / "01_baseline.jpg"

# Fallback image if stress artefacts folder is absent (shouldn't happen).
_FALLBACK_IMAGE = CUSTOM_DIR / "MCFormat25questions_page-0001_filled_sample_markers.jpg"


def _resolve_sample_image() -> Path:
    if SAMPLE_IMAGE.exists():
        return SAMPLE_IMAGE
    if _FALLBACK_IMAGE.exists():
        return _FALLBACK_IMAGE
    raise FileNotFoundError(
        f"No sample image found; tried:\n  {SAMPLE_IMAGE}\n  {_FALLBACK_IMAGE}"
    )


# ---------------------------------------------------------------------------
# Module-level worker function
#
# Must be defined at module level (not as a lambda / nested closure) so that
# Python's ``spawn``-based multiprocessing on Windows can pickle it.
# ---------------------------------------------------------------------------

def _run_entry_point_for_image(payload: dict) -> dict:
    """Worker entrypoint — invokes ``entry_point_for_image`` and reports back.

    Returns a dict so the parent process can assert correctness without any
    shared mutable state.
    """
    # Ensure the repo root is on sys.path inside the spawned process.
    repo_root = payload["repo_root"]
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from src.entry import entry_point_for_image  # noqa: PLC0415

    out_dir = Path(payload["output_dir"])
    try:
        entry_point_for_image(
            image_path=payload["image_path"],
            output_dir=str(out_dir),
            template_payload=payload["template_payload"],
            config_payload=payload["config_payload"],
            template_dir=payload["template_dir"],
            rotation_degrees=0,
        )
        # Gather observable evidence of a successful run.
        results_dir = out_dir / "Results"
        error_dir = out_dir / "Manual" / "ErrorFiles"
        csvs = list(results_dir.glob("Results_*.csv")) if results_dir.exists() else []
        err_csvs = (
            list(error_dir.glob("*.csv")) + list(error_dir.glob("*.jpg"))
            if error_dir.exists()
            else []
        )
        return {
            "index": payload["index"],
            "requested_height": payload["config_payload"]["dimensions"]["processing_height"],
            "requested_width": payload["config_payload"]["dimensions"]["processing_width"],
            "has_results_csv": len(csvs) > 0,
            "has_error_files": len(err_csvs) > 0,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "index": payload["index"],
            "requested_height": payload["config_payload"]["dimensions"]["processing_height"],
            "requested_width": payload["config_payload"]["dimensions"]["processing_width"],
            "has_results_csv": False,
            "has_error_files": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config_payload(processing_height: int, processing_width: int) -> dict:
    """Return a minimal config dict with the specified processing dimensions."""
    return {
        "dimensions": {
            "processing_height": processing_height,
            "processing_width": processing_width,
            "display_height": processing_height,
            "display_width": processing_width,
        },
        "outputs": {
            "show_image_level": 0,
        },
    }


def _setup_template_dir(tmp_path: Path) -> Path:
    """Copy ``template.json`` (and any sibling assets) into a fresh tmp dir.

    Returns the populated template directory path.
    """
    tdir = tmp_path / "template_base"
    tdir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(SAMPLE_TEMPLATE, tdir / "template.json")
    # The aruco-based CropOnMarkers preprocessor doesn't need a marker image
    # file — no extra assets to copy for this template.
    return tdir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBaselineCorrectness:
    """Single-call sanity check — no parallelism, just verifies the function
    runs end-to-end and produces the expected output structure."""

    def test_single_call_produces_output(self, tmp_path: pytest.TempPathFactory) -> None:
        """``entry_point_for_image`` completes and writes a Results CSV or
        an error-files entry (depending on whether aruco markers are visible).
        Either outcome is fine — the key assertion is **no exception**."""
        image_path = _resolve_sample_image()
        template_dir = _setup_template_dir(tmp_path)
        output_dir = tmp_path / "out_baseline"

        from src.entry import entry_point_for_image  # noqa: PLC0415

        config_payload = _make_config_payload(515, 666)

        # Must not raise — that's the baseline contract.
        entry_point_for_image(
            image_path=str(image_path),
            output_dir=str(output_dir),
            template_payload=json.loads(SAMPLE_TEMPLATE.read_text(encoding="utf-8")),
            config_payload=config_payload,
            template_dir=str(template_dir),
            rotation_degrees=0,
        )

        results_dir = output_dir / "Results"
        error_dir = output_dir / "Manual" / "ErrorFiles"
        has_results = results_dir.exists() and bool(list(results_dir.glob("Results_*.csv")))
        has_errors = error_dir.exists() and bool(
            list(error_dir.iterdir())
        )
        assert has_results or has_errors, (
            "Expected either a Results CSV or an ErrorFiles entry after processing. "
            f"output_dir contents: {list(output_dir.rglob('*'))}"
        )

    def test_inmemory_config_no_disk_write(self, tmp_path: pytest.TempPathFactory) -> None:
        """After the fix, ``entry_point_for_image`` must NOT write ``config.json``
        into ``template_dir``. This is the core contract of the Option A fix."""
        image_path = _resolve_sample_image()
        template_dir = _setup_template_dir(tmp_path)
        output_dir = tmp_path / "out_no_config"

        from src.entry import entry_point_for_image  # noqa: PLC0415

        entry_point_for_image(
            image_path=str(image_path),
            output_dir=str(output_dir),
            template_payload=json.loads(SAMPLE_TEMPLATE.read_text(encoding="utf-8")),
            config_payload=_make_config_payload(515, 666),
            template_dir=str(template_dir),
            rotation_degrees=0,
        )

        config_json = template_dir / "config.json"
        assert not config_json.exists(), (
            f"entry_point_for_image must not write config.json to template_dir "
            f"(found at {config_json}). This would re-introduce the shared-file race."
        )

    def test_stats_reset_between_images_in_same_worker(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """STATS counters must be reset at the start of every
        ``entry_point_for_image`` call.

        ``STATS`` is a module-level singleton.  In a real ``ProcessPoolExecutor``
        each worker handles many images, so without the per-image reset the
        ``files_moved`` / ``files_not_moved`` counters accumulate across
        successive tasks and the engine's "Sum Tallied!" check goes out of
        sync.  This test pre-poisons ``STATS`` with fake counts and verifies
        the next call zeroes them.
        """
        from src import entry as entry_module  # noqa: PLC0415
        from src.entry import entry_point_for_image  # noqa: PLC0415

        entry_module.STATS.files_moved = 42
        entry_module.STATS.files_not_moved = 99

        image_path = _resolve_sample_image()
        template_dir = _setup_template_dir(tmp_path)
        output_dir = tmp_path / "out_stats_reset"

        entry_point_for_image(
            image_path=str(image_path),
            output_dir=str(output_dir),
            template_payload=json.loads(SAMPLE_TEMPLATE.read_text(encoding="utf-8")),
            config_payload=_make_config_payload(515, 666),
            template_dir=str(template_dir),
            rotation_degrees=0,
        )

        # After exactly one image, the totals should reflect *this* image only,
        # not the pre-poisoned 42 + 99 = 141 baseline.
        assert (
            entry_module.STATS.files_moved + entry_module.STATS.files_not_moved
        ) <= 2, (
            f"STATS counters not reset between images: "
            f"files_moved={entry_module.STATS.files_moved}, "
            f"files_not_moved={entry_module.STATS.files_not_moved}"
        )

    def test_invalid_rotation_raises(self, tmp_path: pytest.TempPathFactory) -> None:
        """Invalid rotations must fail loudly in the in-memory path.

        The legacy disk-staged web path raises ``ValueError`` for unsupported
        rotations.  ``entry_point_for_image`` must match that behaviour instead
        of silently skipping rotation and producing potentially wrong OMR
        results.
        """
        image_path = _resolve_sample_image()
        template_dir = _setup_template_dir(tmp_path)
        output_dir = tmp_path / "out_invalid_rotation"

        from src.entry import entry_point_for_image  # noqa: PLC0415

        with pytest.raises(ValueError, match="Unsupported rotation: 45"):
            entry_point_for_image(
                image_path=str(image_path),
                output_dir=str(output_dir),
                template_payload=json.loads(SAMPLE_TEMPLATE.read_text(encoding="utf-8")),
                config_payload=_make_config_payload(515, 666),
                template_dir=str(template_dir),
                rotation_degrees=45,
            )


class TestParallelConfigIsolation:
    """Parallelism stress test.

    Spawns 8 workers and submits 20 tasks, each with a deterministically
    distinct ``processing_height``/``processing_width``.  All tasks share the
    same ``template_dir`` (simulating the real production layout where all
    workers share ``<base_root>``).

    Before the fix: workers raced over ``template_dir/config.json``, causing
    ``JSONDecodeError`` → ``preprocess_failures``.
    After the fix: each worker builds its ``TuningConfig`` in-memory, so no
    shared file is written or read.

    Assertions
    ----------
    * No task raises an exception.
    * Every task produces either a Results CSV or ends up in ErrorFiles
      (both are legitimate — the point is no crash / JSONDecodeError).
    """

    # Deterministically distinct dimension pairs — interleave two sets so
    # concurrent workers always carry different values.
    _DIM_VARIANTS: list[tuple[int, int]] = [
        (515, 666),
        (480, 620),
        (500, 645),
        (460, 600),
    ]

    @staticmethod
    def _dims_for_index(idx: int) -> tuple[int, int]:
        variants = TestParallelConfigIsolation._DIM_VARIANTS
        return variants[idx % len(variants)]

    def test_20_tasks_8_workers_no_exception(self, tmp_path: pytest.TempPathFactory) -> None:
        image_path = _resolve_sample_image()
        template_dir = _setup_template_dir(tmp_path)
        template_payload = json.loads(SAMPLE_TEMPLATE.read_text(encoding="utf-8"))

        n_tasks = 20
        n_workers = 8

        payloads = []
        for i in range(n_tasks):
            h, w = self._dims_for_index(i)
            payloads.append({
                "repo_root": str(REPO_ROOT),
                "index": i,
                "image_path": str(image_path),
                "output_dir": str(tmp_path / f"out_{i:03d}"),
                "template_payload": template_payload,
                "config_payload": _make_config_payload(h, w),
                "template_dir": str(template_dir),
            })

        results: list[dict] = []
        errors: list[str] = []

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_run_entry_point_for_image, p): p for p in payloads}
            for future in as_completed(futures):
                result = future.result()  # re-raises if the future itself failed
                results.append(result)
                if result["error"]:
                    errors.append(
                        f"Task {result['index']} (dims {result['requested_height']}×"
                        f"{result['requested_width']}): {result['error']}"
                    )

        assert len(results) == n_tasks, (
            f"Expected {n_tasks} results, got {len(results)}"
        )

        assert not errors, (
            f"{len(errors)} of {n_tasks} tasks raised exceptions:\n"
            + "\n".join(errors)
        )

        # Every task should produce some output (either a Results CSV or
        # an ErrorFiles entry — both are valid outcomes for this image).
        tasks_with_output = [
            r for r in results if r["has_results_csv"] or r["has_error_files"]
        ]
        assert len(tasks_with_output) == n_tasks, (
            f"Only {len(tasks_with_output)}/{n_tasks} tasks produced any output. "
            "Tasks without output (unexpected blank run): "
            + str([r["index"] for r in results if not r["has_results_csv"] and not r["has_error_files"]])
        )
