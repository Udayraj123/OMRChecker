"""Tests for Fix #2 — pipelined OMR processing with in-flight PDF splits.

These tests exercise ``run_batch_sync`` in isolation, patching away the
real OMR engine and process pool so the suite stays fast and deterministic.

Patch strategy
--------------
* ``ProcessPoolExecutor``  → ``ThreadPoolExecutor`` so that the
  ``_process_one_image`` monkeypatch is visible inside the "workers"
  (they run as threads in the same process, not spawned subprocesses).
* ``_process_one_image``   → ``_fake_process_one_image``, a synchronous
  stub that returns a minimal success payload without touching disk.
* ``_prepare_runtime_base``→ a no-op that returns a temp directory;
  avoids template/asset setup for these purely structural tests.
* ``_discover_input_images``→ per-test side-effect function that controls
  which pages the run loop "sees" on each call.

Metadata is written directly via ``batches_service.update_batch_metadata``
so the cancel-check and pipeline-check code paths read real values.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from webui.services import batches as batches_service
from webui.services.omr import run_batch_sync
from webui.settings import Settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(storage_root: Path, tmp_path: Path, pipeline: bool = True) -> Settings:
    """Construct an isolated Settings object for a single test."""
    return Settings(
        storage_root=storage_root,
        default_preset="",
        cache_root=tmp_path / "omr_cache",
        pipeline_omr_with_split=pipeline,
    )


def _write_tiny_jpeg(path: Path) -> None:
    """Write a minimal 8×8 grayscale JPEG to *path*."""
    from PIL import Image as PILImage
    img = PILImage.new("L", (8, 8), 128)
    img.save(str(path), "JPEG", quality=75)


def _setup_batch(
    settings: Settings,
    image_count: int,
) -> tuple[str, list[Path]]:
    """Create a batch, write a minimal template.json, and populate inputs/.

    Returns ``(batch_id, list_of_image_paths)``.
    """
    batch = batches_service.create_batch("Pipeline test", settings)
    batch_root = settings.storage_root / batch.id
    inputs_dir = batch_root / "inputs"
    inputs_dir.mkdir(exist_ok=True)

    (batch_root / "template.json").write_text(
        json.dumps({"bubbleDimensions": [40, 40], "pageDimensions": [600, 800]}),
        encoding="utf-8",
    )

    image_paths: list[Path] = []
    for i in range(1, image_count + 1):
        p = inputs_dir / f"page_{i:04d}.jpg"
        _write_tiny_jpeg(p)
        image_paths.append(p)

    return batch.id, image_paths


def _fake_process_one_image(payload: dict[str, Any]) -> dict[str, Any]:
    """Fast OMR stub — returns a minimal success payload without any disk I/O."""
    image_path = Path(payload["image_path"])
    return {
        "file_name": image_path.name,
        "index": int(payload.get("index", 1)),
        "dynamic_dimensions": {
            "source_height": 8,
            "source_width": 8,
            "display_height": 8,
            "display_width": 8,
            "processing_height": 8,
            "processing_width": 8,
        },
        "runtime_config": {},
        "ended_in_errors": False,
        "worker_outputs_dir": None,
        "results_header": [],
        "results_rows": [],
        "mm_header": [],
        "mm_rows": [],
        "err_header": [],
        "err_rows": [],
        "checked_image": None,
        "error_image": None,
        "mm_image": None,
        "error": None,
    }


def _run_patched(batch_id: str, settings: Settings, mock_discover=None) -> None:
    """Run ``run_batch_sync`` with heavy machinery patched out.

    * ``ProcessPoolExecutor`` → ``ThreadPoolExecutor`` (same-process threads).
    * ``_process_one_image``  → ``_fake_process_one_image``.
    * ``_prepare_runtime_base``→ returns a fresh temp sub-dir.
    * ``_discover_input_images``→ ``mock_discover`` when provided, otherwise
      the real implementation.
    """
    fake_base = settings.cache_root / "fake_base"
    fake_base.mkdir(parents=True, exist_ok=True)

    patches: list[Any] = [
        patch("webui.services.omr.ProcessPoolExecutor", ThreadPoolExecutor),
        patch("webui.services.omr._process_one_image", _fake_process_one_image),
        patch(
            "webui.services.omr._prepare_runtime_base",
            MagicMock(return_value=fake_base),
        ),
    ]
    if mock_discover is not None:
        patches.append(
            patch("webui.services.omr._discover_input_images", mock_discover)
        )

    with patches[0], patches[1], patches[2]:
        if mock_discover is not None:
            with patches[3]:
                run_batch_sync(batch_id, settings)
        else:
            run_batch_sync(batch_id, settings)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelinedOMR:
    """Structural tests for the pipelined-split OMR loop."""

    def test_pipelined_omr_processes_late_arriving_pages(
        self, storage_root: Path, tmp_path: Path
    ) -> None:
        """Pages dropped into inputs/ while OMR is running are picked up.

        Scenario
        --------
        * Batch starts with 3 pages already in ``inputs/``.
        * ``pdf_split_total = 5`` is set in metadata to signal an in-flight
          split.
        * A second ``_discover_input_images`` call (triggered when the first
          iterator is exhausted) returns all 5 pages and resets
          ``pdf_split_total`` to 0 to signal split completion.
        * All 5 pages must be reported as processed.
        """
        settings = _make_settings(storage_root, tmp_path, pipeline=True)
        batch_id, all_paths = _setup_batch(settings, 5)
        initial_paths = all_paths[:3]

        batches_service.update_batch_metadata(
            batch_id, {"pdf_split_total": 5}, settings
        )

        discover_calls: list[int] = []

        def mock_discover(bid: str, s: Settings) -> list[Path]:
            discover_calls.append(len(discover_calls))
            if len(discover_calls) == 1:
                return list(initial_paths)
            # Simulate split completing and all pages now available
            batches_service.update_batch_metadata(
                bid, {"pdf_split_total": 0}, s
            )
            return list(all_paths)

        _run_patched(batch_id, settings, mock_discover)

        meta = batches_service.get_batch_metadata(batch_id, settings)
        assert meta["processed_files"] == 5, (
            f"Expected 5 processed files, got {meta['processed_files']}"
        )
        assert len(discover_calls) >= 2, (
            "Expected at least two _discover_input_images calls (initial + pipeline re-check)"
        )

    def test_legacy_mode_ignores_late_arriving_pages(
        self, storage_root: Path, tmp_path: Path
    ) -> None:
        """With pipeline_omr_with_split=False only the initial snapshot is used.

        Late-arriving pages (simulated by having mock_discover return more on
        subsequent calls) must NOT be processed when pipelining is disabled.
        """
        settings = _make_settings(storage_root, tmp_path, pipeline=False)
        batch_id, all_paths = _setup_batch(settings, 5)
        initial_paths = all_paths[:3]

        batches_service.update_batch_metadata(
            batch_id, {"pdf_split_total": 5}, settings
        )

        discover_calls: list[int] = []

        def mock_discover(bid: str, s: Settings) -> list[Path]:
            discover_calls.append(len(discover_calls))
            if len(discover_calls) == 1:
                return list(initial_paths)
            # This branch should never be reached in legacy mode
            batches_service.update_batch_metadata(
                bid, {"pdf_split_total": 0}, s
            )
            return list(all_paths)

        _run_patched(batch_id, settings, mock_discover)

        meta = batches_service.get_batch_metadata(batch_id, settings)
        assert meta["processed_files"] == 3, (
            f"Legacy mode must process only initial 3 pages, got {meta['processed_files']}"
        )
        assert len(discover_calls) == 1, (
            "Legacy mode must call _discover_input_images exactly once (no re-discovery)"
        )

    def test_pipeline_terminates_when_split_finishes_with_no_new_pages(
        self, storage_root: Path, tmp_path: Path
    ) -> None:
        """Run exits cleanly when the split finishes without producing new pages.

        Scenario
        --------
        * 3 pages in ``inputs/``, ``pdf_split_total = 5``.
        * The pipeline re-discovery returns the SAME 3 pages (no new ones)
          but also marks the split as complete (``pdf_split_total = 0``).
        * The run must terminate normally with 3 processed files, not hang.
        """
        settings = _make_settings(storage_root, tmp_path, pipeline=True)
        batch_id, all_paths = _setup_batch(settings, 3)

        batches_service.update_batch_metadata(
            batch_id, {"pdf_split_total": 5}, settings
        )

        discover_calls: list[int] = []

        def mock_discover(bid: str, s: Settings) -> list[Path]:
            discover_calls.append(len(discover_calls))
            if len(discover_calls) == 1:
                return list(all_paths)
            # Re-discovery: same pages, but split is now done
            batches_service.update_batch_metadata(
                bid, {"pdf_split_total": 0}, s
            )
            return list(all_paths)  # no new pages

        _run_patched(batch_id, settings, mock_discover)

        meta = batches_service.get_batch_metadata(batch_id, settings)
        assert meta["processed_files"] == 3, (
            f"Expected 3 processed files (split produced no new pages), "
            f"got {meta['processed_files']}"
        )

    def test_milestone_logs_reset_when_input_pool_grows(
        self, storage_root: Path, tmp_path: Path, caplog
    ) -> None:
        """Bug 1 regression: progress milestone logs must keep firing after
        the input pool grows mid-run.

        Without the re-anchor fix, completing 10 of an initial 20 items pushes
        ``last_milestone`` to 50.  When the pool grows to 40 items, items
        11–24 all fall under ``_milestone=50`` against the new denominator,
        suppressing every progress log between 25% and 60% of the new total.

        This test sets up exactly that scenario and asserts:
          1. The "OMR input pool grew" log fires with the correct totals.
          2. At least one progress log fires AFTER the pool grew but
             BEFORE the run ends (i.e. mid-second-half logging is no longer
             starved).
        """
        import logging

        settings = _make_settings(storage_root, tmp_path, pipeline=True)
        batch_id, all_paths = _setup_batch(settings, 40)
        initial_paths = all_paths[:20]

        batches_service.update_batch_metadata(
            batch_id, {"pdf_split_total": 40}, settings
        )

        discover_calls: list[int] = []

        def mock_discover(bid: str, s: Settings) -> list[Path]:
            discover_calls.append(len(discover_calls))
            if len(discover_calls) == 1:
                return list(initial_paths)
            batches_service.update_batch_metadata(
                bid, {"pdf_split_total": 0}, s
            )
            return list(all_paths)

        with caplog.at_level(logging.INFO, logger="webui.services.omr"):
            _run_patched(batch_id, settings, mock_discover)

        meta = batches_service.get_batch_metadata(batch_id, settings)
        assert meta["processed_files"] == 40, (
            f"Expected all 40 images processed, got {meta['processed_files']}"
        )

        grew_messages = [r for r in caplog.records if "OMR input pool grew" in r.getMessage()]
        assert grew_messages, (
            "Expected an 'OMR input pool grew' log when input_images was extended"
        )
        grew_msg = grew_messages[0].getMessage()
        assert "previous_total=20" in grew_msg, grew_msg
        assert "new_total=40" in grew_msg, grew_msg
        assert "added=20" in grew_msg, grew_msg

        progress_messages = [r for r in caplog.records if "OMR progress |" in r.getMessage()]
        assert len(progress_messages) >= 2, (
            f"Expected progress logs both before and after pool grew; "
            f"got {len(progress_messages)} progress lines: "
            f"{[m.getMessage() for m in progress_messages]}"
        )
        # At least one progress log must reference the new denominator (40).
        post_grow_progress = [
            r for r in progress_messages if " 40 |" in r.getMessage() or "/40 |" in r.getMessage()
        ]
        assert post_grow_progress, (
            "Expected at least one progress log against the new total of 40; "
            f"got: {[m.getMessage() for m in progress_messages]}"
        )

    def test_pipeline_does_not_double_submit_existing_pages(
        self, storage_root: Path, tmp_path: Path
    ) -> None:
        """Existing pages must not be processed more than once.

        Even when pipeline re-discovery returns the same paths that were
        already submitted, the ``submitted_paths`` guard must prevent
        double-submission.
        """
        settings = _make_settings(storage_root, tmp_path, pipeline=True)
        batch_id, all_paths = _setup_batch(settings, 3)

        batches_service.update_batch_metadata(
            batch_id, {"pdf_split_total": 5}, settings
        )

        discover_calls: list[int] = []
        processed_names: list[str] = []

        original_fake = _fake_process_one_image

        def counting_fake(payload: dict[str, Any]) -> dict[str, Any]:
            processed_names.append(Path(payload["image_path"]).name)
            return original_fake(payload)

        def mock_discover(bid: str, s: Settings) -> list[Path]:
            discover_calls.append(len(discover_calls))
            if len(discover_calls) == 1:
                return list(all_paths)
            # Re-discovery: same 3 pages, split done — no new pages at all
            batches_service.update_batch_metadata(
                bid, {"pdf_split_total": 0}, s
            )
            return list(all_paths)

        fake_base = settings.cache_root / "fake_base"
        fake_base.mkdir(parents=True, exist_ok=True)

        with patch("webui.services.omr.ProcessPoolExecutor", ThreadPoolExecutor), \
             patch("webui.services.omr._process_one_image", counting_fake), \
             patch(
                 "webui.services.omr._prepare_runtime_base",
                 MagicMock(return_value=fake_base),
             ), \
             patch("webui.services.omr._discover_input_images", mock_discover):
            run_batch_sync(batch_id, settings)

        assert len(processed_names) == 3, (
            f"Each page must be processed exactly once; "
            f"got {len(processed_names)} invocations: {processed_names}"
        )
        assert len(set(processed_names)) == 3, (
            f"All 3 distinct pages must be processed; "
            f"got unique={set(processed_names)}"
        )

    def test_pipeline_final_discovery_after_split_clears_progress(
        self, storage_root: Path, tmp_path: Path
    ) -> None:
        """Pages published just before split completion must not be missed.

        Regression coverage for the race where ``pdf_split_total`` reached 0
        before the OMR loop's next re-discovery pass. The loop must do one
        final input scan before declaring the split done.
        """
        settings = _make_settings(storage_root, tmp_path, pipeline=True)
        batch_id, all_paths = _setup_batch(settings, 5)
        initial_paths = all_paths[:3]

        batches_service.update_batch_metadata(
            batch_id, {"pdf_split_total": 5}, settings
        )

        discover_calls: list[int] = []

        def mock_discover(bid: str, s: Settings) -> list[Path]:
            discover_calls.append(len(discover_calls))
            if len(discover_calls) == 1:
                return list(initial_paths)
            batches_service.update_batch_metadata(
                bid, {"pdf_split_total": 0}, s
            )
            return list(all_paths)

        original_get_meta = batches_service.get_batch_metadata

        def mock_get_meta(bid: str, s: Settings) -> dict[str, Any]:
            meta = original_get_meta(bid, s)
            if len(discover_calls) >= 1:
                meta = dict(meta)
                meta["pdf_split_total"] = 0
            return meta

        with patch(
            "webui.services.omr.batches_service.get_batch_metadata",
            mock_get_meta,
        ):
            _run_patched(batch_id, settings, mock_discover)

        meta = batches_service.get_batch_metadata(batch_id, settings)
        assert meta["processed_files"] == 5, (
            f"Expected final discovery to process all 5 pages, got {meta['processed_files']}"
        )
        assert len(discover_calls) >= 2

    def test_pipeline_split_error_marks_batch_failed(
        self, storage_root: Path, tmp_path: Path
    ) -> None:
        """A PDF split error observed mid-pipeline must fail the run clearly."""
        settings = _make_settings(storage_root, tmp_path, pipeline=True)
        batch_id, all_paths = _setup_batch(settings, 3)

        batches_service.update_batch_metadata(
            batch_id,
            {"pdf_split_total": 5, "pdf_split_error": "synthetic split failure"},
            settings,
        )

        _run_patched(batch_id, settings, MagicMock(return_value=list(all_paths)))

        batch = batches_service.get_batch(batch_id, settings)
        assert batch.status.value == "failed"
        assert batch.last_error is not None
        assert "synthetic split failure" in batch.last_error

    def test_pipelined_run_flag_written_and_logged_when_engaged(
        self, storage_root: Path, tmp_path: Path, caplog
    ) -> None:
        """When pipelining engages, both the metadata flag and the
        ``OMR PIPELINED MODE`` log line must be emitted so the UI/operator
        can confirm the feature actually fired.
        """
        import logging

        settings = _make_settings(storage_root, tmp_path, pipeline=True)
        batch_id, all_paths = _setup_batch(settings, 3)
        batches_service.update_batch_metadata(
            batch_id, {"pdf_split_total": 5}, settings
        )

        # Only reset pdf_split_total on the re-discovery call so the run's
        # initial split snapshot remains 5 (pipelining engages).
        discover_calls: list[int] = []

        def mock_discover(bid: str, s: Settings) -> list[Path]:
            discover_calls.append(len(discover_calls))
            if len(discover_calls) == 1:
                return list(all_paths)
            batches_service.update_batch_metadata(
                bid, {"pdf_split_total": 0}, s
            )
            return list(all_paths)

        with caplog.at_level(logging.INFO, logger="webui.services.omr"):
            _run_patched(batch_id, settings, mock_discover)

        meta = batches_service.get_batch_metadata(batch_id, settings)
        assert meta.get("pipelined_run") is True, (
            "pipelined_run metadata flag must be True when pipeline engages"
        )
        engaged = [r for r in caplog.records if "OMR PIPELINED MODE engaged" in r.getMessage()]
        assert engaged, (
            "Expected 'OMR PIPELINED MODE engaged' log when initial_split_total > 0"
        )

    def test_pipelined_run_flag_false_when_split_already_done(
        self, storage_root: Path, tmp_path: Path, caplog
    ) -> None:
        """No in-flight split → metadata flag stays False and SEQUENTIAL log
        is emitted (this matches the UI-driven 'wait then click Process' flow
        that the user reported)."""
        import logging

        settings = _make_settings(storage_root, tmp_path, pipeline=True)
        batch_id, all_paths = _setup_batch(settings, 3)
        # pdf_split_total is implicitly 0 (split already finished).

        with caplog.at_level(logging.INFO, logger="webui.services.omr"):
            _run_patched(
                batch_id,
                settings,
                MagicMock(return_value=list(all_paths)),
            )

        meta = batches_service.get_batch_metadata(batch_id, settings)
        assert meta.get("pipelined_run") is False, (
            "pipelined_run must be False when no in-flight split exists at run start"
        )
        sequential = [r for r in caplog.records if "OMR SEQUENTIAL MODE" in r.getMessage()]
        assert sequential, (
            "Expected 'OMR SEQUENTIAL MODE' log when initial_split_total == 0"
        )
