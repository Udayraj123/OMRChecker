"""Regression tests for the auto-start OMR trigger contract.

These cover the **server side** of the contract: the JSON payload that the
frontend poller in ``webui/static/batch.js`` reads from ``GET
/api/v1/batches/{id}/status`` to decide whether to auto-fire
``POST /api/v1/batches/{id}/process``.

We deliberately do NOT spin up a headless browser — the JS gating logic
has an in-file ``__autoStartTriggerSelfTest`` hook for ad-hoc browser
testing, and the cross-cutting risk here is the server payload contract
(missing field, wrong type, stale cached settings). That is what these
tests pin.

Coverage
--------
* defaults exposed on the status payload (with_split=True, min_pages=10,
  require_config=False).
* runtime override via ``write_overrides`` + ``reload_settings`` flips
  ``auto_start_omr_with_split`` to False on the next /status fetch.
* runtime override flips ``auto_start_omr_min_pages`` from 10 to 100.
* ``has_template`` / ``has_config`` reflect filesystem state and flip
  from False → True when the docs are PUT to the batch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from src.tests.utils import setup_mocker_patches
from webui.app import create_app
from webui.settings import get_settings, reload_settings, write_overrides


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
# We can't use the shared ``client`` fixture from conftest.py because that
# fixture only isolates storage_root — it leaves cache_root pointing at
# the user's real %LOCALAPPDATA%\OMRChecker\cache. write_overrides()
# would happily pollute that. Below we isolate BOTH roots per test.


@pytest.fixture
def isolated_settings_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> Iterator[tuple[TestClient, Path]]:
    """TestClient with isolated storage_root AND cache_root.

    Yields ``(client, cache_root)`` so individual tests can call
    ``write_overrides(..., cache_root)`` without touching the user's real
    cache directory.
    """
    batches_dir = tmp_path / "batches"
    cache_dir = tmp_path / "cache"
    batches_dir.mkdir()
    cache_dir.mkdir()
    monkeypatch.setenv("OMR_WEBUI_STORAGE_ROOT", str(batches_dir))
    monkeypatch.setenv("OMR_WEBUI_CACHE_ROOT", str(cache_dir))
    monkeypatch.setenv("OMR_WEBUI_DEFAULT_PRESET", "")
    get_settings.cache_clear()
    setup_mocker_patches(mocker)
    app = create_app()
    with TestClient(app) as client:
        yield client, cache_dir
    get_settings.cache_clear()


def _create_batch(client: TestClient, name: str = "auto-start regression") -> str:
    resp = client.post("/api/v1/batches", json={"name": name})
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _seed_split_progress(
    batch_id: str, pages: int, total: int, cache_root: Path
) -> None:
    """Write split-progress metadata directly so we don't have to actually
    render a PDF. Mirrors what the background split task writes."""
    # Re-import inside the helper so the test's monkeypatched env is visible.
    from webui.services import batches as batches_service
    settings = get_settings()
    batches_service.update_batch_metadata(
        batch_id,
        {"pdf_split_pages": pages, "pdf_split_total": total},
        settings,
    )


_MIN_TEMPLATE = {
    "bubbleDimensions": [40, 40],
    "pageDimensions": [600, 800],
}


_MIN_CONFIG = {
    "dimensions": {
        "processing_height": 800,
        "processing_width": 600,
        "display_height": 800,
        "display_width": 600,
    },
    "outputs": {"show_image_level": 0},
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_status_payload_exposes_auto_start_fields_with_defaults_on(
    isolated_settings_client: tuple[TestClient, Path],
) -> None:
    """With no overrides, /status must surface the three auto-start fields
    using their Settings defaults so the frontend doesn't need a separate
    /api/v1/settings fetch on every poll tick."""
    client, cache_root = isolated_settings_client
    batch_id = _create_batch(client)
    client.put(f"/api/v1/batches/{batch_id}/template", json=_MIN_TEMPLATE)
    _seed_split_progress(batch_id, pages=15, total=20, cache_root=cache_root)

    resp = client.get(f"/api/v1/batches/{batch_id}/status")
    assert resp.status_code == 200, resp.text
    payload = resp.json()

    # The contract the frontend depends on.
    assert payload["auto_start_omr_with_split"] is True
    assert payload["auto_start_omr_min_pages"] == 10
    assert payload["auto_start_omr_require_config"] is False
    # And the gating-relevant batch flags.
    assert payload["has_template"] is True
    assert payload["has_config"] is False
    # Sanity: split-progress metadata round-trips.
    assert payload["pdf_split_pages"] == 15
    assert payload["pdf_split_total"] == 20


def test_status_payload_reflects_disabled_setting(
    isolated_settings_client: tuple[TestClient, Path],
) -> None:
    """Writing ``auto_start_omr_with_split: false`` to the overrides JSON
    and reloading settings must flip the field on the next /status call.

    This is the contract that lets the /settings page disable auto-start
    without a process restart."""
    client, cache_root = isolated_settings_client
    batch_id = _create_batch(client)

    # Baseline: defaults expose True.
    baseline = client.get(f"/api/v1/batches/{batch_id}/status").json()
    assert baseline["auto_start_omr_with_split"] is True

    # Persist override and force a reload so get_settings() rebuilds.
    write_overrides({"auto_start_omr_with_split": False}, cache_root)
    reload_settings()

    resp = client.get(f"/api/v1/batches/{batch_id}/status")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["auto_start_omr_with_split"] is False
    # The other two fields keep their defaults.
    assert payload["auto_start_omr_min_pages"] == 10
    assert payload["auto_start_omr_require_config"] is False


def test_status_payload_reflects_min_pages_override(
    isolated_settings_client: tuple[TestClient, Path],
) -> None:
    """Setting ``auto_start_omr_min_pages: 100`` via the overrides file
    must propagate to the status payload, so an operator who wants to
    amortise pool spawn over more pages doesn't need to restart."""
    client, cache_root = isolated_settings_client
    batch_id = _create_batch(client)

    write_overrides({"auto_start_omr_min_pages": 100}, cache_root)
    reload_settings()

    resp = client.get(f"/api/v1/batches/{batch_id}/status")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["auto_start_omr_min_pages"] == 100
    # Other auto-start fields keep their defaults.
    assert payload["auto_start_omr_with_split"] is True
    assert payload["auto_start_omr_require_config"] is False


def test_has_template_and_has_config_flags_reflect_filesystem_state(
    isolated_settings_client: tuple[TestClient, Path],
) -> None:
    """has_template / has_config must accurately mirror the on-disk batch
    so the frontend gating check (\"don't auto-fire without a template\")
    is correct on a freshly-created batch."""
    client, _cache_root = isolated_settings_client
    batch_id = _create_batch(client)

    # 1) Fresh batch: neither doc exists.
    payload = client.get(f"/api/v1/batches/{batch_id}/status").json()
    assert payload["has_template"] is False
    assert payload["has_config"] is False

    # 2) Upload template only.
    put_t = client.put(f"/api/v1/batches/{batch_id}/template", json=_MIN_TEMPLATE)
    assert put_t.status_code == 200, put_t.text
    payload = client.get(f"/api/v1/batches/{batch_id}/status").json()
    assert payload["has_template"] is True
    assert payload["has_config"] is False

    # 3) Upload config too.
    put_c = client.put(f"/api/v1/batches/{batch_id}/config", json=_MIN_CONFIG)
    assert put_c.status_code == 200, put_c.text
    payload = client.get(f"/api/v1/batches/{batch_id}/status").json()
    assert payload["has_template"] is True
    assert payload["has_config"] is True
