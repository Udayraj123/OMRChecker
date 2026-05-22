"""Integration tests for the ``/api/v1/settings`` endpoints.

Covers the happy path (GET reflects live settings), partial PUTs, the
five validation rules (jpeg_quality bounds, page_format enum, min_pages
bounds, unknown keys, plus min_pages edges), the meta endpoint, and the
on-disk override file's format + resilience to corruption.

All tests run against the real FastAPI app via ``TestClient`` and use a
per-test ``cache_root`` so nothing touches the real
``%LOCALAPPDATA%\\OMRChecker\\cache`` directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from src.tests.utils import setup_mocker_patches
from webui.app import create_app
from webui.settings import (
    OVERRIDES_FILENAME,
    RUNTIME_MUTABLE_SETTINGS,
    Settings,
    get_settings,
    reload_settings,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the settings overrides file at a per-test tmp directory.

    Without this fixture the API would write to the user's real
    ``%LOCALAPPDATA%\\OMRChecker\\cache`` directory.
    """
    cache_dir = tmp_path / "settings_cache"
    cache_dir.mkdir()
    monkeypatch.setenv("OMR_WEBUI_CACHE_ROOT", str(cache_dir))
    # Re-build the cached Settings now that the env var is set, so the
    # very next ``get_settings()`` call (inside ``create_app``) picks it up.
    get_settings.cache_clear()
    yield cache_dir
    get_settings.cache_clear()


@pytest.fixture
def client(storage_root: Path, cache_root: Path, mocker) -> Iterator[TestClient]:
    """A TestClient with both storage_root and cache_root pointed at tmp dirs."""
    setup_mocker_patches(mocker)
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def _overrides_path() -> Path:
    """Return the live overrides path for the current Settings instance.

    Settings overrides moved out of ``cache_root`` (which can be quarantined
    by aggressive AV products on Windows) into ``storage_root.parent``; this
    helper hides that detail from each test.
    """
    return get_settings().overrides_path()


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_get_settings_returns_runtime_values(client: TestClient) -> None:
    """GET /settings must return every key in the mutable allowlist."""
    response = client.get("/api/v1/settings")
    assert response.status_code == 200, response.text
    body = response.json()

    # Every allowlisted key must appear, and only those keys.
    assert set(body.keys()) == set(RUNTIME_MUTABLE_SETTINGS)

    # Spot-check a few known defaults from the Settings model.
    assert body["auto_start_omr_min_pages"] == 10
    assert body["pdf_jpeg_quality"] == 92
    assert body["pdf_page_format"] == "jpeg"
    assert body["pdf_render_dpi"] == 150
    assert body["inmemory_pipeline"] is True


def test_put_partial_update_persists_and_reloads(
    client: TestClient, cache_root: Path
) -> None:
    """A single-field PUT writes only that field and a follow-up GET reflects it."""
    overrides_path = _overrides_path()
    assert not overrides_path.exists(), "fixture must start with no overrides"

    response = client.put(
        "/api/v1/settings", json={"auto_start_omr_min_pages": 42}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["auto_start_omr_min_pages"] == 42

    # File on disk must contain ONLY the field we wrote.
    assert overrides_path.exists()
    on_disk = json.loads(overrides_path.read_text(encoding="utf-8"))
    assert on_disk == {"auto_start_omr_min_pages": 42}

    # A subsequent GET must see the persisted value (cache was reloaded).
    follow_up = client.get("/api/v1/settings").json()
    assert follow_up["auto_start_omr_min_pages"] == 42


def test_put_merges_with_existing_overrides(
    client: TestClient, cache_root: Path
) -> None:
    """Successive PUTs must merge (not replace) the overrides file."""
    r1 = client.put("/api/v1/settings", json={"auto_start_omr_min_pages": 25})
    assert r1.status_code == 200, r1.text
    r2 = client.put("/api/v1/settings", json={"pdf_jpeg_quality": 88})
    assert r2.status_code == 200, r2.text

    on_disk = json.loads(_overrides_path().read_text(encoding="utf-8"))
    assert on_disk == {"auto_start_omr_min_pages": 25, "pdf_jpeg_quality": 88}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", [59, 0, -1, 101, 1000])
def test_put_validates_pdf_jpeg_quality_bounds(
    client: TestClient, bad_value: int
) -> None:
    """``pdf_jpeg_quality`` must be in ``[60, 100]``."""
    response = client.put("/api/v1/settings", json={"pdf_jpeg_quality": bad_value})
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("good_value", [60, 75, 92, 100])
def test_put_accepts_valid_pdf_jpeg_quality(
    client: TestClient, good_value: int
) -> None:
    response = client.put("/api/v1/settings", json={"pdf_jpeg_quality": good_value})
    assert response.status_code == 200, response.text
    assert response.json()["pdf_jpeg_quality"] == good_value


@pytest.mark.parametrize("bad_value", ["JPEG", "tiff", "bmp", "", "jpg"])
def test_put_validates_pdf_page_format_enum(
    client: TestClient, bad_value: str
) -> None:
    """``pdf_page_format`` must be exactly ``"jpeg"`` or ``"png"``."""
    response = client.put("/api/v1/settings", json={"pdf_page_format": bad_value})
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("good_value", ["jpeg", "png"])
def test_put_accepts_valid_pdf_page_format(
    client: TestClient, good_value: str
) -> None:
    response = client.put("/api/v1/settings", json={"pdf_page_format": good_value})
    assert response.status_code == 200, response.text
    assert response.json()["pdf_page_format"] == good_value


def test_put_validates_auto_start_min_pages_bounds(client: TestClient) -> None:
    """Both inclusive edges accepted, just-outside values rejected."""
    # Out of range: 0 and 10001 both 422.
    for bad in (0, 10001, -5, 50000):
        r = client.put("/api/v1/settings", json={"auto_start_omr_min_pages": bad})
        assert r.status_code == 422, f"expected 422 for {bad}, got {r.status_code}"

    # In range: 1 and 10000 both succeed.
    for good in (1, 10000):
        r = client.put("/api/v1/settings", json={"auto_start_omr_min_pages": good})
        assert r.status_code == 200, r.text
        assert r.json()["auto_start_omr_min_pages"] == good


def test_put_rejects_unknown_keys(client: TestClient) -> None:
    """``extra="forbid"`` must reject typos and non-allowlisted keys."""
    # A typo.
    r = client.put("/api/v1/settings", json={"auto_start_omr_min_page": 5})
    assert r.status_code == 422, r.text

    # A real Settings field that is not in the runtime-mutable allowlist.
    r = client.put("/api/v1/settings", json={"storage_root": "/tmp/foo"})
    assert r.status_code == 422, r.text

    # A field that doesn't exist at all on Settings.
    r = client.put("/api/v1/settings", json={"made_up_field": True})
    assert r.status_code == 422, r.text


# ---------------------------------------------------------------------------
# Meta endpoint
# ---------------------------------------------------------------------------


def test_meta_includes_all_mutable_keys_with_descriptions_and_defaults(
    client: TestClient,
) -> None:
    """``/settings/meta`` must expose every mutable key + its description + default."""
    response = client.get("/api/v1/settings/meta")
    assert response.status_code == 200, response.text
    body = response.json()

    expected_keys = set(RUNTIME_MUTABLE_SETTINGS)
    assert set(body["mutable_keys"]) == expected_keys
    assert body["mutable_keys"] == sorted(expected_keys), "keys must be sorted"

    assert set(body["descriptions"].keys()) == expected_keys
    assert set(body["defaults"].keys()) == expected_keys

    # Every description must be a non-empty string (helps the UI render labels).
    for key in expected_keys:
        desc = body["descriptions"][key]
        assert isinstance(desc, str) and desc.strip(), f"empty description for {key}"

    # Defaults must match the built-in Settings field defaults exactly.
    for key in expected_keys:
        field_info = Settings.model_fields[key]
        if field_info.default_factory is not None:
            expected_default = field_info.default_factory()
        else:
            expected_default = field_info.default
        assert body["defaults"][key] == expected_default, (
            f"default for {key} should be {expected_default!r}, "
            f"got {body['defaults'][key]!r}"
        )


# ---------------------------------------------------------------------------
# File-format guarantees
# ---------------------------------------------------------------------------


def test_overrides_file_is_atomic_and_human_readable(
    client: TestClient, cache_root: Path
) -> None:
    """The overrides JSON must be indented and key-sorted, and no .tmp left over."""
    r = client.put(
        "/api/v1/settings",
        json={
            "pdf_jpeg_quality": 85,
            "auto_start_omr_min_pages": 20,
            "pdf_page_format": "png",
        },
    )
    assert r.status_code == 200, r.text

    overrides_path = _overrides_path()
    raw = overrides_path.read_text(encoding="utf-8")

    # Indented with two spaces means there must be at least one newline + space run.
    assert "\n  " in raw, f"expected indented JSON, got: {raw!r}"

    # sort_keys=True means alphabetical: auto_start_... < pdf_jpeg_... < pdf_page_...
    parsed = json.loads(raw)
    assert list(parsed.keys()) == sorted(parsed.keys())
    assert list(parsed.keys()) == [
        "auto_start_omr_min_pages",
        "pdf_jpeg_quality",
        "pdf_page_format",
    ]

    # The atomic-write tmp file must NOT still be sitting around.
    tmp_leftover = overrides_path.with_suffix(overrides_path.suffix + ".tmp")
    assert not tmp_leftover.exists(), "atomic write left a .tmp file behind"


def test_corrupt_overrides_file_is_ignored_on_next_reload(
    storage_root: Path, cache_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A garbled overrides file must NOT crash startup; defaults take over."""
    overrides_path = _overrides_path()
    overrides_path.parent.mkdir(parents=True, exist_ok=True)
    overrides_path.write_text("{this is not valid json", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="webui.settings"):
        settings = reload_settings()

    # Defaults from the Settings model must be in effect (no override applied).
    assert settings.auto_start_omr_min_pages == 10
    assert settings.pdf_jpeg_quality == 92
    assert settings.pdf_page_format == "jpeg"

    # And a WARNING must have been logged so an operator can see why.
    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "corrupt settings overrides" in r.getMessage()
    ]
    assert warnings, (
        "expected at least one WARNING about corrupt overrides; "
        f"got: {[r.getMessage() for r in caplog.records]}"
    )
