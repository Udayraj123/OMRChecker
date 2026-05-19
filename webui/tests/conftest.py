"""Shared pytest fixtures for the webui integration tests."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from src.tests.utils import setup_mocker_patches
from webui.app import create_app
from webui.settings import get_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
ADRIAN_SAMPLE_DIR = REPO_ROOT / "samples" / "sample2" / "AdrianSample"
SAMPLE_TEMPLATE = REPO_ROOT / "samples" / "sample2" / "template.json"
SAMPLE_CONFIG = REPO_ROOT / "samples" / "sample2" / "config.json"


@pytest.fixture
def storage_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the web UI at a temporary batches directory for isolation."""
    batches_dir = tmp_path / "batches"
    batches_dir.mkdir()
    monkeypatch.setenv("OMR_WEBUI_STORAGE_ROOT", str(batches_dir))
    monkeypatch.setenv("OMR_WEBUI_DEFAULT_PRESET", "")
    get_settings.cache_clear()
    yield batches_dir
    get_settings.cache_clear()


@pytest.fixture
def client(storage_root: Path, mocker) -> Iterator[TestClient]:
    """Provide a FastAPI TestClient with OpenCV UI calls mocked out."""
    setup_mocker_patches(mocker)
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def adrian_images() -> list[Path]:
    return sorted(ADRIAN_SAMPLE_DIR.glob("*.png"))


@pytest.fixture
def sample_template_body() -> dict:
    import json
    return json.loads(SAMPLE_TEMPLATE.read_text(encoding="utf-8"))


@pytest.fixture
def sample_config_body() -> dict:
    import json
    return json.loads(SAMPLE_CONFIG.read_text(encoding="utf-8"))
