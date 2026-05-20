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
CUSTOM_DIR = REPO_ROOT / "custom_25_definitive_final"
CUSTOM_INPUTS_DIR = CUSTOM_DIR / "inputs"
SAMPLE_TEMPLATE = CUSTOM_DIR / "template.json"
SAMPLE_CONFIG = CUSTOM_DIR / "config.json"


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
    return sorted(CUSTOM_INPUTS_DIR.glob("*.jpg"))


@pytest.fixture
def sample_template_body() -> dict:
    import json
    return json.loads(SAMPLE_TEMPLATE.read_text(encoding="utf-8"))


@pytest.fixture
def sample_config_body() -> dict:
    import json
    return json.loads(SAMPLE_CONFIG.read_text(encoding="utf-8"))
