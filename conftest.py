# This will inject the fixture to pytest
import pytest

from src.tests.__fixtures__.pytest_image_snapshot import (
    image_snapshot_fixture,
    image_snapshot_parser_hook,
)
from src.tests.__fixtures__.run_sample import run_sample_fixture, run_sample_parser_hook


# Register a custom image_snapshot fixture
@pytest.fixture
def image_snapshot(request):
    return image_snapshot_fixture(request)


# Register the run_sample fixture
@pytest.fixture
def run_sample(request):
    return run_sample_fixture(request)


# Add hook
def pytest_addoption(parser):
    run_sample_parser_hook(parser)
    image_snapshot_parser_hook(parser)
