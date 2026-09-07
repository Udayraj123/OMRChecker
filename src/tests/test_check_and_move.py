import pytest

from src.constants.common import ERROR_CODES
from src.entry import STATS, check_and_move


@pytest.fixture(autouse=True)
def reset_stats():
    files_moved_before = STATS.files_moved
    files_not_moved_before = STATS.files_not_moved
    STATS.files_moved = 0
    STATS.files_not_moved = 0
    yield
    STATS.files_moved = files_moved_before
    STATS.files_not_moved = files_not_moved_before


def test_check_and_move_copies_file(tmp_path):
    source = tmp_path / "input.png"
    destination = tmp_path / "Manual" / "ErrorFiles" / "input.png"
    destination.parent.mkdir(parents=True)

    source.write_bytes(b"test")

    result = check_and_move(ERROR_CODES.NO_MARKER_ERR, source, destination)

    assert result is True
    assert source.exists()
    assert source.read_bytes() == b"test"
    assert destination.exists()
    assert destination.read_bytes() == b"test"
    assert STATS.files_moved == 1
    assert STATS.files_not_moved == 0


def test_check_and_move_false_when_source_missing(tmp_path):
    source = tmp_path / "input.png"
    destination = tmp_path / "Manual" / "ErrorFiles" / "input.png"
    destination.parent.mkdir(parents=True)

    result = check_and_move(ERROR_CODES.NO_MARKER_ERR, source, destination)

    assert result is False
    assert not destination.exists()
    assert STATS.files_moved == 0
    assert STATS.files_not_moved == 0


def test_check_and_move_false_when_destination_exists(tmp_path):
    source = tmp_path / "input.png"
    destination = tmp_path / "Manual" / "ErrorFiles" / "input.png"
    destination.parent.mkdir(parents=True)

    source.write_bytes(b"source")
    destination.write_bytes(b"existing")

    result = check_and_move(ERROR_CODES.NO_MARKER_ERR, source, destination)

    assert result is False
    assert source.exists()
    assert destination.read_bytes() == b"existing"
    assert STATS.files_moved == 0
    assert STATS.files_not_moved == 0


def test_check_and_move_false_when_destination_dir_missing(tmp_path):
    source = tmp_path / "input.png"
    destination = tmp_path / "Missing" / "input.png"

    source.write_bytes(b"test")

    result = check_and_move(ERROR_CODES.NO_MARKER_ERR, source, destination)

    assert result is False
    assert source.exists()
    assert not destination.exists()
    assert STATS.files_moved == 0
    assert STATS.files_not_moved == 0
