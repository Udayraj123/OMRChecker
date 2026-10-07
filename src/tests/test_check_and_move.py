import shutil

import cv2
import numpy as np
import pytest

from src.constants.common import ERROR_CODES
from src.entry import STATS, check_and_move, process_files


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
    assert not list(destination.parent.glob("tmp_*"))
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
    assert STATS.files_not_moved == 1


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
    assert STATS.files_not_moved == 1


def test_check_and_move_false_when_destination_dir_missing(tmp_path):
    source = tmp_path / "input.png"
    destination = tmp_path / "Missing" / "input.png"

    source.write_bytes(b"test")

    result = check_and_move(ERROR_CODES.NO_MARKER_ERR, source, destination)

    assert result is False
    assert source.exists()
    assert not destination.exists()
    assert STATS.files_moved == 0
    assert STATS.files_not_moved == 1


def test_check_and_move_writes_rendered_image_when_formats_differ(tmp_path):
    # A rejected PDF page is stored under its rendered .png name, so copying
    # the source bytes would produce a file that claims to be an image.
    source = tmp_path / "sheet.pdf"
    source.write_bytes(b"%PDF-1.4 placeholder")
    destination = tmp_path / "Manual" / "ErrorFiles" / "sheet_1.png"
    destination.parent.mkdir(parents=True)

    rendered_image = np.full((16, 16, 3), 255, dtype=np.uint8)

    result = check_and_move(
        ERROR_CODES.NO_MARKER_ERR, source, destination, rendered_image=rendered_image
    )

    assert result is True
    assert source.read_bytes().startswith(b"%PDF")
    saved = cv2.imread(str(destination))
    assert saved is not None
    assert saved.shape == (16, 16, 3)
    assert bool((saved == 255).all())
    assert not list(destination.parent.glob("tmp_*"))
    assert STATS.files_moved == 1
    assert STATS.files_not_moved == 0


def test_check_and_move_copies_source_when_formats_match(tmp_path):
    # Same-format destinations keep the original bytes: no re-encoding of a
    # non-PDF source that can be copied as-is.
    source = tmp_path / "input.pdf"
    source.write_bytes(b"%PDF-1.4 placeholder")
    destination = tmp_path / "Manual" / "ErrorFiles" / "input.pdf"
    destination.parent.mkdir(parents=True)

    rendered_image = np.zeros((16, 16, 3), dtype=np.uint8)

    result = check_and_move(
        ERROR_CODES.NO_MARKER_ERR, source, destination, rendered_image=rendered_image
    )

    assert result is True
    assert destination.read_bytes() == b"%PDF-1.4 placeholder"
    assert STATS.files_moved == 1


def test_check_and_move_cleans_up_interrupted_copy_and_allows_retry(tmp_path, mocker):
    source = tmp_path / "input.png"
    destination = tmp_path / "Manual" / "ErrorFiles" / "input.png"
    destination.parent.mkdir(parents=True)
    source.write_bytes(b"test")

    real_copy2 = shutil.copy2
    copy_mock = mocker.patch(
        "src.entry.shutil.copy2", side_effect=OSError("disk full")
    )

    result = check_and_move(ERROR_CODES.NO_MARKER_ERR, source, destination)

    assert result is False
    assert not destination.exists()
    assert not list(destination.parent.glob("tmp_*"))
    assert STATS.files_moved == 0
    assert STATS.files_not_moved == 1

    # The failed attempt must not leave anything behind that blocks a retry.
    copy_mock.side_effect = real_copy2
    result = check_and_move(ERROR_CODES.NO_MARKER_ERR, source, destination)

    assert result is True
    assert destination.read_bytes() == b"test"
    assert not list(destination.parent.glob("tmp_*"))
    assert STATS.files_moved == 1
    assert STATS.files_not_moved == 1


def test_process_files_resets_movement_counters_per_batch(mocker):
    # process_dir calls process_files once per image-bearing directory, so a
    # later batch must not inherit the earlier batch's movement counts.
    STATS.files_moved = 7
    STATS.files_not_moved = 3
    mocker.patch("src.entry.print_stats")

    process_files(
        [],
        template=None,
        tuning_config=None,
        evaluation_config=None,
        outputs_namespace=None,
    )

    assert STATS.files_moved == 0
    assert STATS.files_not_moved == 0
