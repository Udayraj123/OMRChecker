import os
import shutil
from types import SimpleNamespace

from src.tests.test_samples.sample2.boilerplate import (
    CONFIG_BOILERPLATE,
    TEMPLATE_BOILERPLATE,
)
from src.tests.utils import run_entry_point, setup_mocker_patches, write_modified
from src.utils.image import ImageUtils

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_SAMPLE_PATH = os.path.join(CURRENT_DIR, "test_samples", "sample2")
VALID_IMAGE_NAME = "sample.jpg"
CORRUPT_IMAGE_NAME = "corrupt.jpg"


def _build_sample_dir(tmp_path):
    """Copy the sample2 fixture, write its boilerplate config and add a broken image."""
    sample_dir = tmp_path.joinpath("sample")
    shutil.copytree(BASE_SAMPLE_PATH, sample_dir)
    write_modified(None, TEMPLATE_BOILERPLATE, sample_dir.joinpath("template.json"))
    write_modified(None, CONFIG_BOILERPLATE, sample_dir.joinpath("config.json"))

    with open(sample_dir.joinpath(CORRUPT_IMAGE_NAME), "wb") as corrupt_file:
        corrupt_file.write(b"this is not a decodable image")

    return sample_dir


def test_load_omr_image_returns_empty_list_for_unreadable_image(tmp_path):
    """cv2.imread returns None for a corrupt file, which must not be handed on."""
    corrupt_path = tmp_path.joinpath(CORRUPT_IMAGE_NAME)
    with open(corrupt_path, "wb") as corrupt_file:
        corrupt_file.write(b"this is not a decodable image")

    images = ImageUtils.load_omr_image(corrupt_path, SimpleNamespace())

    assert images == []


def test_load_omr_image_still_returns_the_image_when_readable(tmp_path):
    """The guard must not affect images that decode normally."""
    valid_path = tmp_path.joinpath(VALID_IMAGE_NAME)
    shutil.copy(os.path.join(BASE_SAMPLE_PATH, VALID_IMAGE_NAME), valid_path)

    images = ImageUtils.load_omr_image(valid_path, SimpleNamespace())

    assert len(images) == 1
    assert images[0][0] == VALID_IMAGE_NAME
    assert images[0][1] is not None


def _strip_pre_processors(template_content):
    template_content["preProcessors"] = []


def test_all_images_unreadable_still_finishes_the_run(mocker, tmp_path):
    """A directory where every input is skipped must not crash the summary.

    load_omr_image returns [] for each unreadable file, so process_files
    never increments its counter. print_stats then computed
    time_checking / files_counter for the rate lines and raised
    ZeroDivisionError after the run had otherwise completed cleanly.
    """
    setup_mocker_patches(mocker)
    sample_dir = _build_sample_dir(tmp_path)
    # CropOnMarkers loads its marker at template construction time, which is
    # out of scope here - drop it so every image can be made unreadable.
    write_modified(
        _strip_pre_processors,
        TEMPLATE_BOILERPLATE,
        sample_dir.joinpath("template.json"),
    )
    output_dir = tmp_path.joinpath("outputs")

    for file_name in os.listdir(sample_dir):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            with open(sample_dir.joinpath(file_name), "wb") as image_file:
                image_file.write(b"this is not a decodable image")

    run_entry_point(str(sample_dir), str(output_dir))


def test_unreadable_image_does_not_abort_the_run(mocker, tmp_path):
    """
    A single unreadable file must be skipped, not crash the whole run.

    Before the fix, load_omr_image returned (name, None) and the caller
    dereferenced the None image (entry.py logs in_omr.shape), which raised
    AttributeError and aborted the directory.
    """
    setup_mocker_patches(mocker)
    sample_dir = _build_sample_dir(tmp_path)
    output_dir = tmp_path.joinpath("outputs")

    run_entry_point(str(sample_dir), str(output_dir))

    csv_contents = ""
    for dir_path, _subdirs, files in os.walk(output_dir):
        for file_name in files:
            if file_name.endswith(".csv"):
                with open(os.path.join(dir_path, file_name)) as csv_file:
                    csv_contents += csv_file.read()

    assert VALID_IMAGE_NAME in csv_contents, "readable image should still be processed"
    assert CORRUPT_IMAGE_NAME not in csv_contents
