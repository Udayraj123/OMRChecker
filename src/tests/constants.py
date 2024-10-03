import os
from pathlib import Path

FROZEN_TIMESTAMP = "1970-01-01"
CURRENT_DIR = Path("src", "tests")
BASE_SAMPLE_PATH = CURRENT_DIR.joinpath("test_samples", "sample2")
IMAGE_SNAPSHOTS_PATH = CURRENT_DIR.joinpath("__image_snapshots__")
COLORED_IMAGE_SNAPSHOTS_PATH = IMAGE_SNAPSHOTS_PATH.joinpath("colored")

CHECKED_OMRS_PATH = Path("outputs", "CheckedOMRs")
COLORED_CHECKED_OMRS_PATH = CHECKED_OMRS_PATH.joinpath("colored")
BASE_RESULTS_CSV_PATH = os.path.join(
    "outputs", BASE_SAMPLE_PATH, "Results", "Results_05AM.csv"
)
BASE_MULTIMARKED_CSV_PATH = os.path.join(
    "outputs", BASE_SAMPLE_PATH, "Manual", "MultiMarkedFiles.csv"
)
