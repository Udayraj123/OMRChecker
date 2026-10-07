import os
import shutil
from pathlib import Path

import pandas as pd

from src.tests.test_samples.sample1.boilerplate import (
    TEMPLATE_BOILERPLATE as SAMPLE1_TEMPLATE_BOILERPLATE,
)
from src.tests.test_samples.sample2.boilerplate import (
    CONFIG_BOILERPLATE,
    TEMPLATE_BOILERPLATE,
)
from src.tests.utils import (
    generate_write_jsons_and_run,
    remove_file,
    run_entry_point,
    setup_mocker_patches,
)

from time import strftime, localtime

from freezegun import freeze_time
from src.tests.utils import FROZEN_TIMESTAMP

with freeze_time(FROZEN_TIMESTAMP):
    TIME_NOW_HRS = strftime("%I%p", localtime())

CURRENT_DIR = Path("src/tests")
BASE_SAMPLE_PATH = CURRENT_DIR.joinpath("test_samples", "sample2")
SAMPLE1_PATH = CURRENT_DIR.joinpath("test_samples", "sample1")
SAMPLE1_OUTPUT_PATH = Path("outputs") / SAMPLE1_PATH
BASE_RESULTS_CSV_PATH = os.path.join(
    "outputs", BASE_SAMPLE_PATH, "Results", f"Results_{TIME_NOW_HRS}.csv"
)
BASE_MULTIMARKED_CSV_PATH = os.path.join(
    "outputs", BASE_SAMPLE_PATH, "Manual", "MultiMarkedFiles.csv"
)


def run_sample(mocker, input_path):
    setup_mocker_patches(mocker)
    output_dir = os.path.join("outputs", input_path)
    run_entry_point(input_path, output_dir)


def extract_output_data(path):
    output_data = pd.read_csv(path, keep_default_na=False)
    return output_data


write_jsons_and_run = generate_write_jsons_and_run(
    run_sample,
    sample_path=BASE_SAMPLE_PATH,
    template_boilerplate=TEMPLATE_BOILERPLATE,
    config_boilerplate=CONFIG_BOILERPLATE,
)

write_sample1_jsons_and_run = generate_write_jsons_and_run(
    run_sample,
    sample_path=SAMPLE1_PATH,
    template_boilerplate=SAMPLE1_TEMPLATE_BOILERPLATE,
)


def test_config_low_dimensions(mocker):
    def modify_config(config):
        config["dimensions"]["processing_height"] = 1000
        config["dimensions"]["processing_width"] = 1000

    exception = write_jsons_and_run(mocker, modify_config=modify_config)

    assert str(exception) == "No Error"


def test_different_bubble_dimensions(mocker):
    # Prevent appending to output csv:
    remove_file(BASE_RESULTS_CSV_PATH)
    remove_file(BASE_MULTIMARKED_CSV_PATH)

    exception = write_jsons_and_run(mocker)
    assert str(exception) == "No Error"
    original_output_data = extract_output_data(BASE_RESULTS_CSV_PATH)

    def modify_template(template):
        # Incorrect global bubble size
        template["bubbleDimensions"] = [5, 5]
        # Correct bubble size for MCQBlock1a1
        template["fieldBlocks"]["MCQBlock1a1"]["bubbleDimensions"] = [32, 32]
        # Incorrect bubble size for MCQBlock1a11
        template["fieldBlocks"]["MCQBlock1a11"]["bubbleDimensions"] = [10, 10]

    remove_file(BASE_RESULTS_CSV_PATH)
    remove_file(BASE_MULTIMARKED_CSV_PATH)
    exception = write_jsons_and_run(mocker, modify_template=modify_template)
    assert str(exception) == "No Error"

    results_output_data = extract_output_data(BASE_RESULTS_CSV_PATH)

    assert results_output_data.empty

    output_data = extract_output_data(BASE_MULTIMARKED_CSV_PATH)

    equal_columns = [f"q{i}" for i in range(1, 18)]
    assert (
        output_data[equal_columns].iloc[0].to_list()
        == original_output_data[equal_columns].iloc[0].to_list()
    )

    unequal_columns = [f"q{i}" for i in range(168, 185)]
    assert not (
        output_data[unequal_columns].iloc[0].to_list()
        == original_output_data[unequal_columns].iloc[0].to_list()
    )


def sample1_output_counts():
    """Count rows written to ErrorFiles.csv and to every Results csv."""
    error_rows = 0
    errors_csv = SAMPLE1_OUTPUT_PATH.joinpath("Manual", "ErrorFiles.csv")
    if errors_csv.exists():
        error_rows = len(
            [row for row in errors_csv.read_text().splitlines()[1:] if row.strip()]
        )

    result_rows = 0
    results_dir = SAMPLE1_OUTPUT_PATH.joinpath("Results")
    if results_dir.exists():
        for results_csv in results_dir.glob("*.csv"):
            result_rows += len(
                [row for row in results_csv.read_text().splitlines()[1:] if row.strip()]
            )
    return error_rows, result_rows


def test_croppage_page_not_found_then_continue_on_page_not_found(mocker):
    # morphKernel is enlarged so sample.png has no detectable page boundary:
    # CropPage's fallback is what end users hit with a bad scan or photo.
    def force_missing_boundary(template):
        template["preProcessors"][0]["options"]["morphKernel"] = [600, 600]

    def also_continue_on_page_not_found(template):
        force_missing_boundary(template)
        template["preProcessors"][0]["options"]["continueOnPageNotFound"] = True

    # Error case: boundary is not found and the image is routed to ErrorFiles.
    shutil.rmtree(SAMPLE1_OUTPUT_PATH, ignore_errors=True)
    exception = write_sample1_jsons_and_run(
        mocker, modify_template=force_missing_boundary
    )
    assert str(exception) == "No Error"
    error_rows, result_rows = sample1_output_counts()
    assert error_rows == 1
    assert result_rows == 0

    # No-error case: same input, flag flipped - the image is processed instead.
    shutil.rmtree(SAMPLE1_OUTPUT_PATH, ignore_errors=True)
    exception = write_sample1_jsons_and_run(
        mocker, modify_template=also_continue_on_page_not_found
    )
    assert str(exception) == "No Error"
    error_rows, result_rows = sample1_output_counts()
    assert error_rows == 0
    assert result_rows == 1
