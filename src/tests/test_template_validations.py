import os
from pathlib import Path

import pytest

from src.tests.test_samples.sample1.boilerplate import TEMPLATE_BOILERPLATE
from src.tests.utils import (
    generate_write_jsons_and_run,
    run_entry_point,
    setup_mocker_patches,
)
from src.utils.logger import logger

# All tests in this file use the same shared sample directory (sample1)
# and write to the same files (template.json, config.json, evaluation.json).
# Mark entire file as serial to prevent race conditions in parallel execution.
pytestmark = pytest.mark.serial

FROZEN_TIMESTAMP = "1970-01-01 00:00:00+00:00"
CURRENT_DIR = Path("src/tests")
BASE_SAMPLE_PATH = CURRENT_DIR.joinpath("test_samples", "sample1")
BASE_SAMPLE_TEMPLATE_PATH = BASE_SAMPLE_PATH.joinpath("template.json")


def run_sample(mocker, input_path) -> None:
    setup_mocker_patches(mocker)
    output_dir = Path("outputs", input_path)
    run_entry_point(input_path, output_dir)
    mocker.resetall()


write_jsons_and_run = generate_write_jsons_and_run(
    run_sample,
    sample_path=BASE_SAMPLE_PATH,
    template_boilerplate=TEMPLATE_BOILERPLATE,
)


def test_no_input_dir(mocker) -> None:
    try:
        run_sample(mocker, "X")
    except Exception as e:
        # Updated to match custom exception message format
        assert "Input directory does not exist: 'X'" in str(e)


def test_no_template(mocker) -> None:
    if BASE_SAMPLE_TEMPLATE_PATH.exists():
        os.remove(BASE_SAMPLE_TEMPLATE_PATH)
    try:
        run_sample(mocker, BASE_SAMPLE_PATH)
    except Exception as e:
        # Updated to match custom exception message format
        assert "No template.json found in directory tree" in str(e)


def test_empty_template(mocker) -> None:
    def modify_template(_) -> dict:
        return {}

    logger.debug("\nExpecting invalid template json error logs:")
    _, exception = write_jsons_and_run(mocker, modify_template=modify_template)
    # Updated to match custom exception message format
    assert "Invalid template JSON" in str(exception)


def test_invalid_bubble_field_type(mocker) -> None:
    def modify_template(template) -> None:
        template["fieldBlocks"]["MCQ_Block_1"]["bubbleFieldType"] = "X"

    logger.debug("\nExpecting invalid template json error logs:")
    _, exception = write_jsons_and_run(mocker, modify_template=modify_template)
    # Updated to match custom exception message format
    assert "Invalid template JSON" in str(exception)


def test_overflow_labels(mocker) -> None:
    def modify_template(template) -> None:
        template["fieldBlocks"]["MCQ_Block_1"]["fieldLabels"] = ["q1..100"]

    _, exception = write_jsons_and_run(mocker, modify_template=modify_template)
    assert (
        str(exception)
        == "Overflowing field block 'MCQ_Block_1' with origin [65.0, 60.0] and dimensions [189.0, 5173.0] in template with dimensions [300, 400] (block_name=MCQ_Block_1, bounding_box_origin=[65.0, 60.0], bounding_box_dimensions=[189.0, 5173.0], template_dimensions=[300, 400])"
    )


def test_overflow_safe_dimensions(mocker) -> None:
    def modify_template(template) -> None:
        template["templateDimensions"] = [255, 400]

    _, exception = write_jsons_and_run(mocker, modify_template=modify_template)
    assert str(exception) == "No Error"


def test_field_strings_overlap(mocker) -> None:
    def modify_template(template) -> None:
        template["fieldBlocks"] = {
            **template["fieldBlocks"],
            "New_Block": {
                **template["fieldBlocks"]["MCQ_Block_1"],
                "fieldLabels": ["q5"],
            },
        }

    _, exception = write_jsons_and_run(mocker, modify_template=modify_template)
    assert str(exception) == (
        "The field strings for field block New_Block overlap with other existing fields: {'q5'} (block_name=New_Block, field_labels=['q5'], overlap=['q5'])"
    )


def test_custom_label_strings_overlap_single(mocker) -> None:
    def modify_template(template) -> None:
        template["customLabels"] = {
            "label1": ["q1..2", "q2..3"],
        }

    _, exception = write_jsons_and_run(mocker, modify_template=modify_template)
    assert (
        str(exception)
        == "Given field string 'q2..3' has overlapping field(s) with other fields in 'Custom Label: label1': ['q1..2', 'q2..3'] (field_string=q2..3, key=Custom Label: label1, overlapping_fields=['q2'])"
    )


def test_custom_label_strings_overlap_multiple(mocker) -> None:
    def modify_template(template) -> None:
        template["customLabels"] = {
            "label1": ["q1..2"],
            "label2": ["q2..3"],
        }

    _, exception = write_jsons_and_run(mocker, modify_template=modify_template)
    assert (
        str(exception)
        == "The field strings for custom label 'label2' overlap with other existing custom labels (custom_label=label2, label_strings=['q2..3'])"
    )


def test_missing_field_block_labels(mocker) -> None:
    def modify_template(template) -> None:
        template["customLabels"] = {"Combined": ["qX", "qY"]}

    _, exception = write_jsons_and_run(mocker, modify_template=modify_template)
    assert (
        str(exception)
        == "Missing field block label(s) in the given template for ['qX', 'qY'] from 'Combined' (custom_label=Combined, missing_labels=['qX', 'qY'])"
    )


def test_missing_output_columns(mocker) -> None:
    def modify_template(template) -> None:
        template["outputColumns"] = {
            "sortType": "CUSTOM",
            "customOrder": ["qX", "q1..5"],
        }

    _, exception = write_jsons_and_run(mocker, modify_template=modify_template)
    assert str(exception) == (
        "Some columns are missing in the field blocks for the given output columns (missing_output_columns=['qX'])"
    )


def test_safe_missing_label_columns(mocker) -> None:
    def modify_template(template) -> None:
        template["outputColumns"] = {"sortType": "CUSTOM", "customOrder": ["q1..4"]}

    _, exception = write_jsons_and_run(mocker, modify_template=modify_template)
    assert str(exception) == "No Error"


def test_invalid_sort_type(mocker) -> None:
    def modify_template(template) -> None:
        template["outputColumns"] = {"sortType": "ABC"}

    _, exception = write_jsons_and_run(mocker, modify_template=modify_template)
    # Updated to match custom exception message format
    assert "Invalid template JSON" in str(exception)


def test_invalid_sort_order(mocker) -> None:
    def modify_template(template) -> None:
        template["outputColumns"] = {"sortOrder": "ABC"}

    _, exception = write_jsons_and_run(mocker, modify_template=modify_template)
    # Updated to match custom exception message format
    assert "Invalid template JSON" in str(exception)
