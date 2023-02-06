import os
from pathlib import Path

from src.tests.test_samples.sample1.boilerplate import TEMPLATE_BOILERPLATE
from src.tests.utils import (
    generate_write_jsons_and_run,
    run_entry_point,
    setup_mocker_patches,
)

FROZEN_TIMESTAMP = "1970-01-01"
CURRENT_DIR = Path("src/tests")
BASE_SAMPLE_PATH = CURRENT_DIR.joinpath("test_samples", "sample1")
BASE_SAMPLE_TEMPLATE_PATH = BASE_SAMPLE_PATH.joinpath("template.json")


def run_sample(mocker, input_path):
    setup_mocker_patches(mocker)
    output_dir = os.path.join("outputs", input_path)
    run_entry_point(input_path, output_dir)


write_jsons_and_run = generate_write_jsons_and_run(
    run_sample,
    sample_path=BASE_SAMPLE_PATH,
    template_boilerplate=TEMPLATE_BOILERPLATE,
)


def test_no_input_dir(mocker):
    try:
        run_sample(mocker, "X")
    except Exception as e:
        assert str(e) == "Given input directory does not exist: 'X'"


def test_no_template(mocker):
    if os.path.exists(BASE_SAMPLE_TEMPLATE_PATH):
        os.remove(BASE_SAMPLE_TEMPLATE_PATH)
    try:
        run_sample(mocker, BASE_SAMPLE_PATH)
    except Exception as e:
        assert (
            str(e)
            == "No template file found in the directory tree of src/tests/test_samples/sample1"
        )


def test_empty_template(mocker):
    def modify_template(_):
        return {}

    exception = write_jsons_and_run(mocker, modify_template=modify_template)
    assert (
        str(exception)
        == f"Provided Template JSON is Invalid: '{BASE_SAMPLE_TEMPLATE_PATH}'"
    )


def test_invalid_field_type(mocker):
    def modify_template(template):
        template["fieldBlocks"]["MCQ_Block_1"]["fieldType"] = "X"

    exception = write_jsons_and_run(mocker, modify_template=modify_template)
    assert (
        str(exception)
        == f"Provided Template JSON is Invalid: '{BASE_SAMPLE_TEMPLATE_PATH}'"
    )


def test_overflow_labels(mocker):
    def modify_template(template):
        template["fieldBlocks"]["MCQ_Block_1"]["fieldLabels"] = ["q1..100"]

    exception = write_jsons_and_run(mocker, modify_template=modify_template)
    assert (
        str(exception)
        == "Overflowing field block 'MCQ_Block_1' with origin [65, 60] and dimensions [189, 5173] in template with dimensions [300, 400]"
    )


def test_overflow_safe_dimensions(mocker):
    def modify_template(template):
        template["pageDimensions"] = [255, 400]

    exception = write_jsons_and_run(mocker, modify_template=modify_template)
    assert str(exception) == "No Error"


def test_field_strings_overlap(mocker):
    def modify_template(template):
        template["fieldBlocks"] = {
            **template["fieldBlocks"],
            "New_Block": {
                **template["fieldBlocks"]["MCQ_Block_1"],
                "fieldLabels": ["q5"],
            },
        }

    exception = write_jsons_and_run(mocker, modify_template=modify_template)
    assert str(exception) == (
        "The field strings for field block New_Block overlap with other existing fields"
    )


def test_custom_label_strings_overlap_single(mocker):
    def modify_template(template):
        template["customLabels"] = {
            "label1": ["q1..2", "q2..3"],
        }

    exception = write_jsons_and_run(mocker, modify_template=modify_template)
    assert (
        str(exception)
        == "Given field string 'q2..3' has overlapping field(s) with other fields in 'Custom Label: label1': ['q1..2', 'q2..3']"
    )


def test_custom_label_strings_overlap_multiple(mocker):
    def modify_template(template):
        template["customLabels"] = {
            "label1": ["q1..2"],
            "label2": ["q2..3"],
        }

    exception = write_jsons_and_run(mocker, modify_template=modify_template)
    assert (
        str(exception)
        == "The field strings for custom label 'label2' overlap with other existing custom labels"
    )


def test_missing_field_block_labels(mocker):
    def modify_template(template):
        template["customLabels"] = {"Combined": ["qX", "qY"]}

    exception = write_jsons_and_run(mocker, modify_template=modify_template)
    assert (
        str(exception)
        == "Missing field block label(s) in the given template for ['qX', 'qY'] from 'Combined'"
    )


def test_missing_output_columns(mocker):
    def modify_template(template):
        template["outputColumns"] = ["qX", "q1..5"]

    exception = write_jsons_and_run(mocker, modify_template=modify_template)
    assert str(exception) == (
        "Some columns are missing in the field blocks for the given output columns"
    )


def test_safe_missing_label_columns(mocker):
    def modify_template(template):
        template["outputColumns"] = ["q1..4"]

    exception = write_jsons_and_run(mocker, modify_template=modify_template)
    assert str(exception) == "No Error"
