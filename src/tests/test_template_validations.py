import json
import os
from copy import deepcopy
from pathlib import Path

from freezegun import freeze_time

from main import entry_point_for_args

FROZEN_TIMESTAMP = "1970-01-01"
CURRENT_DIR = Path("src/tests")
BASE_SAMPLE_PATH = CURRENT_DIR.joinpath("test_samples", "base")
BASE_SAMPLE_TEMPALTE_PATH = BASE_SAMPLE_PATH.joinpath("template.json")


def load_json(path, **rest):
    with open(path, "r") as f:
        loaded = json.load(f, **rest)
    return loaded


TEMPLATE_BOILERPLATE = load_json(BASE_SAMPLE_PATH.joinpath("template-boilerplate.json"))


def read_file(path):
    with open(path) as file:
        return file.read()


def run_sample(sample_path, mocker):
    mock_imshow = mocker.patch("cv2.imshow")
    mock_imshow.return_value = True

    mock_destroy_all_windows = mocker.patch("cv2.destroyAllWindows")
    mock_destroy_all_windows.return_value = True

    mock_wait_key = mocker.patch("cv2.waitKey")
    mock_wait_key.return_value = ord("q")

    input_path = os.path.join(f"{CURRENT_DIR}/test_samples", sample_path)
    output_dir = os.path.join("outputs", sample_path)

    args = {
        "input_paths": [input_path],
        "output_dir": output_dir,
        "autoAlign": False,
        "setLayout": False,
        "silent": True,
    }
    with freeze_time(FROZEN_TIMESTAMP):
        entry_point_for_args(args)


def test_no_input_dir(mocker):
    try:
        run_sample("X", mocker)
    except Exception as e:
        assert (
            str(e) == "Given input directory does not exist: 'src/tests/test_samples/X'"
        )


def test_no_template(mocker):
    if os.path.exists(BASE_SAMPLE_TEMPALTE_PATH):
        os.remove(BASE_SAMPLE_TEMPALTE_PATH)
    try:
        run_sample("base", mocker)
    except Exception as e:
        assert (
            str(e)
            == "No template file found in the directory tree of src/tests/test_samples/base"
        )


def run_template_modification(mocker, modify_template, error_message):
    template = deepcopy(TEMPLATE_BOILERPLATE)

    returned_value = modify_template(template)
    if returned_value is not None:
        template = returned_value

    with open(BASE_SAMPLE_TEMPALTE_PATH, "w") as f:
        json.dump(template, f)

    exception = "No Error"
    try:
        run_sample("base", mocker)
    except Exception as e:
        exception = e
    assert str(exception) == error_message

    os.remove(BASE_SAMPLE_TEMPALTE_PATH)

    return exception


def test_empty_template(mocker):
    def modify_template(_):
        return {}

    error_message = f"Provided Template JSON is Invalid: '{BASE_SAMPLE_TEMPALTE_PATH}'"
    run_template_modification(mocker, modify_template, error_message)


def test_invalid_field_type(mocker):
    def modify_template(template):
        template["fieldBlocks"]["MCQ_Block_1"]["fieldType"] = "X"

    error_message = f"Provided Template JSON is Invalid: '{BASE_SAMPLE_TEMPALTE_PATH}'"
    run_template_modification(mocker, modify_template, error_message)


def test_overflow_labels(mocker):
    def modify_template(template):
        template["fieldBlocks"]["MCQ_Block_1"]["fieldLabels"] = ["q1..100"]

    error_message = "Overflowing field block 'MCQ_Block_1' with origin [65, 60] and dimensions [189, 5173] in template with dimensions [300, 400]"
    run_template_modification(mocker, modify_template, error_message)


def test_overflow_safe_dimensions(mocker):
    def modify_template(template):
        template["pageDimensions"] = [255, 400]

    error_message = "No Error"
    run_template_modification(mocker, modify_template, error_message)


def test_field_strings_overlap(mocker):
    def modify_template(template):
        template["fieldBlocks"] = {
            **template["fieldBlocks"],
            "New_Block": {
                **template["fieldBlocks"]["MCQ_Block_1"],
                "fieldLabels": ["q5"],
            },
        }

    error_message = (
        "The field strings for field block New_Block overlap with other existing fields"
    )
    run_template_modification(mocker, modify_template, error_message)


def test_custom_label_strings_overlap_single(mocker):
    def modify_template(template):
        template["customLabels"] = {
            "label1": ["q1..2", "q2..3"],
        }

    error_message = "Given field string 'q2..3' has overlapping field(s) with other fields in 'Custom Label: label1': ['q1..2', 'q2..3']"
    run_template_modification(mocker, modify_template, error_message)


def test_custom_label_strings_overlap_multiple(mocker):
    def modify_template(template):
        template["customLabels"] = {
            "label1": ["q1..2"],
            "label2": ["q2..3"],
        }

    error_message = "The field strings for custom label 'label2' overlap with other existing custom labels"
    run_template_modification(mocker, modify_template, error_message)


def test_missing_field_block_labels(mocker):
    def modify_template(template):
        template["customLabels"] = {"Combined": ["qX", "qY"]}

    error_message = "Missing field block label(s) in the given template for ['qX', 'qY'] from 'Combined'"
    run_template_modification(mocker, modify_template, error_message)


def test_missing_output_columns(mocker):
    def modify_template(template):
        template["outputColumns"] = ["qX", "q1..5"]

    error_message = (
        "Some columns are missing in the field blocks for the given output columns"
    )
    run_template_modification(mocker, modify_template, error_message)


def test_safe_missing_label_columns(mocker):
    def modify_template(template):
        template["outputColumns"] = ["q1..4"]

    error_message = "No Error"
    run_template_modification(mocker, modify_template, error_message)


"""

Evaluation tests
"""
