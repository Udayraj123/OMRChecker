import os

from src.tests.constants import (
    BASE_MULTIMARKED_CSV_PATH,
    BASE_RESULTS_CSV_PATH,
    BASE_SAMPLE_PATH,
)
from src.tests.test_samples.sample2.boilerplate import (
    CONFIG_BOILERPLATE,
    TEMPLATE_BOILERPLATE,
)
from src.tests.utils import (
    extract_output_data,
    generate_write_jsons_and_run,
    remove_file,
    run_entry_point,
    setup_mocker_patches,
)


def run_sample(mocker, input_path):
    setup_mocker_patches(mocker)
    output_dir = os.path.join("outputs", input_path)
    run_entry_point(input_path, output_dir)
    mocker.resetall()


write_jsons_and_run = generate_write_jsons_and_run(
    run_sample,
    sample_path=BASE_SAMPLE_PATH,
    template_boilerplate=TEMPLATE_BOILERPLATE,
    config_boilerplate=CONFIG_BOILERPLATE,
)


def test_config_low_dimensions_error_case(mocker, snapshot):
    def modify_template(template):
        template["preProcessors"][0]["options"]["processingImageShape"] = [
            1640 // 4,
            1332 // 4,
        ]
        template["preProcessors"][0]["options"]["markerDimensions"] = [20, 20]

    _, exception = write_jsons_and_run(mocker, modify_template=modify_template)
    assert str(exception) == snapshot


def test_config_low_dimensions_safe_case(mocker, snapshot):
    def modify_template(template):
        template["preProcessors"][0]["options"]["processingImageShape"] = [
            1640 // 2,
            1332 // 2,
        ]
        template["preProcessors"][0]["options"]["markerDimensions"] = [20, 20]

    sample_outputs, exception = write_jsons_and_run(
        mocker, modify_template=modify_template
    )
    assert str(exception) == "No Error"
    assert snapshot == sample_outputs


def test_different_bubble_dimensions(mocker):
    # Prevent appending to output csv:
    remove_file(BASE_RESULTS_CSV_PATH)
    remove_file(BASE_MULTIMARKED_CSV_PATH)

    _, exception = write_jsons_and_run(mocker)
    assert str(exception) == "No Error"

    original_output_data = extract_output_data(BASE_RESULTS_CSV_PATH)
    assert not original_output_data.empty
    assert len(original_output_data) == 1

    def modify_template(template):
        # Incorrect global bubble size
        template["bubbleDimensions"] = [5, 5]
        # Correct bubble size for MCQBlock1a1
        template["fieldBlocks"]["MCQBlock1a1"]["bubbleDimensions"] = [
            32,
            32,
        ]
        # Incorrect bubble size for MCQBlock1a11
        template["fieldBlocks"]["MCQBlock1a11"]["bubbleDimensions"] = [
            5,
            5,
        ]

    remove_file(BASE_RESULTS_CSV_PATH)
    remove_file(BASE_MULTIMARKED_CSV_PATH)
    _, exception = write_jsons_and_run(mocker, modify_template=modify_template)
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
