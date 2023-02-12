import os
from pathlib import Path

from src.tests.test_samples.sample2.boilerplate import (
    CONFIG_BOILERPLATE,
    TEMPLATE_BOILERPLATE,
)
from src.tests.utils import (
    generate_write_jsons_and_run,
    run_entry_point,
    setup_mocker_patches,
)

FROZEN_TIMESTAMP = "1970-01-01"
CURRENT_DIR = Path("src/tests")
BASE_SAMPLE_PATH = CURRENT_DIR.joinpath("test_samples", "sample2")


def run_sample(mocker, input_path):
    setup_mocker_patches(mocker)
    output_dir = os.path.join("outputs", input_path)
    run_entry_point(input_path, output_dir)
    # sample_outputs = extract_sample_outputs(output_dir)


write_jsons_and_run = generate_write_jsons_and_run(
    run_sample,
    sample_path=BASE_SAMPLE_PATH,
    template_boilerplate=TEMPLATE_BOILERPLATE,
    config_boilerplate=CONFIG_BOILERPLATE,
)


def test_config_low_dimensions(mocker):
    def modify_config(config):
        config["dimensions"]["processing_height"] = 1000
        config["dimensions"]["processing_width"] = 1000

    exception = write_jsons_and_run(mocker, modify_config=modify_config)

    assert str(exception) == "No Error"
