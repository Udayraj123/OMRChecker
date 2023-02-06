import json
import os
from copy import deepcopy

from freezegun import freeze_time

from main import entry_point_for_args

FROZEN_TIMESTAMP = "1970-01-01"


def setup_mocker_patches(mocker):
    mock_imshow = mocker.patch("cv2.imshow")
    mock_imshow.return_value = True

    mock_destroy_all_windows = mocker.patch("cv2.destroyAllWindows")
    mock_destroy_all_windows.return_value = True

    mock_wait_key = mocker.patch("cv2.waitKey")
    mock_wait_key.return_value = ord("q")


def run_entry_point(input_path, output_dir):
    args = {
        "input_paths": [input_path],
        "output_dir": output_dir,
        "autoAlign": False,
        "setLayout": False,
        "silent": True,
    }
    with freeze_time(FROZEN_TIMESTAMP):
        entry_point_for_args(args)


def write_modified(modify_content, boilerplate, sample_json_path):
    if boilerplate is None:
        return

    content = deepcopy(boilerplate)

    if modify_content is not None:
        returned_value = modify_content(content)
        if returned_value is not None:
            content = returned_value

    with open(sample_json_path, "w") as f:
        json.dump(content, f)


def remove_modified(sample_json_path):
    if os.path.exists(sample_json_path):
        os.remove(sample_json_path)


def generate_write_jsons_and_run(
    run_sample,
    sample_path,
    template_boilerplate=None,
    config_boilerplate=None,
    evaluation_boilerplate=None,
):
    def write_jsons_and_run(
        mocker,
        modify_template=None,
        modify_config=None,
        modify_evaluation=None,
    ):
        sample_template_path, sample_config_path, sample_evaluation_path = (
            sample_path.joinpath("template.json"),
            sample_path.joinpath("config.json"),
            sample_path.joinpath("evaluation.json"),
        )
        write_modified(modify_template, template_boilerplate, sample_template_path)
        write_modified(modify_config, config_boilerplate, sample_config_path)
        write_modified(
            modify_evaluation, evaluation_boilerplate, sample_evaluation_path
        )

        exception = "No Error"
        try:
            run_sample(mocker, sample_path)
        except Exception as e:
            exception = e

        remove_modified(sample_template_path)
        remove_modified(sample_config_path)
        remove_modified(sample_evaluation_path)

        return exception

    return write_jsons_and_run
