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
        "autoAlign": False,
        "debug": False,
        "input_paths": [input_path],
        "output_dir": output_dir,
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


def remove_file(path):
    if os.path.exists(path):
        os.remove(path)


def generate_write_jsons_and_run(
    run_sample,
    sample_path,
    template_boilerplate=None,
    config_boilerplate=None,
    evaluation_boilerplate=None,
):
    if (template_boilerplate or config_boilerplate or evaluation_boilerplate) is None:
        raise Exception(
            f"No boilerplates found. Provide atleast one boilerplate to write json."
        )

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

        remove_file(sample_template_path)
        remove_file(sample_config_path)
        remove_file(sample_evaluation_path)

        return exception

    return write_jsons_and_run
