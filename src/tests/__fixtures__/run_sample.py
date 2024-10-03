import os
import shutil

from src.tests.utils import (
    extract_all_csv_outputs,
    run_entry_point,
    setup_mocker_patches,
)


def run_sample_parser_hook(parser):
    parser.addoption(
        "--keep-outputs", action="store_true", help="Keep outputs after running sample"
    )


def run_sample_core(mocker, sample_path):
    setup_mocker_patches(mocker)

    input_path = os.path.join("samples", sample_path)
    output_dir = os.path.join("outputs", sample_path)

    if os.path.exists(output_dir):
        print(f"Warning: output directory already exists: {output_dir}. Removing it.")
        shutil.rmtree(output_dir)

    run_entry_point(input_path, output_dir)

    sample_outputs = extract_all_csv_outputs(output_dir)

    def remove_sample_output_dir():
        if os.path.exists(output_dir):
            print(f"Note: removing output directory: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            print(f"Note: output directory already deleted: {output_dir}")

    mocker.resetall()

    return sample_outputs, remove_sample_output_dir


def run_sample_fixture(request):
    def run_sample(*args, **kwargs):
        config = request.config
        keep_outputs = config.getoption("--keep-outputs")
        sample_outputs, remove_sample_output_dir = run_sample_core(*args, **kwargs)

        # https://docs.pytest.org/en/6.2.x/fixture.html#adding-finalizers-directly
        if not keep_outputs:
            request.addfinalizer(remove_sample_output_dir)
        else:
            print("Note: keeping outputs of the test since --keep-outputs is passed")
        return sample_outputs

    return run_sample
