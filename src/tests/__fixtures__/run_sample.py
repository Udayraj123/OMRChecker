import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.tests.utils import (
    extract_all_csv_outputs,
    run_entry_point,
    setup_mocker_patches,
)


def run_sample_parser_hook(parser) -> None:
    parser.addoption(
        "--keep-outputs", action="store_true", help="Keep outputs after running sample"
    )


def run_sample_core(mocker, sample_path) -> tuple[dict[str, str], Callable]:
    setup_mocker_patches(mocker)

    input_path = Path("samples", sample_path)
    output_dir = Path("outputs", sample_path)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    run_entry_point(input_path, output_dir)

    sample_outputs = extract_all_csv_outputs(output_dir)

    def remove_sample_output_dir() -> None:
        if output_dir.exists():
            shutil.rmtree(output_dir)

    mocker.resetall()

    return sample_outputs, remove_sample_output_dir


def run_sample_fixture(request) -> Callable:
    def run_sample(*args, **kwargs) -> dict[str, Any]:
        config = request.config
        keep_outputs = config.getoption("--keep-outputs")
        sample_outputs, remove_sample_output_dir = run_sample_core(*args, **kwargs)

        # https://docs.pytest.org/en/6.2.x/fixture.html#adding-finalizers-directly
        if not keep_outputs:
            request.addfinalizer(remove_sample_output_dir)
        return sample_outputs

    return run_sample
