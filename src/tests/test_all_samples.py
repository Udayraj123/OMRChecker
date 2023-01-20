import os
import shutil
from glob import glob

from freezegun import freeze_time

from main import entry_point_for_args

FROZEN_TIMESTAMP = "1970-01-01"


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

    input_path = os.path.join("samples", sample_path)
    output_dir = os.path.join("outputs", sample_path)
    if os.path.exists(output_dir):
        print(
            f"Warning: output directory already exists: {output_dir}. This may affect the test execution."
        )
    args = {
        "input_paths": [input_path],
        "output_dir": output_dir,
        "autoAlign": False,
        "setLayout": False,
        "silent": True,
    }
    with freeze_time(FROZEN_TIMESTAMP):
        entry_point_for_args(args)

    sample_outputs = extract_sample_outputs(output_dir)

    print(f"Note: removing output directory: {output_dir}")
    shutil.rmtree(output_dir)

    return sample_outputs


EXT = "*.csv"


def extract_sample_outputs(output_dir):
    sample_outputs = {}
    for _dir, _subdir, _files in os.walk(output_dir):
        for file in glob(os.path.join(_dir, EXT)):
            relative_path = os.path.relpath(file, output_dir)
            sample_outputs[relative_path] = read_file(file)
    return sample_outputs


def test_run_sample1(mocker, snapshot):
    sample_outputs = run_sample("sample1", mocker)
    assert snapshot == sample_outputs


def test_run_sample2(mocker, snapshot):
    sample_outputs = run_sample("sample2", mocker)
    assert snapshot == sample_outputs


def test_run_sample3(mocker, snapshot):
    sample_outputs = run_sample("sample3", mocker)
    assert snapshot == sample_outputs


def test_run_sample4(mocker, snapshot):
    sample_outputs = run_sample("sample4", mocker)
    assert snapshot == sample_outputs


def test_run_sample5(mocker, snapshot):
    sample_outputs = run_sample("sample5", mocker)
    assert snapshot == sample_outputs


def test_run_sample6(mocker, snapshot):
    sample_outputs = run_sample("sample6", mocker)
    assert snapshot == sample_outputs


def test_run_community_Antibodyy(mocker, snapshot):
    sample_outputs = run_sample("community/Antibodyy", mocker)
    assert snapshot == sample_outputs


def test_run_community_ibrahimkilic(mocker, snapshot):
    sample_outputs = run_sample("community/ibrahimkilic", mocker)
    assert snapshot == sample_outputs


def test_run_community_Sandeep_1507(mocker, snapshot):
    sample_outputs = run_sample("community/Sandeep-1507", mocker)
    assert snapshot == sample_outputs


def test_run_community_Shamanth(mocker, snapshot):
    sample_outputs = run_sample("community/Shamanth", mocker)
    assert snapshot == sample_outputs


def test_run_community_UmarFarootAPS(mocker, snapshot):
    sample_outputs = run_sample("community/UmarFarootAPS", mocker)
    assert snapshot == sample_outputs


def test_run_community_UPSC_mock(mocker, snapshot):
    sample_outputs = run_sample("community/UPSC-mock", mocker)
    assert snapshot == sample_outputs
