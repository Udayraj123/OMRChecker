import os
import shutil
from glob import glob

from src.tests.utils import run_entry_point, setup_mocker_patches


def read_file(path):
    with open(path) as file:
        return file.read()


def run_sample(mocker, sample_path):
    setup_mocker_patches(mocker)

    input_path = os.path.join("samples", sample_path)
    output_dir = os.path.join("outputs", sample_path)
    if os.path.exists(output_dir):
        print(
            f"Warning: output directory already exists: {output_dir}. This may affect the test execution."
        )

    run_entry_point(input_path, output_dir)

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


def test_run_omr_marker_mobile(mocker, snapshot):
    sample_outputs = run_sample(mocker, "1-mobile-camera")
    assert snapshot == sample_outputs


def test_run_omr_marker_scanned(mocker, snapshot):
    sample_outputs = run_sample(mocker, "2-omr-marker-scanned")
    assert snapshot == sample_outputs


def test_run_bonus_marking(mocker, snapshot):
    sample_outputs = run_sample(mocker, "3-answer-key/bonus-marking")
    assert snapshot == sample_outputs


def test_run_answer_key_using_csv(mocker, snapshot):
    sample_outputs = run_sample(mocker, "3-answer-key/using-csv")
    assert snapshot == sample_outputs


def test_run_answer_key_using_image(mocker, snapshot):
    sample_outputs = run_sample(mocker, "3-answer-key/using-image")
    assert snapshot == sample_outputs


def test_run_answer_key_weighted_answers(mocker, snapshot):
    sample_outputs = run_sample(mocker, "3-answer-key/weighted-answers")
    assert snapshot == sample_outputs


def test_run_crop_four_dots(mocker, snapshot):
    sample_outputs = run_sample(mocker, "experimental/1-timelines-and-dots/four-dots")
    assert snapshot == sample_outputs


def test_run_crop_two_dots_one_line(mocker, snapshot):
    sample_outputs = run_sample(mocker, "experimental/1-timelines-and-dots/four-dots")
    assert snapshot == sample_outputs


def test_run_two_lines(mocker, snapshot):
    sample_outputs = run_sample(mocker, "experimental/1-timelines-and-dots/two-lines")
    assert snapshot == sample_outputs


def test_run_template_shifts(mocker, snapshot):
    sample_outputs = run_sample(mocker, "experimental/2-template-shifts")
    assert snapshot == sample_outputs


def test_run_feature_based_alignment(mocker, snapshot):
    sample_outputs = run_sample(mocker, "experimental/3-feature-based-alignment")
    assert snapshot == sample_outputs


def test_run_community_Antibodyy(mocker, snapshot):
    sample_outputs = run_sample(mocker, "community/Antibodyy")
    assert snapshot == sample_outputs


def test_run_community_ibrahimkilic(mocker, snapshot):
    sample_outputs = run_sample(mocker, "community/ibrahimkilic")
    assert snapshot == sample_outputs


def test_run_community_Sandeep_1507(mocker, snapshot):
    sample_outputs = run_sample(mocker, "community/Sandeep-1507")
    assert snapshot == sample_outputs


def test_run_community_Shamanth(mocker, snapshot):
    sample_outputs = run_sample(mocker, "community/Shamanth")
    assert snapshot == sample_outputs


def test_run_community_UmarFarootAPS(mocker, snapshot):
    sample_outputs = run_sample(mocker, "community/UmarFarootAPS")
    assert snapshot == sample_outputs
