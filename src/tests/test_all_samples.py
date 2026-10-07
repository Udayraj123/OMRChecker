import os
import re
import shutil
from glob import glob

import pytest

from src.tests.utils import run_entry_point, setup_mocker_patches

# These samples process PDFs that sit next to their template, but the committed
# snapshots were generated before those outputs were recorded; the
# community/celin-mampilly entry was generated on a Windows host and embeds
# backslashes. They are stale for every platform, so they are tracked as
# expected failures instead of failing each run. strict=True turns the marker
# into a failure as soon as the snapshots are regenerated, which forces the
# marker to be removed with the fix. See issues #316 and #325.
stale_snapshot = pytest.mark.xfail(
    reason=(
        "Committed snapshot is stale (missing PDF outputs / Windows "
        "separators); see #316 and #325."
    ),
    strict=True,
)


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
    """Collect the generated CSVs keyed by their path relative to the output dir.

    The key has to identify the same file on every host and at every run time,
    otherwise the committed snapshot cannot match:

    - separators are normalised to ``/`` so Windows and POSIX agree;
    - the run-hour the Results file name embeds (``Results_05AM.csv`` comes from
      ``strftime("%I%p", localtime())``) is replaced by a fixed placeholder, so
      the comparison no longer depends on when the test happened to run.
    """
    sample_outputs = {}
    for _dir, _subdir, _files in os.walk(output_dir):
        for file in glob(os.path.join(_dir, EXT)):
            relative_path = os.path.relpath(file, output_dir)
            relative_path = relative_path.replace(os.sep, "/")
            relative_path = re.sub(
                r"Results_\d{1,2}[AP]M\.csv", "Results_HH.csv", relative_path
            )
            sample_outputs[relative_path] = read_file(file)
    return sample_outputs


def test_run_answer_key_using_csv(mocker, snapshot):
    sample_outputs = run_sample(mocker, "answer-key/using-csv")
    assert snapshot == sample_outputs


def test_run_answer_key_weighted_answers(mocker, snapshot):
    sample_outputs = run_sample(mocker, "answer-key/weighted-answers")
    assert snapshot == sample_outputs


@stale_snapshot
def test_run_sample1(mocker, snapshot):
    sample_outputs = run_sample(mocker, "sample1")
    assert snapshot == sample_outputs


@stale_snapshot
def test_run_sample2(mocker, snapshot):
    sample_outputs = run_sample(mocker, "sample2")
    assert snapshot == sample_outputs


def test_run_sample3(mocker, snapshot):
    sample_outputs = run_sample(mocker, "sample3")
    assert snapshot == sample_outputs


@stale_snapshot
def test_run_sample4(mocker, snapshot):
    sample_outputs = run_sample(mocker, "sample4")
    assert snapshot == sample_outputs


def test_run_sample5(mocker, snapshot):
    sample_outputs = run_sample(mocker, "sample5")
    assert snapshot == sample_outputs


def test_run_sample6(mocker, snapshot):
    sample_outputs = run_sample(mocker, "sample6")
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


def test_run_community_UPSC_mock(mocker, snapshot):
    sample_outputs = run_sample(mocker, "community/UPSC-mock")
    assert snapshot == sample_outputs


@stale_snapshot
def test_run_community_celin_mampilly(mocker, snapshot):
    sample_outputs = run_sample(mocker, "community/celin-mampilly")
    assert snapshot == sample_outputs
