import os
import shutil

from src.tests.utils import (
    assert_image_snapshot,
    extract_all_csv_outputs,
    run_entry_point,
    setup_mocker_patches,
)


def run_sample(mocker, sample_path, keep_outputs=False):
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

    if not keep_outputs:
        # Remove outputs directory after running the sample
        remove_sample_output_dir()

    mocker.resetall()

    return sample_outputs, remove_sample_output_dir


def test_run_omr_marker_mobile(mocker, snapshot):
    sample_outputs, *_ = run_sample(mocker, "1-mobile-camera")
    assert snapshot == sample_outputs


def test_run_omr_marker(mocker, snapshot, image_snapshot):
    sample_outputs, remove_sample_output_dir = run_sample(
        mocker, "2-omr-marker", keep_outputs=True
    )

    assert snapshot == sample_outputs
    # Check image snapshots
    # Note: image snapshots are updated using the --image-snapshot-update flag
    output_relative_dir = "outputs/2-omr-marker/ScanBatch1/CheckedOMRs"
    assert_image_snapshot(output_relative_dir, "camscanner-1.jpg", image_snapshot)
    assert_image_snapshot(
        output_relative_dir, "colored/camscanner-1.jpg", image_snapshot
    )

    # TODO: make run_sample a fixture to do cleanup automatically at the end of the test
    remove_sample_output_dir()


def test_run_bonus_marking(mocker, snapshot):
    sample_outputs, *_ = run_sample(mocker, "3-answer-key/bonus-marking")
    assert snapshot == sample_outputs


def test_run_bonus_marking_grouping(mocker, snapshot, image_snapshot):
    sample_path = "3-answer-key/bonus-marking-grouping"
    sample_outputs, remove_sample_output_dir = run_sample(
        mocker, sample_path, keep_outputs=True
    )

    assert snapshot == sample_outputs
    # Check image snapshots
    output_relative_dir = f"outputs/{sample_path}/CheckedOMRs"
    assert_image_snapshot(
        output_relative_dir, "IMG_20201116_143512.jpg", image_snapshot
    )
    assert_image_snapshot(
        output_relative_dir, "colored/IMG_20201116_143512.jpg", image_snapshot
    )
    remove_sample_output_dir()


def test_run_answer_key_using_csv(mocker, snapshot):
    sample_outputs, *_ = run_sample(mocker, "3-answer-key/using-csv")
    assert snapshot == sample_outputs


def test_run_answer_key_using_image(mocker, snapshot):
    sample_outputs, *_ = run_sample(mocker, "3-answer-key/using-image")
    assert snapshot == sample_outputs


def test_run_answer_key_using_image_grouping(mocker, snapshot, image_snapshot):
    sample_path = "3-answer-key/using-image-grouping"
    sample_outputs, remove_sample_output_dir = run_sample(
        mocker, sample_path, keep_outputs=True
    )
    assert snapshot == sample_outputs
    output_relative_dir = f"outputs/{sample_path}/CheckedOMRs"
    assert_image_snapshot(output_relative_dir, "angle-1.jpg", image_snapshot)
    assert_image_snapshot(output_relative_dir, "colored/angle-1.jpg", image_snapshot)

    remove_sample_output_dir()


def test_run_answer_key_weighted_answers(mocker, snapshot):
    sample_outputs, *_ = run_sample(mocker, "3-answer-key/weighted-answers")
    assert snapshot == sample_outputs


def test_run_crop_four_dots(mocker, snapshot):
    sample_outputs, *_ = run_sample(
        mocker, "experimental/1-timelines-and-dots/four-dots"
    )
    assert snapshot == sample_outputs


def test_run_crop_two_dots_one_line(mocker, snapshot):
    sample_outputs, *_ = run_sample(
        mocker, "experimental/1-timelines-and-dots/four-dots"
    )
    assert snapshot == sample_outputs


def test_run_two_lines(mocker, snapshot):
    sample_outputs, *_ = run_sample(
        mocker, "experimental/1-timelines-and-dots/two-lines"
    )
    assert snapshot == sample_outputs


def test_run_template_shifts(mocker, snapshot):
    sample_outputs, *_ = run_sample(mocker, "experimental/2-template-shifts")
    assert snapshot == sample_outputs


def test_run_feature_based_alignment(mocker, snapshot):
    sample_outputs, *_ = run_sample(mocker, "experimental/3-feature-based-alignment")
    assert snapshot == sample_outputs


def test_run_community_Antibodyy(mocker, snapshot):
    sample_outputs, *_ = run_sample(mocker, "community/Antibodyy")
    assert snapshot == sample_outputs


def test_run_community_ibrahimkilic(mocker, snapshot):
    sample_outputs, *_ = run_sample(mocker, "community/ibrahimkilic")
    assert snapshot == sample_outputs


def test_run_community_Sandeep_1507(mocker, snapshot):
    sample_outputs, *_ = run_sample(mocker, "community/Sandeep-1507")
    assert snapshot == sample_outputs


def test_run_community_Shamanth(mocker, snapshot):
    sample_outputs, *_ = run_sample(mocker, "community/Shamanth")
    assert snapshot == sample_outputs


def test_run_community_UmarFarootAPS(mocker, snapshot):
    sample_outputs, *_ = run_sample(mocker, "community/UmarFarootAPS")
    assert snapshot == sample_outputs
