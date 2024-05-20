from src.tests.utils import assert_image_snapshot


def test_run_omr_marker_mobile(run_sample, mocker, snapshot):
    sample_outputs = run_sample(mocker, "1-mobile-camera")
    assert snapshot == sample_outputs


def test_run_omr_marker(run_sample, mocker, snapshot, image_snapshot):
    sample_outputs = run_sample(mocker, "2-omr-marker")

    assert snapshot == sample_outputs
    # Check image snapshots
    # Note: image snapshots are updated using the --image-snapshot-update flag
    output_relative_dir = "outputs/2-omr-marker/ScanBatch1/CheckedOMRs"
    assert_image_snapshot(output_relative_dir, "camscanner-1.jpg", image_snapshot)
    assert_image_snapshot(
        output_relative_dir, "colored/camscanner-1.jpg", image_snapshot
    )


def test_run_bonus_marking(run_sample, mocker, snapshot):
    sample_outputs = run_sample(mocker, "3-answer-key/bonus-marking")
    assert snapshot == sample_outputs


def test_run_bonus_marking_grouping(run_sample, mocker, snapshot, image_snapshot):
    sample_path = "3-answer-key/bonus-marking-grouping"
    sample_outputs = run_sample(mocker, sample_path)

    assert snapshot == sample_outputs
    # Check image snapshots
    output_relative_dir = f"outputs/{sample_path}/CheckedOMRs"
    assert_image_snapshot(
        output_relative_dir, "IMG_20201116_143512.jpg", image_snapshot
    )
    assert_image_snapshot(
        output_relative_dir, "colored/IMG_20201116_143512.jpg", image_snapshot
    )


def test_run_answer_key_using_csv(run_sample, mocker, snapshot):
    sample_outputs = run_sample(mocker, "3-answer-key/using-csv")
    assert snapshot == sample_outputs


def test_run_answer_key_using_image(run_sample, mocker, snapshot):
    sample_outputs = run_sample(mocker, "3-answer-key/using-image")
    assert snapshot == sample_outputs


def test_run_answer_key_using_image_grouping(
    run_sample, mocker, snapshot, image_snapshot
):
    sample_path = "3-answer-key/using-image-grouping"
    sample_outputs = run_sample(mocker, sample_path)
    assert snapshot == sample_outputs
    output_relative_dir = f"outputs/{sample_path}/CheckedOMRs"
    assert_image_snapshot(output_relative_dir, "angle-1.jpg", image_snapshot)
    assert_image_snapshot(output_relative_dir, "colored/angle-1.jpg", image_snapshot)


def test_run_answer_key_weighted_answers(run_sample, mocker, snapshot):
    sample_outputs = run_sample(mocker, "3-answer-key/weighted-answers")
    assert snapshot == sample_outputs


def test_run_crop_four_dots(run_sample, mocker, snapshot):
    sample_outputs = run_sample(mocker, "experimental/1-timelines-and-dots/four-dots")
    assert snapshot == sample_outputs


def test_run_crop_two_dots_one_line(run_sample, mocker, snapshot):
    sample_outputs = run_sample(mocker, "experimental/1-timelines-and-dots/four-dots")
    assert snapshot == sample_outputs


def test_run_two_lines(run_sample, mocker, snapshot):
    sample_outputs = run_sample(mocker, "experimental/1-timelines-and-dots/two-lines")
    assert snapshot == sample_outputs


def test_run_template_shifts(run_sample, mocker, snapshot):
    sample_outputs = run_sample(mocker, "experimental/2-template-shifts")
    assert snapshot == sample_outputs


def test_run_feature_based_alignment(run_sample, mocker, snapshot):
    sample_outputs = run_sample(mocker, "experimental/3-feature-based-alignment")
    assert snapshot == sample_outputs


def test_run_community_Antibodyy(run_sample, mocker, snapshot):
    sample_outputs = run_sample(mocker, "community/Antibodyy")
    assert snapshot == sample_outputs


def test_run_community_ibrahimkilic(run_sample, mocker, snapshot):
    sample_outputs = run_sample(mocker, "community/ibrahimkilic")
    assert snapshot == sample_outputs


def test_run_community_Sandeep_1507(run_sample, mocker, snapshot):
    sample_outputs = run_sample(mocker, "community/Sandeep-1507")
    assert snapshot == sample_outputs


def test_run_community_Shamanth(run_sample, mocker, snapshot):
    sample_outputs = run_sample(mocker, "community/Shamanth")
    assert snapshot == sample_outputs


def test_run_community_UmarFarootAPS(run_sample, mocker, snapshot):
    sample_outputs = run_sample(mocker, "community/UmarFarootAPS")
    assert snapshot == sample_outputs
