import pytest

from src.tests.common_utils import assert_image_snapshot


@pytest.mark.sample_1_mobile_camera
def test_run_omr_marker_mobile(run_sample, mocker, snapshot) -> None:
    """Test using sample: 1-mobile-camera"""
    sample_outputs = run_sample(mocker, "1-mobile-camera")
    assert snapshot == sample_outputs


@pytest.mark.sample_2_omr_marker
def test_run_omr_marker(run_sample, mocker, snapshot, image_snapshot) -> None:
    """Test using sample: 2-omr-marker"""
    sample_outputs = run_sample(mocker, "2-omr-marker")

    assert snapshot == sample_outputs
    # Check image snapshots
    # Note: image snapshots are updated using the --image-snapshot-update flag
    output_relative_dir = "outputs/2-omr-marker/ScanBatch1/CheckedOMRs"
    assert_image_snapshot(output_relative_dir, "camscanner-1.jpg", image_snapshot)
    assert_image_snapshot(
        output_relative_dir, "colored/camscanner-1.jpg", image_snapshot
    )


@pytest.mark.sample_3_answer_key_bonus_marking
def test_run_bonus_marking(run_sample, mocker, snapshot) -> None:
    """Test using sample: 3-answer-key/bonus-marking"""
    sample_outputs = run_sample(mocker, "3-answer-key/bonus-marking")
    assert snapshot == sample_outputs


@pytest.mark.sample_3_answer_key_bonus_marking_grouping
def test_run_bonus_marking_grouping(
    run_sample, mocker, snapshot, image_snapshot
) -> None:
    """Test using sample: 3-answer-key/bonus-marking-grouping"""
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


@pytest.mark.sample_3_answer_key_using_csv
def test_run_answer_key_using_csv(run_sample, mocker, snapshot) -> None:
    """Test using sample: 3-answer-key/using-csv"""
    sample_outputs = run_sample(mocker, "3-answer-key/using-csv")
    assert snapshot == sample_outputs


@pytest.mark.sample_3_answer_key_using_image
def test_run_answer_key_using_image(run_sample, mocker, snapshot) -> None:
    """Test using sample: 3-answer-key/using-image"""
    sample_outputs = run_sample(mocker, "3-answer-key/using-image")
    assert snapshot == sample_outputs


@pytest.mark.sample_3_answer_key_using_image_grouping
def test_run_answer_key_using_image_grouping(
    run_sample, mocker, snapshot, image_snapshot
) -> None:
    """Test using sample: 3-answer-key/using-image-grouping"""
    sample_path = "3-answer-key/using-image-grouping"
    sample_outputs = run_sample(mocker, sample_path)
    assert snapshot == sample_outputs
    output_relative_dir = f"outputs/{sample_path}/CheckedOMRs"
    assert_image_snapshot(output_relative_dir, "angle-1.jpg", image_snapshot)
    assert_image_snapshot(output_relative_dir, "colored/angle-1.jpg", image_snapshot)


@pytest.mark.sample_3_answer_key_weighted_answers
def test_run_answer_key_weighted_answers(run_sample, mocker, snapshot) -> None:
    """Test using sample: 3-answer-key/weighted-answers"""
    sample_outputs = run_sample(mocker, "3-answer-key/weighted-answers")
    assert snapshot == sample_outputs


@pytest.mark.sample_experimental_1_timelines_and_dots_four_dots
def test_run_crop_four_dots(run_sample, mocker, snapshot) -> None:
    """Test using sample: experimental/1-timelines-and-dots/four-dots"""
    sample_outputs = run_sample(mocker, "experimental/1-timelines-and-dots/four-dots")
    assert snapshot == sample_outputs


@pytest.mark.sample_experimental_1_timelines_and_dots_four_dots
def test_run_crop_two_dots_one_line(run_sample, mocker, snapshot) -> None:
    """Test using sample: experimental/1-timelines-and-dots/four-dots (shared with test_run_crop_four_dots)"""
    sample_outputs = run_sample(mocker, "experimental/1-timelines-and-dots/four-dots")
    assert snapshot == sample_outputs


@pytest.mark.sample_experimental_1_timelines_and_dots_two_lines
def test_run_two_lines(run_sample, mocker, snapshot) -> None:
    """Test using sample: experimental/1-timelines-and-dots/two-lines"""
    sample_outputs = run_sample(mocker, "experimental/1-timelines-and-dots/two-lines")
    assert snapshot == sample_outputs


@pytest.mark.sample_experimental_2_template_shifts
def test_run_template_shifts(run_sample, mocker, snapshot) -> None:
    """Test using sample: experimental/2-template-shifts"""
    sample_outputs = run_sample(mocker, "experimental/2-template-shifts")
    assert snapshot == sample_outputs


@pytest.mark.sample_experimental_3_feature_based_alignment
def test_run_feature_based_alignment(run_sample, mocker, snapshot) -> None:
    """Test using sample: experimental/3-feature-based-alignment"""
    sample_outputs = run_sample(mocker, "experimental/3-feature-based-alignment")
    assert snapshot == sample_outputs


@pytest.mark.sample_community_Antibodyy
def test_run_community_Antibodyy(run_sample, mocker, snapshot) -> None:
    """Test using sample: community/Antibodyy"""
    sample_outputs = run_sample(mocker, "community/Antibodyy")
    assert snapshot == sample_outputs


@pytest.mark.sample_community_ibrahimkilic
def test_run_community_ibrahimkilic(run_sample, mocker, snapshot) -> None:
    """Test using sample: community/ibrahimkilic"""
    sample_outputs = run_sample(mocker, "community/ibrahimkilic")
    assert snapshot == sample_outputs


@pytest.mark.sample_community_Sandeep_1507
def test_run_community_Sandeep_1507(run_sample, mocker, snapshot) -> None:
    """Test using sample: community/Sandeep-1507"""
    sample_outputs = run_sample(mocker, "community/Sandeep-1507")
    assert snapshot == sample_outputs


@pytest.mark.sample_community_Shamanth
def test_run_community_Shamanth(run_sample, mocker, snapshot) -> None:
    """Test using sample: community/Shamanth"""
    sample_outputs = run_sample(mocker, "community/Shamanth")
    assert snapshot == sample_outputs


@pytest.mark.sample_community_UmarFarootAPS
def test_run_community_UmarFarootAPS(run_sample, mocker, snapshot) -> None:
    """Test using sample: community/UmarFarootAPS"""
    sample_outputs = run_sample(mocker, "community/UmarFarootAPS")
    assert snapshot == sample_outputs


@pytest.mark.sample_community_JoyChopra1298
def test_run_community_JoyChopra1298(run_sample, mocker, snapshot) -> None:
    """Test using sample: community/JoyChopra1298"""
    sample_outputs = run_sample(mocker, "community/JoyChopra1298")
    assert snapshot == sample_outputs
