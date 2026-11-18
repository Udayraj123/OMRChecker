"""Regression tests for issue #317.

When a sample is graded from an answer key image, that key image must be
excluded from grading: it must not appear as a row in Results_*.csv and
must not be copied into CheckedOMRs/ as if it were a candidate sheet.
"""

from src.tests.test_all_samples import run_sample


def test_answer_key_image_is_not_graded(mocker):
    sample_outputs = run_sample(mocker, "community/UPSC-mock")

    # The answer key image must not appear in any generated CSV output,
    # neither as a Results row nor via a CheckedOMRs output path.
    for relative_path, content in sample_outputs.items():
        assert (
            "answer_key.jpg" not in content
        ), f"answer key image appears in {relative_path}"

    # The root Results file must hold no graded rows: the answer key was
    # the only sheet in the sample root, so nothing remains to grade there.
    # (It is absent entirely when no root sheet is processed.)
    for relative_path, content in sample_outputs.items():
        if relative_path.startswith("Results/Results_"):
            data_rows = [line for line in content.splitlines() if line.strip()]
            assert len(data_rows) == 1  # header only

    # Real candidate sheets in the sample must still be graded.
    assert any("angle-1.jpg" in content for content in sample_outputs.values())
