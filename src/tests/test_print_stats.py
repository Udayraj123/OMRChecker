from types import SimpleNamespace

from src.entry import print_stats


def _tuning_config(show_image_level=0):
    return SimpleNamespace(outputs=SimpleNamespace(show_image_level=show_image_level))


def test_print_stats_does_not_divide_by_zero_when_no_file_was_processed(caplog):
    """
    When no image could be loaded, files_counter is 0.

    The rate calculations used to divide by it, which raised ZeroDivisionError
    and aborted the whole run after processing finished.
    """
    with caplog.at_level("INFO"):
        print_stats(start_time=0, files_counter=0, tuning_config=_tuning_config())

    assert "No image could be loaded" in caplog.text


def test_print_stats_still_reports_rate_for_processed_files(caplog):
    """The rate output must be unchanged when files were processed."""
    with caplog.at_level("INFO"):
        print_stats(start_time=0, files_counter=4, tuning_config=_tuning_config())

    assert "OMR Processing Rate" in caplog.text
    assert "seconds/OMR" in caplog.text


def test_print_stats_handles_zero_files_for_interactive_levels(caplog):
    """The non-default show_image_level branch must also survive zero files."""
    with caplog.at_level("INFO"):
        print_stats(
            start_time=0, files_counter=0, tuning_config=_tuning_config(2)
        )

    assert "Total script time" in caplog.text
