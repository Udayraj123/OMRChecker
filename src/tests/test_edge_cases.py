import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils.image import ImageUtils
from src.tests.test_samples.sample2.boilerplate import (
    CONFIG_BOILERPLATE,
    TEMPLATE_BOILERPLATE,
)
from src.tests.utils import (
    generate_write_jsons_and_run,
    remove_file,
    run_entry_point,
    setup_mocker_patches,
)

from time import strftime, localtime

from freezegun import freeze_time
from src.tests.utils import FROZEN_TIMESTAMP

with freeze_time(FROZEN_TIMESTAMP):
    TIME_NOW_HRS = strftime("%I%p", localtime())

CURRENT_DIR = Path("src/tests")
BASE_SAMPLE_PATH = CURRENT_DIR.joinpath("test_samples", "sample2")
BASE_RESULTS_CSV_PATH = os.path.join(
    "outputs", BASE_SAMPLE_PATH, "Results", f"Results_{TIME_NOW_HRS}.csv"
)
BASE_MULTIMARKED_CSV_PATH = os.path.join(
    "outputs", BASE_SAMPLE_PATH, "Manual", "MultiMarkedFiles.csv"
)


def run_sample(mocker, input_path):
    setup_mocker_patches(mocker)
    output_dir = os.path.join("outputs", input_path)
    run_entry_point(input_path, output_dir)


def extract_output_data(path):
    output_data = pd.read_csv(path, keep_default_na=False)
    return output_data


write_jsons_and_run = generate_write_jsons_and_run(
    run_sample,
    sample_path=BASE_SAMPLE_PATH,
    template_boilerplate=TEMPLATE_BOILERPLATE,
    config_boilerplate=CONFIG_BOILERPLATE,
)


def test_config_low_dimensions(mocker):
    def modify_config(config):
        config["dimensions"]["processing_height"] = 1000
        config["dimensions"]["processing_width"] = 1000

    exception = write_jsons_and_run(mocker, modify_config=modify_config)

    assert str(exception) == "No Error"


def test_different_bubble_dimensions(mocker):
    # Prevent appending to output csv:
    remove_file(BASE_RESULTS_CSV_PATH)
    remove_file(BASE_MULTIMARKED_CSV_PATH)

    exception = write_jsons_and_run(mocker)
    assert str(exception) == "No Error"
    original_output_data = extract_output_data(BASE_RESULTS_CSV_PATH)

    def modify_template(template):
        # Incorrect global bubble size
        template["bubbleDimensions"] = [5, 5]
        # Correct bubble size for MCQBlock1a1
        template["fieldBlocks"]["MCQBlock1a1"]["bubbleDimensions"] = [32, 32]
        # Incorrect bubble size for MCQBlock1a11
        template["fieldBlocks"]["MCQBlock1a11"]["bubbleDimensions"] = [10, 10]

    remove_file(BASE_RESULTS_CSV_PATH)
    remove_file(BASE_MULTIMARKED_CSV_PATH)
    exception = write_jsons_and_run(mocker, modify_template=modify_template)
    assert str(exception) == "No Error"

    results_output_data = extract_output_data(BASE_RESULTS_CSV_PATH)

    assert results_output_data.empty

    output_data = extract_output_data(BASE_MULTIMARKED_CSV_PATH)

    equal_columns = [f"q{i}" for i in range(1, 18)]
    assert (
        output_data[equal_columns].iloc[0].to_list()
        == original_output_data[equal_columns].iloc[0].to_list()
    )

    unequal_columns = [f"q{i}" for i in range(168, 185)]
    assert not (
        output_data[unequal_columns].iloc[0].to_list()
        == original_output_data[unequal_columns].iloc[0].to_list()
    )


def test_watermark_disabled():
    blank = np.ones((400, 400, 3), dtype=np.uint8) * 255
    res = ImageUtils.apply_watermark(blank, {"enabled": False})
    assert np.array_equal(blank, res)


def test_watermark_grayscale_shape():
    gray = np.ones((500, 400), dtype=np.uint8) * 200
    res = ImageUtils.apply_watermark(gray, {"enabled": True, "text": "TEST"})
    assert res.shape == (500, 400)
    assert len(res.shape) == 2


def test_watermark_positions():
    shape = (600, 500)
    text_size = (150, 25)
    baseline = 5
    margin = 20
    positions = ["bottom-right", "top-left", "top-right", "bottom-left", "center"]
    for pos in positions:
        x, y = ImageUtils.calculate_watermark_position(
            shape, text_size, position=pos, margin=margin, baseline=baseline
        )
        assert margin <= x <= shape[1] - margin - text_size[0]
        assert margin + text_size[1] <= y <= shape[0] - margin - baseline

    # Custom pixel coordinates
    cx, cy = ImageUtils.calculate_watermark_position(
        shape, text_size, position=[80, 120], margin=margin, baseline=baseline
    )
    assert (cx, cy) == (80, 120)


def test_watermark_overflow_raises_error():
    img = np.zeros((300, 200, 3), dtype=np.uint8)
    long_text = "VERY LONG WATERMARK TEXT THAT EXCEEDS IMAGE WIDTH SIGNIFICANTLY"
    with pytest.raises(ValueError):
        ImageUtils.apply_watermark(
            img,
            {
                "enabled": True,
                "text": long_text,
                "font_scale": 1.0,
                "margin": 20,
            },
        )

    # Out-of-bounds custom pixel position
    with pytest.raises(ValueError):
        ImageUtils.apply_watermark(
            img,
            {
                "enabled": True,
                "text": "TEST",
                "position": [190, 295],
                "font_scale": 1.0,
            },
        )


def test_watermark_format_variables():
    img = np.zeros((200, 500, 3), dtype=np.uint8)
    res = ImageUtils.apply_watermark(
        img,
        {
            "enabled": True,
            "text": "Roll: {Roll} | Q1: {q1}",
            "position": [30, 100],
            "opacity": 1.0,
            "color": (0, 255, 0),
        },
        omr_response={"Roll": "E101", "q1": "B"},
    )
    assert np.any(res > 0)


def test_watermark_pipeline_integration(mocker):
    def modify_config(config):
        config["watermark"] = {
            "enabled": True,
            "text": "Processed using OMRChecker - Q1: {q1}",
            "position": "bottom-right",
            "margin": 20,
            "opacity": 0.5,
        }

    exception = write_jsons_and_run(mocker, modify_config=modify_config)
    assert str(exception) == "No Error"
