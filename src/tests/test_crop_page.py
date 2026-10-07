import numpy as np

from dotmap import DotMap

from src.processors.manager import PROCESSOR_MANAGER

CropPage = PROCESSOR_MANAGER.processors["CropPage"]


def make_croppage(continue_on_page_not_found=None):
    fake_ops = type("FakeImageInstanceOps", (), {})()
    fake_ops.tuning_config = DotMap(outputs=DotMap(show_image_level=0))
    options = {}
    if continue_on_page_not_found is not None:
        options["continueOnPageNotFound"] = continue_on_page_not_found
    return CropPage(
        options=options,
        relative_dir=None,
        image_instance_ops=fake_ops,
    )


def random_image(shape=(100, 100, 3)):
    return np.random.randint(0, 256, shape, dtype=np.uint8)


def test_crop_page_fails_when_page_not_found_by_default():
    croppage = make_croppage()
    image = random_image()
    # Area is below MIN_PAGE_AREA_THRESHOLD, so no page can be detected.
    assert croppage.apply_filter(image, "test.png") is None


def test_crop_page_returns_original_image_when_continue_requested():
    croppage = make_croppage(continue_on_page_not_found=True)
    image = random_image()
    output = croppage.apply_filter(image, "test.png")
    assert output is not None
    # Downstream preprocessor must receive the exact input pixels, not the
    # normalized Gaussian-blurred working copy prepared for boundary detection.
    assert np.array_equal(output, image)


def test_crop_page_fallback_does_not_mutate_input():
    croppage = make_croppage(continue_on_page_not_found=True)
    image = random_image()
    before = image.copy()
    croppage.apply_filter(image, "test.png")
    assert np.array_equal(image, before)


def test_continue_on_page_not_found_defaults_to_false():
    croppage = make_croppage()
    assert croppage.continue_on_page_not_found is False


def test_explicit_false_keeps_failing():
    croppage = make_croppage(continue_on_page_not_found=False)
    image = random_image()
    assert croppage.apply_filter(image, "test.png") is None
