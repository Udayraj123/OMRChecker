import numpy as np

from dotmap import DotMap

from src.processors.manager import PROCESSOR_MANAGER

CropPage = PROCESSOR_MANAGER.processors["CropPage"]


def make_croppage(should_fail_if_page_not_found=True):
    fake_ops = type("FakeImageInstanceOps", (), {})()
    fake_ops.tuning_config = DotMap(outputs=DotMap(show_image_level=0))
    options = {"shouldFailIfPageNotFound": should_fail_if_page_not_found}
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


def test_crop_page_returns_image_when_skip_requested():
    croppage = make_croppage(should_fail_if_page_not_found=False)
    image = random_image()
    output = croppage.apply_filter(image, "test.png")
    assert output is not None
    assert output.shape == image.shape


def test_should_fail_if_page_not_found_defaults_to_true():
    fake_ops = type("FakeImageInstanceOps", (), {})()
    fake_ops.tuning_config = DotMap(outputs=DotMap(show_image_level=0))
    croppage = CropPage(
        options={},
        relative_dir=None,
        image_instance_ops=fake_ops,
    )
    assert croppage.should_fail_if_page_not_found is True
