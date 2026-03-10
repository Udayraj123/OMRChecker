# ref: https://github.com/bmihelac/pytest-image-snapshot/blob/main/pytest_image_snapshot.py
import shutil
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike


def copy_image_to_snapshot(source_path, destination_path) -> None:
    shutil.copyfile(source_path, destination_path)
    # cv2.imwrite(str(image_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def open_image(image_path: Path) -> MatLike:
    if not image_path.exists():
        msg = f"Image not found at: {image_path}."
        raise Exception(msg)

    return cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)


def image_snapshot_parser_hook(parser) -> None:
    parser.addoption(
        "--image-snapshot-update", action="store_true", help="Update image snapshots"
    )
    parser.addoption(
        "--show-images-on-fail",
        action="store_true",
        help="Show image snapshots for failing tests",
    )


def resize_util(img, u_width=None, u_height=None) -> MatLike:
    h, w = img.shape[:2]

    if u_height is None:
        u_height = int(h * u_width / w)
    if u_width is None:
        u_width = int(w * u_height / h)
    if u_height == h and u_width == w:
        # No need to resize
        return img
    return cv2.resize(img, (int(u_width), int(u_height)))


def extend_to_match_size(img1, img2) -> tuple[MatLike, MatLike]:
    """Extend the smaller image to match the size of the larger one."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    max_width = max(w1, w2)
    max_height = max(h1, h2)
    return resize_util(img1, max_width, max_height), resize_util(
        img2, max_width, max_height
    )


def image_diff(image1, image2) -> bool:
    # TODO: try out cv2.diff() or channel-wise diff
    # Check if images have the same dimensions
    if image1.shape != image2.shape:
        msg = "Images must have the same dimensions."
        raise ValueError(msg)
    difference = np.absolute(image1 - image2)

    return difference.any()


def image_snapshot_fixture(request) -> Callable:
    def _image_snapshot(source_image_path, snapshot_path) -> None:
        source_image_path, snapshot_path = Path(source_image_path), Path(snapshot_path)
        config = request.config
        update_snapshots = config.getoption("--image-snapshot-update")
        show_on_fail = config.getoption("--show-images-on-fail")
        source_image = open_image(source_image_path)
        if not update_snapshots and snapshot_path.exists():
            current_snapshot = open_image(snapshot_path)
            # source_image, current_snapshot = extend_to_match_size(source_image, current_snapshot)
            diff_score = image_diff(source_image, current_snapshot)
            if diff_score != 0:
                if show_on_fail is True:
                    cv2.imshow("diff", cv2.subtract(source_image, current_snapshot))
                    cv2.imshow("current_snapshot", current_snapshot)
                    # TODO: fix code is not waiting despite unmocking mocker
                    close_all_on_wait_key()
                msg = f"{snapshot_path}: Snapshot does not match the image {source_image_path}"
                raise AssertionError(msg)
            return

        if snapshot_path.exists():
            pass

        copy_image_to_snapshot(source_image_path, snapshot_path)
        return

    return _image_snapshot


def close_all_on_wait_key(key="q") -> None:
    esc_key = 27
    while cv2.waitKey(1) & 0xFF not in {ord(key), esc_key}:
        pass
    cv2.destroyAllWindows()
