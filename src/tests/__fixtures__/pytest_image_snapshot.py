# ref: https://github.com/bmihelac/pytest-image-snapshot/blob/main/pytest_image_snapshot.py
import os
import numpy as np
# import pytest
import cv2


def save_image(image_path, image):
    cv2.imwrite(str(image_path), image)


def open_image(image_path, base_width=800):
    if not os.path.exists(image_path):
        raise Exception(f"Image not found at: {image_path}.")

    image = cv2.imread(str(image_path))

    return resize_util(image, u_width=base_width)


# class ImageMismatchError(AssertionError):
#     """Exception raised when images do not match."""


def image_snapshot_parser_hook(parser):
    parser.addoption(
        "--image-snapshot-update", action="store_true", help="Update image snapshots"
    )


def resize_util(img, u_width=None, u_height=None):
    h, w = img.shape[:2]

    if u_height is None:
        u_height = int(h * u_width / w)
    if u_width is None:
        u_width = int(w * u_height / h)
    if u_height == h and u_width == w:
        # No need to resize
        return img
    return cv2.resize(img, (int(u_width), int(u_height)))


def extend_to_match_size(img1, img2):
    """
    Extend the smaller image to match the size of the larger one.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    max_width = max(w1, w2)
    max_height = max(h1, h2)
    return resize_util(img1, max_width, max_height), resize_util(
        img2, max_width, max_height
    )


def image_diff(image1, image2):
    # TODO: try out cv2.diff() or channel-wise diff
  # Check if images have the same dimensions
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions.")
    difference=np.absolute(image1 - image2)
    
    return difference.any()
    


def image_snapshot_fixture(request):
    def _image_snapshot(source_image_path, snapshot_path, threshold=0.05):
        config = request.config
        update_snapshots = config.getoption("--image-snapshot-update")
        source_image = open_image(source_image_path)

        if not update_snapshots and os.path.exists(snapshot_path):
            current_snapshot = open_image(snapshot_path)
            img1, img2 = extend_to_match_size(source_image, current_snapshot)
            diff_score = image_diff(img1, img2)
            if diff_score!=0:
                # if config.option.verbose > 2:
                #     diff.show(title="diff")
                #     if config.option.verbose > 3:
                #         current_snapshot.show(title="original")
                #         img.show(title="new")
                raise AssertionError(
                    f"Diff score {diff_score:.2f} > 0 for {source_image_path}: Image does not match the snapshot"
                )
            else:
                print(
                    f"Diff score {diff_score:.2f} == 0 passed for snapshot: {snapshot_path}"
                )
                return
        save_image(snapshot_path, source_image)
        return

    return _image_snapshot
