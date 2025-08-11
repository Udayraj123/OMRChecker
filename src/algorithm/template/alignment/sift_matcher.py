import cv2
import numpy as np

from src.algorithm.template.alignment.piecewise_affine_delaunay import (
    apply_piecewise_affine,
)
from src.algorithm.template.alignment.utils import show_displacement_overlay
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger
from src.utils.math import MathUtils

MIN_MATCH_COUNT = 10
FLANN_INDEX_KDTREE = 0


class SiftMatcher:
    sift = None
    flann = None

    def singleton_init() -> None:
        # Initiate SIFT detector
        SiftMatcher.sift = cv2.SIFT_create()

        index_params = {"algorithm": FLANN_INDEX_KDTREE, "trees": 5}
        search_params = {"checks": 50}

        # Initiate Flann matcher
        SiftMatcher.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # TODO: [later] add support for more matchers

    @staticmethod
    def get_matches(
        field_block_name, gray_image, alignment_image, max_displacement, config
    ):
        # TODO: add a small blur/denoise to get better matches

        # find the keypoints and descriptors with SIFT
        source_features, des1 = SiftMatcher.sift.detectAndCompute(alignment_image, None)
        destination_features, des2 = SiftMatcher.sift.detectAndCompute(gray_image, None)
        logger.debug(
            f"SIFT: {field_block_name}: alignment_image.shape", alignment_image.shape
        )
        logger.debug(f"SIFT: {field_block_name}: gray_image.shape", gray_image.shape)
        logger.debug(f"SIFT: {field_block_name}: alignment_image", alignment_image)
        logger.debug(f"SIFT: {field_block_name}: gray_image", gray_image)
        matches = SiftMatcher.flann.knnMatch(des1, des2, k=2)
        # TODO: sort the matches and add maxMatchCount argument

        # store all the good matches as per Lowe's ratio test.
        good = []
        displacement_pairs = []

        for m, n in matches:
            # TODO: fix the max displacement filter for a more general case
            # print(m.distance, n.distance, max_displacement)
            source_feature_point, destination_feature_point = (
                source_features[m.queryIdx].pt,
                destination_features[m.trainIdx].pt,
            )

            if (
                m.distance < n.distance
                and MathUtils.distance(source_feature_point, destination_feature_point)
                <= max_displacement
            ):
                good.append(m)
                # the "destination" matches from sift are actually "source" for warping because we want to reverse the matching positions onto the alignment image
                displacement_pairs.append(
                    [destination_feature_point, source_feature_point]
                )

        if len(good) > MIN_MATCH_COUNT:
            # store all if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32(
                [source_features[m.queryIdx].pt for m in good]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [destination_features[m.trainIdx].pt for m in good]
            ).reshape(-1, 1, 2)

            logger.debug(f"SIFT: {field_block_name}: source points", src_pts)
            logger.debug(f"SIFT: {field_block_name}: destination points", dst_pts)

            # TODO: understand matches_mask and need of homography for SIFT
            homography_matrix, mask = cv2.findHomography(
                src_pts, dst_pts, cv2.RANSAC, max_displacement
            )
            matches_mask = mask.ravel().tolist()

            h, w = alignment_image.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
                -1, 1, 2
            )
            dst = cv2.perspectiveTransform(pts, homography_matrix)
            logger.debug(f"SIFT: {field_block_name}: homography dst points", dst)

            gray_image = cv2.polylines(
                # ruff: noqa: FBT003
                gray_image,
                [np.int32(dst)],
                True,
                155,
                3,
                cv2.LINE_AA,
            )

        else:
            logger.critical(
                f"Not enough matches are found - {len(good)}/{MIN_MATCH_COUNT}"
            )
            matches_mask = None

        if config.outputs.show_image_level >= 6:
            draw_params = {
                "matchColor": (0, 255, 0),  # draw matches in green color
                "singlePointColor": None,
                # TODO: debug how to filter displacement_pairs using matches_mask
                "matchesMask": matches_mask,  # draw only inliers
                "flags": 2,
            }

            display_feature_matches = cv2.drawMatches(
                alignment_image,
                source_features,
                gray_image,
                destination_features,
                good,
                None,
                **draw_params,
            )

            InteractionUtils.show(
                f"Matches for {field_block_name}", display_feature_matches, 0
            )
        return displacement_pairs


# TODO: check static class
# Call once at import level
SiftMatcher.singleton_init()


def apply_sift_shifts(
    # ruff: noqa: PLR0913
    field_block_name,
    block_gray_image,
    block_colored_image,
    block_gray_alignment_image,
    # block_colored_alignment_image,
    max_displacement,
    margins,
    dimensions,
    config,
):
    local_displacement_pairs = SiftMatcher.get_matches(
        field_block_name,
        block_gray_image,
        block_gray_alignment_image,
        max_displacement,
        config,
    )

    # Rectangle that includes all of the 2D points that are to be added to the subdivision.
    warped_rectangle = (
        0,
        0,
        margins["left"] + margins["right"] + dimensions[0],
        margins["top"] + margins["bottom"] + dimensions[1],
    )

    # TODO: find a better way than fixed corners -
    # Add rectangle corner pairs for complete triangulation
    local_displacement_pairs += [
        [tuple(point), tuple(point)]
        for point in MathUtils.get_rectangle_points(
            1, 1, warped_rectangle[2] - 2, warped_rectangle[3] - 2
        )
    ]

    warped_block_image, warped_colored_image = apply_piecewise_affine(
        block_gray_image,
        block_colored_image,
        local_displacement_pairs,
        warped_rectangle,
        config,
    )
    if warped_block_image.shape != block_gray_image.shape:
        msg = f"Warped block image shape {warped_block_image.shape} does not match the original block image shape {block_gray_image.shape}"
        raise Exception(msg)

    show_displacement_overlay(
        block_gray_alignment_image, block_gray_image, warped_block_image
    )

    return warped_block_image, warped_colored_image
