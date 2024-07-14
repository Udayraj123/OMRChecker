import cv2
import numpy as np

from src.algorithm.phase_correlation import get_phase_correlation_shifts
from src.utils.image import ImageUtils, ImageWarpUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger

# from src.utils.math import MathUtils
cv = cv2


class SiftMatcher:
    sift = None
    flann = None

    def singleton_init() -> None:
        SiftMatcher.sift = cv2.SIFT_create()

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        SiftMatcher.flann = cv2.FlannBasedMatcher(index_params, search_params)

    @staticmethod
    def get_matches(field_block_name, gray_image, alignment_image, max_displacement):
        # Initiate SIFT detector
        # Resize to same (redundant)
        # temp resize 250, 133

        # TODO: fix this size dependency
        alignment_image = ImageUtils.resize_util(
            alignment_image, u_height=250, u_width=133
        )
        gray_image = ImageUtils.resize_util(gray_image, u_height=250, u_width=133)
        logger.info(alignment_image.shape, "alignment_image.shape")

        # find the keypoints and descriptors with SIFT
        kp1, des1 = SiftMatcher.sift.detectAndCompute(alignment_image, None)
        kp2, des2 = SiftMatcher.sift.detectAndCompute(gray_image, None)

        matches = SiftMatcher.flann.knnMatch(des1, des2, k=2)
        MIN_MATCH_COUNT = 10

        # store all the good matches as per Lowe's ratio test.
        good = []
        good_points = []

        for i, (m, n) in enumerate(matches):
            print(m.distance, n.distance, max_displacement)
            if m.distance < n.distance and m.distance <= 20 * max_displacement:
                good.append(m)
                # TODO: filter m.distance <= max_displacement
                pt1, pt2 = kp1[m.queryIdx].pt, kp2[m.trainIdx].pt
                good_points.append([pt1, pt2])

        if len(good) > MIN_MATCH_COUNT:
            # store all if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
            matchesMask = mask.ravel().tolist()

            h, w = alignment_image.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
                -1, 1, 2
            )
            dst = cv2.perspectiveTransform(pts, M)

            gray_image = cv2.polylines(
                gray_image, [np.int32(dst)], True, 155, 3, cv2.LINE_AA
            )

        else:
            print(f"Not enough matches are found - {len(good)}/{MIN_MATCH_COUNT}")
            matchesMask = None

        draw_params = dict(
            matchColor=(0, 255, 0),  # draw matches in green color
            singlePointColor=None,
            # TODO: debug how to filter good_points using matchesMask
            matchesMask=matchesMask,  # draw only inliers
            flags=2,
        )

        img3 = cv2.drawMatches(
            alignment_image, kp1, gray_image, kp2, good, None, **draw_params
        )

        InteractionUtils.show(f"Matches for {field_block_name}", img3)
        return good
        # TODO: apply triangulation or approach 1


# TODO: check static class
# Call once at import level
SiftMatcher.singleton_init()


def piecewise_affine(gray_image, colored_image, sift_displacement_pairs, rect):
    # Make a copy for warping
    warped_image, warped_colored_image = gray_image.copy(), colored_image.copy()

    # get DN triangles
    subdiv = cv2.Subdiv2D(rect)
    for [_source_point, destination_point] in sift_displacement_pairs:
        subdiv.insert((int(destination_point[0]), int(destination_point[1])))

    # Insert points into subdiv
    # for p in points:

    delaunay_triangles = subdiv.getTriangleList()
    # Get reverse-index mapping

    # TODO: filter the list of triangles based on the rectangle boundary
    # if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):

    #  then loop over triangles
    for source_points, destination_points in delaunay_triangles:
        # TODO: modify this loop to support 4-point transforms too!
        # if len(source_points == 4):
        # TODO: warp in colored image as well.
        ImageWarpUtils.warp_triangle(
            gray_image, warped_image, source_points, destination_points
        )

    return warped_image, warped_colored_image


def apply_template_alignment(gray_image, colored_image, template):
    if "gray_alignment_image" not in template.alignment:
        logger.info(f"Note: Alignment not enabled for template {template}")
        return gray_image, colored_image

    # Parsed
    margins, max_displacement = (
        template.alignment["margins"],
        template.alignment["maxDisplacement"],
    )

    # Pre-processed
    gray_alignment_image, colored_alignment_image = (
        template.alignment["gray_alignment_image"],
        template.alignment["colored_alignment_image"],
    )

    gray_image = ImageUtils.resize_to_dimensions(
        gray_image, template.template_dimensions
    )
    gray_alignment_image = ImageUtils.resize_to_dimensions(
        gray_alignment_image, template.template_dimensions
    )

    for field_block in template.field_blocks:
        field_block_name, origin, dimensions, field_block_alignment = map(
            lambda attr: getattr(field_block, attr),
            ["name", "origin", "dimensions", "alignment"],
        )
        # TODO: wrap this loop body into a function and generalize into passing *any* scanZone in this.

        local_max_displacement = field_block_alignment.get(
            "maxDisplacement", max_displacement
        )

        # TODO: uncomment this
        # if local_max_displacement == 0:
        #     # Skip alignment computation if allowed displacement is zero
        #     continue

        # compute zone and clip to image dimensions
        zone_start = [
            int(origin[0] - margins["left"]),
            int(origin[1] - margins["top"]),
        ]
        zone_end = [
            int(origin[0] + margins["right"] + dimensions[0]),
            int(origin[1] + margins["bottom"] + dimensions[1]),
        ]

        block_image = gray_image[
            zone_start[1] : zone_end[1], zone_start[0] : zone_end[0]
        ]
        block_alignment_image = gray_alignment_image[
            zone_start[1] : zone_end[1], zone_start[0] : zone_end[0]
        ]
        apply_phase_correlation_shifts(field_block, block_alignment_image, block_image)

        # sift_displacement_pairs = SiftMatcher.get_matches(
        #     field_block_name,
        #     gray_image,
        #     gray_alignment_image,
        #     origin,
        #     dimensions,
        #     margins,
        #     local_max_displacement,
        # )

        # # rect	Rectangle that includes all of the 2D points that are to be added to the subdivision.

        # rect = (zone_start[0], zone_start[1], zone_end[0], zone_end[1])
        # warped_image, warped_colored_image = piecewise_affine(
        #     gray_image, colored_image, sift_displacement_pairs, rect
        # )

        # MathUtils.get_rectangle_points_from_box(origin, dimensions)

    return gray_image, colored_image, template


def apply_phase_correlation_shifts(field_block, block_alignment_image, block_image):
    field_block.shifts, corr_image = get_phase_correlation_shifts(
        block_alignment_image, block_image
    )
    logger.info(field_block.name, field_block.shifts)

    # Translucent
    overlay = block_alignment_image.copy()
    overlay_shifted = block_alignment_image.copy()
    transparency = 0.5
    cv2.addWeighted(
        overlay,
        transparency,
        block_image,
        1 - transparency,
        0,
        overlay,
    )
    M = np.float32(
        [[1, 0, -1 * field_block.shifts[0]], [0, 1, -1 * field_block.shifts[1]]]
    )
    shifted_block_image = cv2.warpAffine(
        block_image, M, (block_image.shape[1], block_image.shape[0])
    )
    cv2.addWeighted(
        overlay_shifted,
        transparency,
        shifted_block_image,
        1 - transparency,
        0,
        overlay_shifted,
    )
    InteractionUtils.show("Correlation", corr_image, 0)

    InteractionUtils.show(
        "Alignment + Input Field Block",
        ImageUtils.get_padded_hstack([block_alignment_image, block_image]),
        0,
    )

    InteractionUtils.show(
        "Shifts Overlay",
        ImageUtils.get_padded_hstack([overlay, overlay_shifted]),
    )
