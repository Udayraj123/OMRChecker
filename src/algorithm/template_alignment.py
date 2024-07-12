import cv2
import numpy as np

from src.utils.image import ImageUtils
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
    def get_matches(
        field_block_name, img1, img2, origin, dimensions, margins, max_displacement
    ):
        # compute zone and clip to image dimensions
        zone_start = [
            int(origin[0] - margins["left"]),
            int(origin[1] - margins["top"]),
        ]
        zone_end = [
            int(origin[0] + margins["right"] + dimensions[0]),
            int(origin[1] + margins["bottom"] + dimensions[1]),
        ]

        # Initiate SIFT detector
        img1 = img1[zone_start[1] : zone_end[1], zone_start[0] : zone_end[0]]
        img2 = img2[zone_start[1] : zone_end[1], zone_start[0] : zone_end[0]]

        # Resize to same (redundant)
        # temp resize 250, 133
        # TODO: fix this size dependency
        img1 = ImageUtils.resize_util(img1, u_height=250, u_width=133)
        img2 = ImageUtils.resize_util(img2, u_height=250, u_width=133)
        logger.info(img1.shape, "img1.shape")

        # find the keypoints and descriptors with SIFT
        kp1, des1 = SiftMatcher.sift.detectAndCompute(img1, None)
        kp2, des2 = SiftMatcher.sift.detectAndCompute(img2, None)

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

            h, w = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
                -1, 1, 2
            )
            dst = cv2.perspectiveTransform(pts, M)

            img2 = cv2.polylines(img2, [np.int32(dst)], True, 155, 3, cv2.LINE_AA)

        else:
            print(f"Not enough matches are found - {len(good)}/{MIN_MATCH_COUNT}")
            matchesMask = None

        draw_params = dict(
            matchColor=(0, 255, 0),  # draw matches in green color
            singlePointColor=None,
            # matchesMask=matchesMask,  # draw only inliers
            flags=2,
        )

        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

        InteractionUtils.show(f"Matches for {field_block_name}", img3)

        # TODO: apply triangulation or approach 1


# TODO: check static class
# Call once at import level
SiftMatcher.singleton_init()


def apply_template_alignment(template, gray_image, colored_image):
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
        local_max_displacement = field_block_alignment.get(
            "maxDisplacement", max_displacement
        )
        if local_max_displacement == 0: 
            # Don't compute alignment if allowed displacement is zero
            continue
        SiftMatcher.get_matches(
            field_block_name,
            gray_alignment_image,
            gray_image,
            origin,
            dimensions,
            margins,
            local_max_displacement,
        )
        # MathUtils.get_rectangle_points_from_box(origin, dimensions)

    return gray_image, colored_image
