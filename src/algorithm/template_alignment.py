import cv2
import numpy as np

from src.algorithm.phase_correlation import get_phase_correlation_shifts
from src.utils.drawing import DrawingUtils
from src.utils.image import ImageUtils, ImageWarpUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger
from src.utils.math import MathUtils


class SiftMatcher:
    sift = None
    flann = None

    def singleton_init() -> None:
        # Initiate SIFT detector
        SiftMatcher.sift = cv2.SIFT_create()

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        # Initiate Flann matcher
        SiftMatcher.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # TODO: [later] add support for more matchers

    @staticmethod
    def get_matches(field_block_name, gray_image, alignment_image, max_displacement):
        # TODO: add a small blur/denoise to get better matches

        # find the keypoints and descriptors with SIFT
        source_features, des1 = SiftMatcher.sift.detectAndCompute(alignment_image, None)
        destination_features, des2 = SiftMatcher.sift.detectAndCompute(gray_image, None)

        matches = SiftMatcher.flann.knnMatch(des1, des2, k=2)
        # TODO: sort the matches and add maxMatchCount argument
        MIN_MATCH_COUNT = 10

        # store all the good matches as per Lowe's ratio test.
        good = []
        good_points = []

        for i, (m, n) in enumerate(matches):
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
                good_points.append([source_feature_point, destination_feature_point])

        if len(good) > MIN_MATCH_COUNT:
            # store all if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32(
                [source_features[m.queryIdx].pt for m in good]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [destination_features[m.trainIdx].pt for m in good]
            ).reshape(-1, 1, 2)

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
        return good_points
        # TODO: apply triangulation or approach 1


# TODO: check static class
# Call once at import level
SiftMatcher.singleton_init()


def apply_piecewise_affine(
    gray_block_image, colored_block_image, displacement_pairs, rect
):
    # ## TODO: remove this filter
    filtered_displacement_pairs = [
        [list(map(round, source_point)), list(map(round, destination_point))]
        for [source_point, destination_point] in displacement_pairs
        if MathUtils.rectangle_contains(source_point, rect)
        and MathUtils.rectangle_contains(destination_point, rect)
    ]
    print(
        f"displacement_pairs: {len(displacement_pairs)} -> {len(filtered_displacement_pairs)}"
    )
    if len(filtered_displacement_pairs) == 0:
        logger.warning(
            f"Invalid displacement points, no point-pair found in the given rectangle: {rect}"
        )
        logger.info(displacement_pairs)
        return gray_block_image, colored_block_image
    # ##

    # Make a copy for warping
    warped_block_image, warped_colored_image = gray_block_image.copy(), (
        None if colored_block_image is None else colored_block_image.copy()
    )

    # Bulk insert all destination points
    destination_subdiv = cv2.Subdiv2D(rect)
    destination_subdiv.insert(
        [
            (int(destination_point[0]), int(destination_point[1]))
            for [_source_point, destination_point] in filtered_displacement_pairs
        ]
    )

    destination_delaunay_triangles_list = [
        [(round(triangle[2 * i]), round(triangle[2 * i + 1])) for i in range(3)]
        for triangle in destination_subdiv.getTriangleList()
    ]

    destination_delaunay_triangles = [
        triangle
        for triangle in destination_delaunay_triangles_list
        # TODO[think]: why exactly do we need to filter outside triangles at start?
        # How to get rid of "zero-sized triangles" e.g. lines
        if all(MathUtils.rectangle_contains(point, rect) for point in triangle)
    ]
    print(
        f"destination_delaunay_triangles: {len(destination_delaunay_triangles_list)} -> {len(destination_delaunay_triangles)}"
    )
    if len(destination_delaunay_triangles) == 0:
        logger.warning(
            f"Invalid displacement points, no point-pair found in the given rectangle: {rect}"
        )
        logger.info(destination_delaunay_triangles)
        return warped_block_image, warped_colored_image

    # Store the reverse point mapping
    destination_to_source_point_map = {
        tuple(destination_point): source_point
        for [source_point, destination_point] in filtered_displacement_pairs
    }
    logger.info("filtered_displacement_pairs", filtered_displacement_pairs)
    logger.info("destination_delaunay_triangles", destination_delaunay_triangles)
    logger.info("destination_to_source_point_map", destination_to_source_point_map)

    # Get the corresponding source triangles
    source_delaunay_triangles = [
        list(map(destination_to_source_point_map.get, destination_triangle))
        for destination_triangle in destination_delaunay_triangles
    ]

    # TODO: visualise the triangles here
    #  then loop over triangles
    for source_points, destination_points in zip(
        source_delaunay_triangles, destination_delaunay_triangles
    ):
        # TODO: modify this loop to support 4-point transforms too!
        # if len(source_points == 4):
        logger.info(source_points, destination_points)
        # TODO: modify warped_colored_image
        ImageWarpUtils.warp_triangle_inplace(
            gray_block_image, warped_block_image, source_points, destination_points
        )

        DrawingUtils.draw_polygon(gray_block_image, source_points)
        DrawingUtils.draw_polygon(warped_block_image, destination_points)

        # ImageWarpUtils.warp_triangle_inplace(
        #     colored_image, warped_colored_image, source_points, destination_points
        # )

    return warped_block_image, warped_colored_image


def apply_template_alignment(gray_image, colored_image, template):
    if "gray_alignment_image" not in template.alignment:
        logger.info(f"Note: Alignment not enabled for template {template}")
        return gray_image, colored_image, template

    # Parsed
    template_margins, template_max_displacement = (
        template.alignment["margins"],
        template.alignment["maxDisplacement"],
    )

    # Pre-processed
    gray_alignment_image, colored_alignment_image = (
        template.alignment["gray_alignment_image"],
        template.alignment["colored_alignment_image"],
    )
    # Note: resize also creates a copy
    gray_image, colored_image, gray_alignment_image, colored_alignment_image = map(
        lambda image: (
            None
            if image is None
            else ImageUtils.resize_to_dimensions(image, template.template_dimensions)
        ),
        [gray_image, colored_image, gray_alignment_image, colored_alignment_image],
    )

    for field_block in template.field_blocks:
        field_block_name, origin, dimensions, field_block_alignment = map(
            lambda attr: getattr(field_block, attr),
            ["name", "origin", "dimensions", "alignment"],
        )
        # TODO: wrap this loop body into a function and generalize into passing *any* scanZone in this.

        margins = field_block_alignment.get("margins", template_margins)
        max_displacement = field_block_alignment.get(
            "maxDisplacement", template_max_displacement
        )

        # TODO: uncomment this
        if max_displacement == 0:
            # Skip alignment computation if allowed displacement is zero
            continue

        # compute zone and clip to image dimensions
        zone_start = [
            int(origin[0] - margins["left"]),
            int(origin[1] - margins["top"]),
        ]
        zone_end = [
            int(origin[0] + margins["right"] + dimensions[0]),
            int(origin[1] + margins["bottom"] + dimensions[1]),
        ]

        block_gray_image, block_colored_image, block_gray_alignment_image = map(
            lambda image: (
                None
                if image is None
                else image[zone_start[1] : zone_end[1], zone_start[0] : zone_end[0]]
            ),
            [gray_image, colored_image, gray_alignment_image],
        )

        InteractionUtils.show("block_gray_image-before", block_gray_image)
        block_gray_image = hough_circles(block_gray_image)
        InteractionUtils.show("block_gray_image", block_gray_image)

        InteractionUtils.show(
            "block_gray_alignment_image-before", block_gray_alignment_image
        )
        block_gray_alignment_image = hough_circles(block_gray_alignment_image)
        InteractionUtils.show("block_gray_alignment_image", block_gray_alignment_image)

        # TODO: move to a processor:
        # warped_block_image = apply_phase_correlation_shifts(
        #     field_block, block_gray_alignment_image, block_gray_image
        # )

        warped_block_image, warped_colored_image = apply_sift_shifts(
            field_block_name,
            block_gray_image,
            block_colored_image,
            block_gray_alignment_image,
            # block_colored_alignment_image,
            max_displacement,
            margins,
            dimensions,
        )

        show_displacement_overlay(
            block_gray_alignment_image, block_gray_image, warped_block_image
        )

        # TODO: assignment outside loop?
        # Set warped field block back into original image
        gray_image[
            zone_start[1] : zone_end[1], zone_start[0] : zone_end[0]
        ] = warped_block_image
        if colored_image:
            colored_image[
                zone_start[1] : zone_end[1], zone_start[0] : zone_end[0]
            ] = warped_colored_image

        # TODO: ideally apply detection on these copies so that we can support overlapping field blocks!

    return gray_image, colored_image, template


def apply_sift_shifts(
    field_block_name,
    block_gray_image,
    block_colored_image,
    block_gray_alignment_image,
    # block_colored_alignment_image,
    max_displacement,
    margins,
    dimensions,
):
    local_displacement_pairs = SiftMatcher.get_matches(
        field_block_name,
        block_gray_image,
        block_gray_alignment_image,
        max_displacement,
    )

    # rect:	Rectangle that includes all of the 2D points that are to be added to the subdivision.
    shifted_rect = (
        0,
        0,
        margins["left"] + margins["right"] + dimensions[0],
        margins["top"] + margins["bottom"] + dimensions[1],
    )
    warped_block_image, warped_colored_image = apply_piecewise_affine(
        block_gray_image,
        block_colored_image,
        local_displacement_pairs,
        shifted_rect,
    )
    assert warped_block_image.shape == block_gray_image.shape

    return warped_block_image, warped_colored_image


def apply_phase_correlation_shifts(
    field_block, block_gray_alignment_image, block_gray_image
):
    field_block.shifts, corr_image = get_phase_correlation_shifts(
        block_gray_alignment_image, block_gray_image
    )
    logger.info(field_block.name, field_block.shifts)

    M = np.float32(
        [[1, 0, -1 * field_block.shifts[0]], [0, 1, -1 * field_block.shifts[1]]]
    )
    shifted_block_image = cv2.warpAffine(
        block_gray_image, M, (block_gray_image.shape[1], block_gray_image.shape[0])
    )
    InteractionUtils.show("Correlation", corr_image, 0)

    show_displacement_overlay(
        block_gray_alignment_image, block_gray_image, shifted_block_image
    )

    return shifted_block_image


def show_displacement_overlay(
    block_gray_alignment_image, block_gray_image, shifted_block_image
):
    # InteractionUtils.show(
    #     "Reference + Input Field Block",
    #     ImageUtils.get_padded_hstack([block_gray_alignment_image, block_gray_image]),
    #     0,
    # )

    transparency = 0.5
    overlay = block_gray_alignment_image.copy()
    cv2.addWeighted(
        overlay,
        transparency,
        block_gray_image,
        1 - transparency,
        0,
        overlay,
    )

    transparency = 0.5
    overlay_shifted = block_gray_alignment_image.copy()
    cv2.addWeighted(
        overlay_shifted,
        transparency,
        shifted_block_image,
        1 - transparency,
        0,
        overlay_shifted,
    )

    InteractionUtils.show(
        "Alignment Overlays",
        ImageUtils.get_padded_hstack([overlay, overlay_shifted]),
    )


# TODO: move this into drawing utils
# TODO: use this and display the facets
def draw_voronoi(img, subdiv):
    (facets, centers) = subdiv.getVoronoiFacetList([])

    for i in xrange(0, len(facets)):
        ifacet_arr = []
        for f in facets[i]:
            ifacet_arr.append(f)

        ifacet = np.array(ifacet_arr, np.int)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        cv2.fillConvexPoly(img, ifacet, color, cv2.CV_AA, 0)
        ifacets = np.array([ifacet])
        cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.CV_AA, 0)
        cv2.circle(
            img,
            (centers[i][0], centers[i][1]),
            3,
            (0, 0, 0),
            cv2.cv.CV_FILLED,
            cv2.CV_AA,
            0,
        )


def hough_circles(gray):
    gray = cv2.medianBlur(gray, 5)
    rows = gray.shape[0]
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        1,
        rows / 10,
        param1=100,
        param2=30,
        minRadius=1,
        maxRadius=30,
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        logger.info("circles", circles)
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(gray, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(gray, center, radius, (255, 0, 255), 3)
    return gray
