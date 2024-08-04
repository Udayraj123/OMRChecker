import random

import cv2
import numpy as np

from src.utils.constants import CLR_DARK_BLUE, CLR_DARK_GREEN, CLR_DARK_RED
from src.utils.drawing import DrawingUtils
from src.utils.image import ImageUtils
from src.utils.image_warp import ImageWarpUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger
from src.utils.math import MathUtils


# TODO: move this into drawing utils
# TODO: use this and display the facets
def draw_voronoi(image, subdiv):
    voronoi_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    (facets, centers) = subdiv.getVoronoiFacetList([])

    for i in range(0, len(facets)):
        ifacet_arr = []
        for f in facets[i]:
            ifacet_arr.append(f)

        ifacet = np.array(ifacet_arr, int)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        cv2.fillConvexPoly(voronoi_image, ifacet, color, cv2.LINE_AA, 0)
        ifacets = np.array([ifacet])
        cv2.polylines(voronoi_image, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)
        cv2.circle(
            voronoi_image,
            (int(centers[i][0]), int(centers[i][1])),
            3,
            (0, 0, 0),
            cv2.FILLED,
            cv2.LINE_AA,
            0,
        )
    return voronoi_image


def apply_piecewise_affine(
    gray_block_image, colored_block_image, displacement_pairs, warped_rectangle, config
):
    # ## TODO: remove/update this filter
    # Can we somehow allow displacements in one direction(majority) and eliminate the ones going the other way too?

    parsed_displacement_pairs = [
        [list(map(round, source_point)), list(map(round, destination_point))]
        for [source_point, destination_point] in displacement_pairs
        if MathUtils.rectangle_contains(source_point, warped_rectangle)
        and MathUtils.rectangle_contains(destination_point, warped_rectangle)
    ]
    logger.info(
        "parsed_displacement_pairs", parsed_displacement_pairs, warped_rectangle
    )
    logger.info(
        f"displacement_pairs: {len(displacement_pairs)} -> {len(parsed_displacement_pairs)}"
    )
    if len(parsed_displacement_pairs) == 0:
        logger.warning(
            f"Invalid displacement points, no point-pair found in the given rectangle: {warped_rectangle}"
        )
        logger.info(displacement_pairs)
        return gray_block_image, colored_block_image
    # ##

    # Make a copy for warping
    warped_block_image, warped_colored_image = gray_block_image.copy(), (
        None if colored_block_image is None else colored_block_image.copy()
    )

    # Bulk insert all destination points
    destination_subdiv = cv2.Subdiv2D(warped_rectangle)
    destination_subdiv.insert(
        [
            (int(destination_point[0]), int(destination_point[1]))
            for [_source_point, destination_point] in parsed_displacement_pairs
        ]
    )
    initial_voronoi_image = draw_voronoi(warped_block_image, destination_subdiv)
    InteractionUtils.show("initial_voronoi_image", initial_voronoi_image, 0)

    destination_delaunay_triangles_list = [
        [(round(triangle[2 * i]), round(triangle[2 * i + 1])) for i in range(3)]
        for triangle in destination_subdiv.getTriangleList()
    ]

    destination_delaunay_triangles = [
        triangle
        for triangle in destination_delaunay_triangles_list
        # TODO[think]: why exactly do we need to filter outside triangles at start?
        # How to get rid of "zero-sized triangles" e.g. lines
        if all(
            MathUtils.rectangle_contains(point, warped_rectangle) for point in triangle
        )
    ]
    logger.info(
        f"destination_delaunay_triangles: {len(destination_delaunay_triangles_list)} -> {len(destination_delaunay_triangles)} inside rectangle {warped_rectangle}"
    )
    if len(destination_delaunay_triangles) == 0:
        logger.warning(
            f"Invalid displacement points, no point-pair found in the given rectangle: {warped_rectangle}"
        )
        logger.info(destination_delaunay_triangles)
        return warped_block_image, warped_colored_image

    # Store the reverse point mapping
    destination_to_source_point_map = {
        tuple(destination_point): source_point
        for [source_point, destination_point] in parsed_displacement_pairs
    }
    logger.info("parsed_displacement_pairs", parsed_displacement_pairs)
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

        # TODO: remove this debug snippet
        logger.info(source_points, destination_points)
        if config.outputs.show_image_level >= 5:
            gray_block_image_before = cv2.cvtColor(gray_block_image, cv2.COLOR_GRAY2BGR)
            warped_block_image_before = cv2.cvtColor(
                warped_block_image, cv2.COLOR_GRAY2BGR
            )

        ImageWarpUtils.warp_triangle_inplace(
            gray_block_image,
            warped_block_image,
            source_points,
            destination_points,
            config,
        )

        # TODO: modify warped_colored_image as well
        # ImageWarpUtils.warp_triangle_inplace(
        #     colored_image, warped_colored_image, source_points, destination_points
        # )

        if config.outputs.show_image_level >= 5:
            warped_block_image_after = cv2.cvtColor(
                warped_block_image, cv2.COLOR_GRAY2BGR
            )

            # TODO: remove - Mutates input image box
            DrawingUtils.draw_polygon(
                gray_block_image, source_points, color=CLR_DARK_RED
            )
            # DrawingUtils.draw_polygon(gray_block_image, destination_points, color=CLR_DARK_GREEN)

            DrawingUtils.draw_polygon(
                gray_block_image_before, source_points, color=CLR_DARK_RED
            )
            DrawingUtils.draw_polygon(
                gray_block_image_before, destination_points, color=CLR_DARK_GREEN
            )
            DrawingUtils.draw_polygon(
                warped_block_image_before, destination_points, color=CLR_DARK_BLUE
            )
            DrawingUtils.draw_polygon(
                warped_block_image_after, destination_points, color=CLR_DARK_GREEN
            )
            overlay = ImageUtils.overlay_image(
                gray_block_image_before, warped_block_image_after
            )

            InteractionUtils.show(
                f"warped_block_image-{destination_points}",
                ImageUtils.get_padded_hstack(
                    [
                        gray_block_image_before,
                        warped_block_image_before,
                        warped_block_image_after,
                        overlay,
                    ]
                ),
                0,
            )

    return warped_block_image, warped_colored_image
