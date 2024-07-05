from typing import *

import cv2
import numpy as np
from rich.table import Table
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import linemerge

from src.processors.constants import EDGE_TYPES_IN_ORDER
from src.utils.logger import console, logger
from src.utils.math import MathUtils

# from .mapping import scale_map


def rectify(
    *,
    edge_contours_map: Dict[str, List[Tuple[int, int]]],
    enable_cropping: bool,
    # output_shape: Tuple[int, int],
    # resolution_map: int = 512,
) -> np.ndarray:
    """
    Rectifies an image based on the given mask. Corners can be provided to speed up the rectification.

    Args:
        image (np.ndarray): The input image as a NumPy array. Shape: (height, width, channels).
        mask (np.ndarray): The mask indicating the region of interest as a NumPy array. Shape: (height, width).
        output_shape (Tuple[int, int]): The desired output shape of the rectified image. First element: height, second element: width.
        corners (Optional[List[Tuple[int, int]]], optional): The corners of the region of interest. List of points with first x, then y coordinate. Has to contain exactly 4 points. Defaults to None.
        resolution_map (int, optional): The resolution of the mapping grid. Higher grid size increases computation time but generates better results.  Defaults to 512.

    Returns:
        np.ndarray: The rectified image as a NumPy array.
    """

    segments = {
        edge_type.lower(): LineString(edge_contours_map[edge_type])
        for edge_type in EDGE_TYPES_IN_ORDER
    }

    # logger.info("segments", segments)

    # given_segments = colored_image.copy()
    # for name, segment in segments.items():
    #     min_x, min_y, max_x, max_y = segment.bounds
    #     DrawingUtils.draw_text(given_segments, name, [min_x - 10, min_y + 10])
    #     DrawingUtils.draw_contour(given_segments, list(segment.coords))
    # InteractionUtils.show("given_segments", given_segments)

    output_shape = _get_output_shape_for_segments(edge_contours_map, enable_cropping)

    scaled_map = _create_backward_output_map(segments, output_shape, enable_cropping)
    return scaled_map
    # resolution_map = 10  # temp
    # backward_map = _create_backward_map(segments, resolution_map)
    # logger.info("backward_map.shape", backward_map.shape)
    # scaled_map = scale_map(backward_map, output_shape)

    # return warped_colored_image


def _get_output_shape_for_segments(
    edge_contours_map: Dict[str, List[Tuple[int, int]]], enable_cropping: bool
):
    max_width = max(
        [
            MathUtils.distance(edge_contours_map[name][0], edge_contours_map[name][-1])
            for name in ["TOP", "BOTTOM"]
        ]
    )
    max_height = max(
        [
            MathUtils.distance(edge_contours_map[name][0], edge_contours_map[name][-1])
            for name in ["LEFT", "RIGHT"]
        ]
    )
    return int(max_height), int(max_width)


def transform_line(line, M):
    line_points = np.array(line.coords).reshape(-1, 1, 2)
    line_points = cv2.perspectiveTransform(line_points, M).squeeze()
    return LineString(line_points)


def _create_backward_output_map(
    segments: Dict[str, LineString],
    output_shape: Tuple[int, int],
    enable_cropping: bool,
) -> np.ndarray:
    resolution_h, resolution_w = output_shape
    transformed_backward_points_map = np.zeros((resolution_h, resolution_w, 2))

    # TODO: support for in-place transforms when enable_cropping is False
    # map geometric objects to normalized space
    coord_map = {
        (0, 0): segments["top"].coords[0],
        (resolution_w, 0): segments["top"].coords[-1],
        (0, resolution_h): segments["bottom"].coords[-1],  # Clockwise order of points
        (resolution_w, resolution_h): segments["bottom"].coords[0],
    }
    # Get page corners tranform matrix
    dst = np.array(list(coord_map.keys())).astype("float32")
    src = np.array(list(coord_map.values())).astype("float32")
    print_points_mapping_table(f"Corners Transform", src, dst)

    # TODO: can get this matrix from homography as well! (for inplace alignment)
    M = cv2.getPerspectiveTransform(src=src, dst=dst)

    transformed_segments = {
        name: transform_line(segments[name], M) for name in segments.keys()
    }

    combined_segments = Polygon(linemerge(segments.values()))

    # Reshape to match with perspectiveTransform() expected input
    # (x, 2) -> (x, 1, 2)
    combined_segments_points = np.array(combined_segments.exterior.coords)
    combined_segments_points_for_transform = combined_segments_points.reshape(-1, 1, 2)
    cropped_combined_segments_points = cv2.perspectiveTransform(
        combined_segments_points_for_transform, M
    ).squeeze()

    cropped_combined_segments = Polygon(cropped_combined_segments_points)
    (
        cropped_page_min_x,
        cropped_page_min_y,
        cropped_page_max_x,
        cropped_page_max_y,
    ) = cropped_combined_segments.buffer(1).bounds

    control_points = [
        [int(point[0]), int(point[1])] for point in combined_segments_points
    ]
    destination_points = [
        [int(point[0]), int(point[1])] for point in cropped_combined_segments_points
    ]
    print_points_mapping_table(
        f"Contours Approx Transform", control_points, destination_points
    )

    logger.debug(
        f"Approx Transformed Bounds: ({cropped_page_min_x:.2f},{cropped_page_min_y:.2f}) -> ({cropped_page_max_x:.2f},{cropped_page_max_y:.2f})",
    )

    transformed_box_with_vertical_curves = MultiLineString(
        [transformed_segments["left"], transformed_segments["right"]]
    )

    # TODO: >> see if RegularGridInterpolator can be used for making this efficient!
    grid_step_points_h = list(range(0, resolution_h))  # np.linspace(0, 1, resolution_h)
    edge_cases = {0: 0, resolution_h - 1: -1}

    # calculate y - values by interpolating x values
    for y_grid_index, grid_step_point in enumerate(grid_step_points_h):
        # logger.info(f"{grid_step_point}/{len(grid_step_points_h)}")
        if y_grid_index in edge_cases:
            pos = edge_cases[y_grid_index]
            intersections = [
                # Clockwise order adjustment
                Point(transformed_segments["left"].coords[-1 if pos == 0 else 0]),
                Point(transformed_segments["right"].coords[pos]),
            ]
        else:
            scan_line_x = LineString(
                [
                    (cropped_page_min_x, grid_step_point),
                    (cropped_page_max_x, grid_step_point),
                ]
            )
            # logger.info(transformed_box_with_vertical_curves, scan_line_x)
            intersections = list(
                transformed_box_with_vertical_curves.intersection(scan_line_x).geoms
            )

        if len(intersections) > 2:
            # TODO later: here we can find non-corner anchor points to align too!
            logger.warning(grid_step_point, intersections)

        intersections = [
            min(intersections, key=lambda p: p.x),
            max(intersections, key=lambda p: p.x),
        ]

        # TODO: maybe do this using perspectiveTransform on the whole line-strip (based on resolution?
        # TODO: optimization - O(h) -> O(len(contour)) can do per boundary point in a y-sorted fashion

        # TODO: support for maxDisplacements/max_displacements (dx, dy) and check (resolution_w - intersections[1].x) < dx and
        # stretch_start = min( max_displacements[0], intersections[0].x)
        # stretch_end = resolution_w - min(max_displacements[1], resolution_w - intersections[1].x)

        # stretching 'x' intersections to the resolution grid size
        stretched_x_line = np.linspace(
            intersections[0].x, intersections[1].x, resolution_w
        )
        transformed_backward_points_map[
            y_grid_index, 0:resolution_w, 0
        ] = stretched_x_line

    transformed_box_with_horizontal_curves = MultiLineString(
        [transformed_segments["top"], transformed_segments["bottom"]]
    )
    grid_step_points_w = list(range(0, resolution_w))  # np.linspace(0, 1, resolution_w)
    edge_cases = {0: 0, resolution_w - 1: -1}
    # calculate x - values by interpolating y values
    for x_grid_index, grid_step_point in enumerate(grid_step_points_w):
        # logger.info(f"{x_grid_index}/{len(grid_step_points_w)}")
        if x_grid_index in edge_cases:
            pos = edge_cases[x_grid_index]
            intersections = [
                Point(transformed_segments["top"].coords[pos]),
                # Clockwise order adjustment
                Point(transformed_segments["bottom"].coords[-1 if pos == 0 else 0]),
            ]
        else:
            scan_line_y = LineString(
                [
                    (grid_step_point, cropped_page_min_y),
                    (grid_step_point, cropped_page_max_y),
                ]
            )
            intersections = list(
                transformed_box_with_horizontal_curves.intersection(scan_line_y).geoms
            )

        # if len(intersections) > 2:
        intersections = [
            min(intersections, key=lambda p: p.y),
            max(intersections, key=lambda p: p.y),
        ]

        # stretching 'y' intersections to the resolution grid size
        stretched_y_line = np.linspace(
            intersections[0].y, intersections[1].y, resolution_h
        )
        transformed_backward_points_map[
            0:resolution_h, x_grid_index, 1
        ] = stretched_y_line

    # transform grid back to original space
    # (resolution_h, resolution_w, 2) -> (resolution_h * resolution_w, 2)
    backward_map_points = transformed_backward_points_map.reshape(-1, 1, 2)
    # The values represent curve's points
    ordered_rich_curve_points = cv2.perspectiveTransform(
        backward_map_points, np.linalg.inv(M)
    ).squeeze()
    #  (resolution_h * resolution_w, 2) -> (resolution_h, resolution_w, 2)
    backward_points_map = ordered_rich_curve_points.reshape(
        resolution_h, resolution_w, 2
    )

    # flip x and y coordinate: first y, then x
    # backward_points_map = np.roll(backward_points_map, shift=1, axis=-1)
    backward_points_map = np.float32(backward_points_map)
    return backward_points_map


def print_points_mapping_table(title, control_points, destination_points):
    table = Table(
        title=title,
        show_header=True,
        show_lines=False,
    )
    table.add_column("Control", style="cyan", no_wrap=True)
    table.add_column("Destination", style="magenta")
    for c, d in zip(control_points, destination_points):
        table.add_row(str(c), str(d))
    console.print(table, justify="center")
