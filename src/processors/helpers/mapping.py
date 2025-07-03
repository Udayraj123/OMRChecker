from typing import *

import cv2
import numpy as np
import pandas as pd

# import torch
# import torch.nn.functional as F
# from einops import rearrange
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
    box,
)
from shapely.ops import linemerge, nearest_points, snap, split


def create_identity_meshgrid(
    # ruff: noqa: FBT001
    resolution: Union[int, Tuple],
    with_margin: bool = False,
):
    if isinstance(resolution, int):
        resolution = (resolution, resolution)

    margin_0 = 0.5 / resolution[0] if with_margin else 0
    margin_1 = 0.5 / resolution[1] if with_margin else 0
    return np.mgrid[
        margin_0 : 1 - margin_0 : complex(0, resolution[0]),
        margin_1 : 1 - margin_1 : complex(0, resolution[1]),
    ].transpose(1, 2, 0)


def scale_map(input_map: np.ndarray, output_shape: Union[int, Tuple[int, int]]):
    if isinstance(output_shape, int):
        output_shape = (output_shape, output_shape)

    h, w, channels = input_map.shape
    if output_shape[0] == h and output_shape[1] == w:
        return input_map.copy()

    if np.any(np.isnan(input_map)):
        pass

    # The range of x and y coordinates to get a cross product
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    # Fit the interpolator grid to input_map
    interp = RegularGridInterpolator((y, x), input_map, method="linear")

    # xi = values to evaluate at
    xi = create_identity_meshgrid(output_shape, with_margin=False).reshape(-1, 2)

    return interp(xi).reshape(*output_shape, channels)


# def apply_map(
#     image: np.ndarray,
#     bm: np.ndarray,
#     resolution: Union[None, int, Tuple[int, int]] = None,
# ):
#     if resolution is not None:
#         bm = scale_map(bm, resolution)

#     input_dtype = image.dtype
#     img = rearrange(image, "h w c -> 1 c h w")

#     img = torch.from_numpy(img).double()

#     bm = torch.from_numpy(bm).unsqueeze(0).double()
#     # normalise bm to [-1, 1]
#     bm = (bm * 2) - 1
#     bm = torch.roll(bm, shifts=1, dims=-1)

#     res = F.grid_sample(input=img, grid=bm, align_corners=True, padding_mode="border")
#     res = rearrange(res[0], "c h w -> h w c")

#     res = res.numpy().astype(input_dtype)
#     return res


def _create_backward_map(
    segments: Dict[str, LineString],
    resolution: int,  # Tuple[int, int]
) -> np.ndarray:
    # map geometric objects to normalized space
    coord_map = {
        (0, 0): segments["top"].coords[0],
        (1, 0): segments["top"].coords[-1],
        (0, 1): segments["bottom"].coords[0],
        (1, 1): segments["bottom"].coords[-1],
    }
    dst = np.array(list(coord_map.keys())).astype("float32")
    src = np.array(list(coord_map.values())).astype("float32")
    transform_matrix = cv2.getPerspectiveTransform(src=src, dst=dst)

    polygon = Polygon(linemerge(segments.values()))
    pts = np.array(polygon.exterior.coords).reshape(-1, 1, 2)
    pts = cv2.perspectiveTransform(pts, transform_matrix).squeeze()

    p_norm = Polygon(pts)

    def transform_line(line: LineString) -> LineString:
        line_points = np.array(line.coords).reshape(-1, 1, 2)
        line_points = cv2.perspectiveTransform(line_points, transform_matrix).squeeze()
        return LineString(line_points)

    h_norm = [transform_line(segments[name]) for name in ["top", "bottom"]]
    v_norm = [transform_line(segments[name]) for name in ["left", "right"]]
    p_h_norm = MultiLineString(h_norm)
    p_v_norm = MultiLineString(v_norm)

    (minx, miny, maxx, maxy) = p_norm.buffer(1).bounds

    backward_map = np.zeros((resolution, resolution, 2))
    grid_step_points = np.linspace(0, 1, resolution)
    edge_cases = {0: 0, resolution - 1: -1}

    # calculate y - values by interpolating x values
    for y_grid_index, grid_step_point in enumerate(grid_step_points):
        if y_grid_index in edge_cases:
            pos = edge_cases[y_grid_index]
            intersections = [Point(v_norm[0].coords[pos]), Point(v_norm[1].coords[pos])]
        else:
            divider = LineString([(minx, grid_step_point), (maxx, grid_step_point)])
            intersections = list(p_v_norm.intersection(divider).geoms)

        if len(intersections) > 2:
            intersections = [
                min(intersections, key=lambda p: p.x),
                max(intersections, key=lambda p: p.x),
            ]

        if len(intersections) != 2:
            msg = "Unexpected number of intersections!"
            raise Exception(msg)

        # stretching 'x' intersections to the resolution grid size
        stretched_x_line = np.linspace(
            intersections[0].x, intersections[1].x, resolution
        )
        backward_map[y_grid_index, 0:resolution, 0] = stretched_x_line

    # calculate x - values by interpolating y values
    for x_grid_index, grid_step_point in enumerate(grid_step_points):
        if x_grid_index in edge_cases:
            pos = edge_cases[x_grid_index]
            intersections = [Point(h_norm[0].coords[pos]), Point(h_norm[1].coords[pos])]
        else:
            divider = LineString([(grid_step_point, miny), (grid_step_point, maxy)])
            intersections = list(p_h_norm.intersection(divider).geoms)

        if len(intersections) > 2:
            intersections = [
                min(intersections, key=lambda p: p.y),
                max(intersections, key=lambda p: p.y),
            ]

            if len(intersections) != 2:
                msg = "Unexpected number of intersections!"
                raise Exception(msg)

        # stretching 'y' intersections to the resolution grid size
        stretched_y_line = np.linspace(
            intersections[0].y, intersections[1].y, resolution
        )
        backward_map[0:resolution, x_grid_index, 1] = stretched_y_line

    # transform grid back to original space

    # (resolution, resolution, 2) -> (resolution * resolution, 2)
    backward_map_points = backward_map.reshape(-1, 1, 2)
    # The values represent curve's points
    ordered_rich_curve_points = cv2.perspectiveTransform(
        backward_map_points, np.linalg.inv(transform_matrix)
    ).squeeze()
    # (resolution * resolution, 2) -> (resolution, resolution, 2)
    backward_map = ordered_rich_curve_points.reshape(resolution, resolution, 2)

    # flip x and y coordinate: first y, then x
    return np.roll(backward_map, shift=1, axis=-1)


def _snap_corners(polygon: Polygon, corners: List[Point]) -> List[Point]:
    corners = [nearest_points(polygon, point)[0] for point in corners]
    if len(corners) != 4:
        msg = "Unexpected number of corners!"
        raise Exception(msg)
    return corners


def _split_polygon(polygon: Polygon, corners: List[Point]) -> List[LineString]:
    boundary = snap(polygon.boundary, MultiPoint(corners), 0.0000001)
    segments = split(boundary, MultiPoint(corners)).geoms
    if len(segments) not in {4, 5}:
        msg = "Unexpected number of segments!"
        raise Exception(msg)

    if len(segments) == 4:
        return segments  # pyright: ignore [reportReturnType]

    return [
        linemerge([segments[0], segments[4]]),
        segments[1],
        segments[2],
        segments[3],
    ]  # pyright: ignore [reportReturnType]


def _classify_segments(
    segments: Dict[str, LineString], corners: List[Point]
) -> Dict[str, LineString]:
    bbox = box(*MultiLineString(segments).bounds)
    bbox_points = np.array(bbox.exterior.coords[:-1])

    # classify bbox nodes
    data_frame = pd.DataFrame(data=bbox_points, columns=["bbox_x", "bbox_y"])
    name_x = (data_frame.bbox_x == data_frame.bbox_x.min()).replace(
        {True: "left", False: "right"}
    )
    name_y = (data_frame.bbox_y == data_frame.bbox_y.min()).replace(
        {True: "top", False: "bottom"}
    )
    data_frame["name"] = name_y + "-" + name_x
    if len(data_frame.name.unique()) != 4:
        msg = "Unexpected number of unique names!"
        raise Exception(msg)

    # find bbox node to corner node association
    approx_points = np.array([c.xy for c in corners]).squeeze()
    if approx_points.shape != (4, 2):
        msg = "Unexpected number of points!"
        raise Exception(msg)

    assignments = [
        np.roll(np.array(range(4))[::step], shift=i)
        for i in range(4)
        for step in [1, -1]
    ]

    costs = [
        np.linalg.norm(bbox_points - approx_points[assignment], axis=-1).sum()
        for assignment in assignments
    ]

    min_assignment = min(zip(costs, assignments, strict=False))[1]
    data_frame["corner_x"] = approx_points[min_assignment][:, 0]
    data_frame["corner_y"] = approx_points[min_assignment][:, 1]

    # retrieve correct segment and fix direction if necessary
    segment_endpoints = {frozenset([s.coords[0], s.coords[-1]]): s for s in segments}

    def get_directed_segment(start_name: str, end_name: str) -> LineString:
        start = data_frame[data_frame.name == start_name]
        start = (float(start.corner_x), float(start.corner_y))

        end = data_frame[data_frame.name == end_name]
        end = (float(end.corner_x), float(end.corner_y))

        segment = segment_endpoints[frozenset([start, end])]
        if start != segment.coords[0]:
            segment = LineString(reversed(segment.coords))

        return segment

    return {
        "top": get_directed_segment("top-left", "top-right"),
        "bottom": get_directed_segment("bottom-left", "bottom-right"),
        "left": get_directed_segment("top-left", "bottom-left"),
        "right": get_directed_segment("top-right", "bottom-right"),
    }  # pyright: ignore [reportReturnType]
