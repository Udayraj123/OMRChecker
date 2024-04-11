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


def create_identity_meshgrid(resolution: Union[int, Tuple], with_margin: bool = False):
    if isinstance(resolution, int):
        resolution = (resolution, resolution)

    margin_0 = 0.5 / resolution[0] if with_margin else 0
    margin_1 = 0.5 / resolution[1] if with_margin else 0
    return np.mgrid[
        margin_0 : 1 - margin_0 : complex(0, resolution[0]),
        margin_1 : 1 - margin_1 : complex(0, resolution[1]),
    ].transpose(1, 2, 0)


def scale_map(input_map: np.ndarray, output_shape: Union[int, Tuple[int, int]]):
    try:
        output_shape = (int(output_shape), int(output_shape))
    except (ValueError, TypeError):
        output_shape = tuple(int(v) for v in output_shape)

    H, W, C = input_map.shape
    if H == output_shape[0] and W == output_shape[1]:
        return input_map.copy()

    if np.any(np.isnan(input_map)):
        print(
            "WARNING: scaling maps containing nan values will result in unsteady borders!"
        )

    # The range of x and y coordinates to get a cross product
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    # Fit the interpolator grid to input_map
    interp = RegularGridInterpolator((y, x), input_map, method="linear")

    # xi = values to evaluate at
    xi = create_identity_meshgrid(output_shape, with_margin=False).reshape(-1, 2)

    return interp(xi).reshape(*output_shape, C)


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
    segments: Dict[str, LineString], resolution: int  # Tuple[int, int]
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
    M = cv2.getPerspectiveTransform(src=src, dst=dst)

    polygon = Polygon(linemerge(segments.values()))
    pts = np.array(polygon.exterior.coords).reshape(-1, 1, 2)
    pts = cv2.perspectiveTransform(pts, M).squeeze()

    p_norm = Polygon(pts)

    def transform_line(line):
        line_points = np.array(line.coords).reshape(-1, 1, 2)
        line_points = cv2.perspectiveTransform(line_points, M).squeeze()
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

        assert len(intersections) == 2, "Unexpected number of intersections!"

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

            assert len(intersections) == 2, "Unexpected number of intersections!"

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
        backward_map_points, np.linalg.inv(M)
    ).squeeze()
    # (resolution * resolution, 2) -> (resolution, resolution, 2)
    backward_map = ordered_rich_curve_points.reshape(resolution, resolution, 2)

    # flip x and y coordinate: first y, then x
    backward_map = np.roll(backward_map, shift=1, axis=-1)

    return backward_map


def _snap_corners(polygon: Polygon, corners: List[Point]) -> List[Point]:
    corners = [nearest_points(polygon, point)[0] for point in corners]
    assert len(corners) == 4
    return corners


def _split_polygon(polygon: Polygon, corners: List[Point]) -> List[LineString]:
    boundary = snap(polygon.boundary, MultiPoint(corners), 0.0000001)
    segments = split(boundary, MultiPoint(corners)).geoms
    assert len(segments) in [4, 5]

    if len(segments) == 4:
        return segments

    return [
        linemerge([segments[0], segments[4]]),
        segments[1],
        segments[2],
        segments[3],
    ]


def _classify_segments(
    segments: Dict[str, LineString], corners: List[Point]
) -> Dict[str, LineString]:
    bbox = box(*MultiLineString(segments).bounds)
    bbox_points = np.array(bbox.exterior.coords[:-1])

    # classify bbox nodes
    df = pd.DataFrame(data=bbox_points, columns=["bbox_x", "bbox_y"])
    name_x = (df.bbox_x == df.bbox_x.min()).replace({True: "left", False: "right"})
    name_y = (df.bbox_y == df.bbox_y.min()).replace({True: "top", False: "bottom"})
    df["name"] = name_y + "-" + name_x
    assert len(df.name.unique()) == 4

    # find bbox node to corner node association
    approx_points = np.array([c.xy for c in corners]).squeeze()
    assert approx_points.shape == (4, 2)

    assignments = [
        np.roll(np.array(range(4))[::step], shift=i)
        for i in range(4)
        for step in [1, -1]
    ]

    costs = [
        np.linalg.norm(bbox_points - approx_points[assignment], axis=-1).sum()
        for assignment in assignments
    ]

    min_assignment = min(zip(costs, assignments))[1]
    df["corner_x"] = approx_points[min_assignment][:, 0]
    df["corner_y"] = approx_points[min_assignment][:, 1]

    # retrieve correct segment and fix direction if necessary
    segment_endpoints = {frozenset([s.coords[0], s.coords[-1]]): s for s in segments}

    def get_directed_segment(start_name, end_name):
        start = df[df.name == start_name]
        start = (float(start.corner_x), float(start.corner_y))

        end = df[df.name == end_name]
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
    }
