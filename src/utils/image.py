import cv2
import numpy as np
from matplotlib import pyplot
from shapely import LineString, Point

from src.processors.constants import EDGE_TYPES_IN_ORDER, EdgeType
from src.utils.constants import CLR_WHITE
from src.utils.logger import logger
from src.utils.math import MathUtils

pyplot.rcParams["figure.figsize"] = (10.0, 8.0)
CLAHE_HELPER = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))


class ImageUtils:
    """A Static-only Class to hold common image processing utilities & wrappers over OpenCV functions"""

    @staticmethod
    def read_image_util(file_path, tuning_config):
        encoded_path = file_path
        if tuning_config.outputs.colored_outputs_enabled:
            colored_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            colored_image = cv2.imdecode(
                np.fromfile(encoded_path, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if colored_image is None:
                raise IOError(f"Unable to read image: {file_path}")
            gray_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            gray_image = cv2.imdecode(
                np.fromfile(encoded_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
            )
            if gray_image is None:
                raise IOError(f"Unable to read image: {file_path}")
            colored_image = None

        return gray_image, colored_image

    @staticmethod
    def save_img(path, final_marked):
        cv2.imwrite(path, final_marked)

    @staticmethod
    def save_marked_image(save_marked_dir, file_id, final_marked):
        image_path = str(save_marked_dir.joinpath(file_id))
        logger.info(f"Saving Image to '{image_path}'")
        ImageUtils.save_img(image_path, final_marked)

    @staticmethod
    def resize_to_shape(image_shape, *images):
        h, w = image_shape
        return ImageUtils.resize_multiple(images, w, h)

    @staticmethod
    def resize_to_dimensions(image_dimensions, *images):
        w, h = image_dimensions
        return ImageUtils.resize_multiple(images, w, h)

    @staticmethod
    def resize_multiple(images, u_width=None, u_height=None):
        if len(images) == 1:
            return ImageUtils.resize_single(images[0], u_width, u_height)
        return list(
            map(
                lambda image: ImageUtils.resize_single(image, u_width, u_height), images
            )
        )

    @staticmethod
    def resize_single(image, u_width=None, u_height=None):
        if image is None:
            return None
        h, w = image.shape[:2]
        if u_height is None:
            u_height = int(h * u_width / w)
        if u_width is None:
            u_width = int(w * u_height / h)

        if u_height == h and u_width == w:
            # No need to resize
            return image
        return cv2.resize(image, (int(u_width), int(u_height)))

    @staticmethod
    def get_cropped_warped_rectangle_points(ordered_page_corners):
        # Note: This utility would just find a good size ratio for the cropped image to look more realistic
        # but since we're anyway resizing the image, it doesn't make much sense to use these calculations
        (tl, tr, br, bl) = ordered_page_corners

        length_t = MathUtils.distance(tr, tl)
        length_b = MathUtils.distance(br, bl)
        length_r = MathUtils.distance(tr, br)
        length_l = MathUtils.distance(tl, bl)

        # compute the width of the new image, which will be the
        max_width = max(int(length_t), int(length_b))

        # compute the height of the new image, which will be the
        max_height = max(int(length_r), int(length_l))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image

        warped_points = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype="float32",
        )
        warped_box_dimensions = (max_width, max_height)
        return warped_points, warped_box_dimensions

    @staticmethod
    def grab_contours(cnts):
        # source: imutils package

        # if the length the contours tuple returned by cv2.findContours
        # is '2' then we are using either OpenCV v2.4, v4-beta, or
        # v4-official
        if len(cnts) == 2:
            cnts = cnts[0]

        # if the length of the contours tuple is '3' then we are using
        # either OpenCV v3, v4-pre, or v4-alpha
        elif len(cnts) == 3:
            cnts = cnts[1]

        # otherwise OpenCV has changed their cv2.findContours return
        # signature yet again and I have no idea WTH is going on
        else:
            raise Exception(
                (
                    "Contours tuple must have length 2 or 3, "
                    "otherwise OpenCV changed their cv2.findContours return "
                    "signature yet again. Refer to OpenCV's documentation "
                    "in that case"
                )
            )

        # return the actual contours array
        return cnts

    @staticmethod
    def normalize_single(image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX):
        if image is None or image.max() == image.min():
            return image

        return cv2.normalize(image, None, alpha, beta, norm_type)

    @staticmethod
    def normalize(*images, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX):
        if len(images) == 1:
            return ImageUtils.normalize_single(images[0], alpha, beta, norm_type)
        return list(
            map(
                lambda image: ImageUtils.normalize_single(
                    image, alpha, beta, norm_type
                ),
                images,
            )
        )

    @staticmethod
    def auto_canny(image, sigma=0.93):
        # compute the median of the single channel pixel intensities
        v = np.median(image)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)

        # return the edged image
        return edged

    @staticmethod
    def adjust_gamma(image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    @staticmethod
    def get_control_destination_points_from_contour(
        source_contour, warped_line, max_points=None
    ):
        total_points = len(source_contour)
        if max_points is None:
            max_points = total_points
        assert max_points >= 2
        start, end = warped_line

        warped_line_length = MathUtils.distance(start, end)
        contour_length = 0
        for i in range(1, total_points):
            contour_length += MathUtils.distance(
                source_contour[i], source_contour[i - 1]
            )

        # TODO: replace with this if the assertion passes on more samples
        # cv2_arclength = cv2.arcLength(
        #     np.array(source_contour, dtype="float32"), closed=False
        # )
        # assert (
        #     abs(cv2_arclength - contour_length) < 0.001
        # ), f"{contour_length:.3f} != {cv2_arclength:.3f}"

        # average_min_gap = (contour_length / (max_points - 1)) - 1

        # Initialize with first point mapping
        control_points, warped_points = [source_contour[0]], [start]
        current_arc_length = 0
        # current_arc_gap = 0
        previous_point = None
        for i in range(1, total_points):
            boundary_point, previous_point = source_contour[i], source_contour[i - 1]
            edge_length = MathUtils.distance(boundary_point, previous_point)

            # TODO: figure out an alternative to support maxPoints
            # current_arc_gap += edge_length
            # if current_arc_gap > average_min_gap :
            #     current_arc_length += current_arc_gap
            #     current_arc_gap = 0

            # Including all points for now -
            current_arc_length += edge_length
            length_ratio = current_arc_length / contour_length
            warped_point = MathUtils.get_point_on_line_by_ratio(
                warped_line, length_ratio
            )
            control_points.append(boundary_point)
            warped_points.append(warped_point)

        assert len(warped_points) <= max_points

        # Assert that the float error is not skewing the estimations badly
        assert (
            MathUtils.distance(warped_points[-1], end) / warped_line_length < 0.02
        ), f"{warped_points[-1]} != {end}"

        return control_points, warped_points

    @staticmethod
    def split_patch_contour_on_corners(patch_corners, source_contour):
        ordered_patch_corners, _ = MathUtils.order_four_points(
            patch_corners, dtype="float32"
        )

        # TODO: consider snapping the corners to the contour
        # TODO: consider using split from shapely after snap_corners

        source_contour = np.float32(source_contour)

        edge_contours_map = {
            EdgeType.TOP: [],
            EdgeType.RIGHT: [],
            EdgeType.BOTTOM: [],
            EdgeType.LEFT: [],
        }
        edge_line_strings = {}
        for i, edge_type in enumerate(EDGE_TYPES_IN_ORDER):
            edge_line_strings[edge_type] = LineString(
                [ordered_patch_corners[i], ordered_patch_corners[(i + 1) % 4]]
            )

        #  segments = split(boundary, MultiPoint(corners)).geoms
        for boundary_point in source_contour:
            edge_distances = [
                (
                    Point(boundary_point).distance(edge_line_strings[edge_type]),
                    edge_type,
                )
                for edge_type in EDGE_TYPES_IN_ORDER
            ]
            min_distance, nearest_edge_type = min(edge_distances)
            distance_warning = "*" if min_distance > 10 else ""
            logger.debug(
                f"boundary_point={boundary_point}\t nearest_edge_type={nearest_edge_type}\t min_distance={min_distance:.2f}{distance_warning}"
            )
            # TODO: Each edge contour's points should be in the clockwise order
            edge_contours_map[nearest_edge_type].append(tuple(boundary_point))

        # Add corner points and ensure clockwise order
        for i, edge_type in enumerate(EDGE_TYPES_IN_ORDER):
            start_point, end_point = (
                ordered_patch_corners[i],
                ordered_patch_corners[(i + 1) % 4],
            )
            edge_contour = edge_contours_map[edge_type]

            if len(edge_contour) == 0:
                logger.critical(
                    ordered_patch_corners, source_contour, edge_contours_map
                )
                logger.warning(
                    f"No closest points found for {edge_type}: {edge_contours_map}"
                )
            else:
                # Ensure correct order
                if MathUtils.distance(
                    start_point, edge_contour[-1]
                ) < MathUtils.distance(start_point, edge_contour[0]):
                    edge_contours_map[edge_type].reverse()

            # Each contour should necessarily start & end with a corner point
            edge_contours_map[edge_type].insert(0, tuple(start_point))
            edge_contours_map[edge_type].append(tuple(end_point))

        return ordered_patch_corners, edge_contours_map

    @staticmethod
    def get_vstack_image_grid(debug_vstack):
        padded_hstack = [
            ImageUtils.get_padded_hstack(hstack) for hstack in debug_vstack
        ]
        return ImageUtils.get_padded_vstack(padded_hstack)

    @staticmethod
    def get_padded_hstack(hstack):
        max_height = max(image.shape[0] for image in hstack)
        padded_hstack = [
            ImageUtils.pad_image_to_height(image, max_height) for image in hstack
        ]

        return np.hstack(padded_hstack)

    @staticmethod
    def get_padded_vstack(vstack):
        max_width = max(image.shape[1] for image in vstack)
        padded_vstack = [
            ImageUtils.pad_image_to_width(image, max_width) for image in vstack
        ]

        return np.vstack(padded_vstack)

    @staticmethod
    def pad_image_to_height(image, max_height, value=CLR_WHITE):
        return cv2.copyMakeBorder(
            image,
            0,
            max_height - image.shape[0],
            0,
            0,
            cv2.BORDER_CONSTANT,
            value,
        )

    @staticmethod
    def pad_image_to_width(image, max_width, value=CLR_WHITE):
        # Pads from the right side
        return cv2.copyMakeBorder(
            image,
            0,
            0,
            0,
            max_width - image.shape[1],
            cv2.BORDER_CONSTANT,
            value,
        )

    def pad_image_from_center(image, padding_width, padding_height=0, value=255):
        # TODO: support colored images for this util
        input_height, input_width = image.shape[:2]
        pad_range = [
            padding_height,
            padding_height + input_height,
            padding_width,
            padding_width + input_width,
        ]
        white_image = value * np.ones(
            (padding_height * 2 + input_height, padding_width * 2 + input_width),
            np.uint8,
        )
        white_image[pad_range[0] : pad_range[1], pad_range[2] : pad_range[3]] = image

        return white_image, pad_range

    @staticmethod
    def clip_zone_to_image_bounds(rectangle, image):
        h, w = image.shape[:2]
        zone_start, zone_end = rectangle
        # Clip to Image top left
        zone_start = [max(0, zone_start[0]), max(0, zone_start[1])]
        zone_end = [max(0, zone_end[0]), max(0, zone_end[1])]
        # Clip to Image bottom right
        zone_start = [min(w, zone_start[0]), min(h, zone_start[1])]
        zone_end = [min(w, zone_end[0]), min(h, zone_end[1])]
        return [zone_start, zone_end]

    @staticmethod
    def rotate(image, rotation, keep_original_shape):
        if keep_original_shape:
            image_shape = image.shape[0:2]
            image = cv2.rotate(image, rotation)
            return ImageUtils.resize_to_shape(image_shape, image)
        else:
            return cv2.rotate(image, rotation)

    @staticmethod
    def overlay_image(image1, image2, transparency=0.5):
        overlay = image1.copy()
        cv2.addWeighted(
            overlay,
            transparency,
            image2,
            1 - transparency,
            0,
            overlay,
        )
        return overlay
