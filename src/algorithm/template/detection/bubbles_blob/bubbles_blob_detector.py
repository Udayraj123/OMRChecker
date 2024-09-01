import math

import cv2
import numpy as np

from src.algorithm.template.detection.base.field_type_detector import FieldTypeDetector
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger


class BubblesBlobDetector(FieldTypeDetector):
    def read_field(self, field, gray_image, colored_image, file_aggregate_params):
        self.detected_string = "TODO_Blob"

    # source: https://learnopencv.com/blob-detection-using-opencv-python-c/
    def create_bubble_blob_detector(field_block):
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
        box_w, box_h = field_block.bubble_dimensions
        box_zone = box_w * box_h

        # # Change thresholds
        # params.minThreshold = 10;
        # params.maxThreshold = 200;

        # # Filter by Zone.
        params.filterByZone = True
        params.minZone = box_zone * 0.25

        # Filter by Circularity
        # params.filterByCircularity = True
        # params.minCircularity = 0.75

        # # Filter by Convexity
        # params.filterByConvexity = True
        # params.minConvexity = 0.87

        # # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.09

        # Create a detector with the parameters
        return cv2.SimpleBlobDetector_create(params)

    def get_bubble_blob_metrics(
        detector, final_marked, bubble, field_block, draw_keypoints=True
    ):
        x, y = bubble.get_shifted_position(field_block.shifts)

        box_w, box_h = field_block.bubble_dimensions
        box_zone = box_w * box_h

        # TODO: apply bubble_likeliness_threshold
        # Note: assumes direction == "horizontal"
        x_margin = (min(field_block.bubbles_gap - box_w, box_w) * 11 / 14) // 2
        y_margin = (min(field_block.labels_gap - box_h, box_h) * 11 / 14) // 2

        scan_rect = [
            int(y - y_margin),
            int(y + box_h + y_margin),
            int(x - x_margin),
            int(x + box_w + x_margin),
        ]

        scan_box = final_marked[
            scan_rect[0] : scan_rect[1], scan_rect[2] : scan_rect[3]
        ]
        keypoints = detector.detect(scan_box)
        if len(keypoints) < 1:
            # TODO: highlight low confidence metric
            logger.warning(
                f"No bubble-like blob found in scan zone for the bubble {bubble}"
            )

        if len(keypoints) > 1:
            # TODO: highlight low confidence metric
            logger.warning(f"Found multiple blobs in scan zone for the bubble {bubble}")

        zone_ratio = (
            0
            if len(keypoints) == 0
            else np.square(keypoints[0].size / 2) * math.pi / box_zone
        )

        logger.info(
            f"zone_ratio={round(zone_ratio, 2)}",
            [(keypoint.pt, keypoint) for keypoint in keypoints],
        )
        InteractionUtils.show("scan_box", scan_box, 1)

        if draw_keypoints:
            # Draw detected blobs as red circles.
            # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
            im_with_keypoints = cv2.drawKeypoints(
                scan_box,
                keypoints,
                np.array([]),
                (0, 0, 255),
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )
            im_with_keypoints = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)
            # modify output image
            final_marked[
                scan_rect[0] : scan_rect[1], scan_rect[2] : scan_rect[3]
            ] = im_with_keypoints
            InteractionUtils.show("im_with_keypoints", im_with_keypoints, 1)

        return keypoints, scan_rect, zone_ratio
