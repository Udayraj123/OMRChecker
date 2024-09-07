"""
Image based feature alignment
Credits: https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
"""

import cv2
import numpy as np

from src.processors.interfaces.ImageTemplatePreprocessor import (
    ImageTemplatePreprocessor,
)
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils

# TODO: support WarpOnPointsCommon?


class FeatureBasedAlignment(ImageTemplatePreprocessor):
    def get_class_name(self):
        return f"FeatureBasedAlignment"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        options = self.options

        # process reference image
        self.ref_path = self.relative_dir.joinpath(options["reference"])
        ref_img = cv2.imread(str(self.ref_path), cv2.IMREAD_GRAYSCALE)
        self.ref_img = ImageUtils.resize_to_shape(self.processing_image_shape, ref_img)
        # get options with defaults
        self.max_features = int(options.get("maxFeatures", 500))
        self.good_match_percent = options.get("goodMatchPercent", 0.10)

        matcher_type = options.get("matcherType", "BRUTEFORCE_HAMMING")
        if matcher_type == "NORM_HAMMING":
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif matcher_type == "BRUTEFORCE_HAMMING":
            self.matcher = cv2.DescriptorMatcher_create(
                cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
            )

        self.transform_2_d = options.get("2d", True)
        # Extract keypoints and description of source image
        self.orb = cv2.ORB_create(self.max_features)
        self.to_keypoints, self.to_descriptors = self.orb.detectAndCompute(
            self.ref_img, None
        )

    def __str__(self):
        return self.ref_path.name

    def exclude_files(self):
        return [self.ref_path]

    def apply_filter(self, image, colored_image, _template, _file_path):
        config = self.tuning_config
        # Convert images to grayscale
        # im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        # im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        image = cv2.normalize(image, None, 0, 255, norm_type=cv2.NORM_MINMAX)

        # Detect Oriented Fast and Rotated Brief (ORB) features and compute descriptors.
        from_keypoints, from_descriptors = self.orb.detectAndCompute(image, None)

        # Match features.

        matches = np.array(
            self.matcher.match(from_descriptors, self.to_descriptors, None)
        )

        # Sort matches by score
        matches = sorted(matches, key=lambda x: x.distance, reverse=False)

        # Note: we can also enhance the algorithm to discard matches after the first "large jump" if present
        # print(list(map(lambda x: x.distance, matches)))

        # Remove not so good matches
        num_good_matches = int(len(matches) * self.good_match_percent)
        matches = matches[:num_good_matches]

        # Draw top matches
        if config.outputs.show_image_level > 2:
            im_matches = cv2.drawMatches(
                image, from_keypoints, self.ref_img, self.to_keypoints, matches, None
            )
            im_matches = ImageUtils.resize_single(
                im_matches, u_height=config.outputs.display_image_dimensions[1]
            )
            InteractionUtils.show(
                "Alignment Matches", im_matches, resize_to_height=True, config=config
            )

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = from_keypoints[match.queryIdx].pt
            points2[i, :] = self.to_keypoints[match.trainIdx].pt

        # Find homography
        height, width = self.ref_img.shape[:2]
        if self.transform_2_d:
            # Note: estimateAffinePartial2D might save on computation as we expect no noise in the data
            m, _inliers = cv2.estimateAffine2D(points1, points2)
            # 2D == in image plane:

            warped_image = cv2.warpAffine(image, m, (width, height))

            if config.outputs.colored_outputs_enabled:
                colored_image = cv2.warpAffine(colored_image, m, (width, height))
        else:
            # Use homography
            h, _mask = cv2.findHomography(points1, points2, cv2.RANSAC)
            # 3D == perspective from out of plane:
            warped_image = cv2.warpPerspective(image, h, (width, height))

            if config.outputs.colored_outputs_enabled:
                colored_image = cv2.warpPerspective(colored_image, h, (width, height))

        return warped_image, colored_image, _template
