"""
Image based feature alignment
Credits: https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
"""

import cv2
import numpy as np

from src.config import CONFIG_DEFAULTS as config
from src.utils.imgutils import MainOperations

from .interfaces.ImagePreprocessor import ImagePreprocessor


class FeatureBasedAlignment(ImagePreprocessor):
    def __init__(self, options, path):
        # process reference image
        self.ref_path = path.joinpath(options["reference"])
        self.ref_img = cv2.imread(str(self.ref_path), cv2.IMREAD_GRAYSCALE)
        # get options with defaults
        self.max_features = int(options.get("maxFeatures", 500))
        self.good_match_percent = options.get("goodMatchPercent", 0.15)
        self.transform_2_d = options.get("2d", False)
        # Extract keypoints and description of source image
        self.orb = cv2.ORB_create(self.max_features)
        self.to_keypoints, self.to_descriptors = self.orb.detectAndCompute(
            self.ref_img, None
        )

    def __str__(self):
        return self.ref_path.name

    def exclude_files(self):
        return [self.ref_path]

    def apply_filter(self, img, _args):

        # Convert images to grayscale
        # im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        # im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        img = cv2.normalize(img, 0, 255, norm_type=cv2.NORM_MINMAX)

        # Detect ORB features and compute descriptors.
        from_keypoints, from_descriptors = self.orb.detectAndCompute(img, None)

        # Match features.
        matcher = cv2.DescriptorMatcher_create(
            cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
        )
        matches = np.array(matcher.match(from_descriptors, self.to_descriptors, None))

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        num_good_matches = int(len(matches) * self.good_match_percent)
        matches = matches[:num_good_matches]

        # Draw top matches
        if config.outputs.show_image_level > 2:
            im_matches = cv2.drawMatches(
                img, from_keypoints, self.ref_img, self.to_keypoints, matches, None
            )
            MainOperations.show("Aligning", im_matches, resize=True)

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = from_keypoints[match.queryIdx].pt
            points2[i, :] = self.to_keypoints[match.trainIdx].pt

        # Find homography
        height, width = self.ref_img.shape
        if self.transform_2_d:
            m, _inliers = cv2.estimateAffine2D(points1, points2)
            return cv2.warpAffine(img, m, (width, height))

        # Use homography
        h, _mask = cv2.findHomography(points1, points2, cv2.RANSAC)
        return cv2.warpPerspective(img, h, (width, height))
