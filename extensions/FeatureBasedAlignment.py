import os
import cv2
import numpy as np
from extension import ImagePreprocessor

# defaults
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

class FeatureBasedAlignment(ImagePreprocessor):
    def __init__(self, options, path):
        # process reference image
        self.ref_path = os.path.join(os.path.dirname(path), options['reference'])
        self.ref_img = cv2.imread(self.ref_path, cv2.IMREAD_GRAYSCALE)
        self.MAX_FEATURES = options.get('MaxFeatures', MAX_FEATURES)
        self.GOOD_MATCH_PERCENT = options.get('GoodMatchPercent', GOOD_MATCH_PERCENT)
        self.TRANSFORM_2D = options.get('2D', False)
        # Extract keypoints and description of source image
        self.orb = cv2.ORB_create(self.MAX_FEATURES)
        self.to_keypoints, self.to_descriptors = self.orb.detectAndCompute(self.ref_img, None)


    def __str__(self):
        return self.ref_path

    def exclude_files(self):
        return [self.ref_path]

    ''' Image based feature alignment
    Credits: https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/'''
    def apply_filter(self, img, args):
        
        # Convert images to grayscale
        # im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        # im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        img = cv2.normalize(img, 0, 255, norm_type=cv2.NORM_MINMAX)
        
        # Detect ORB features and compute descriptors.
        from_keypoints, from_descriptors = self.orb.detectAndCompute(img, None)
        
        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(from_descriptors, self.to_descriptors, None)
        
        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)
        
        # Remove not so good matches
        numGoodMatches = int(len(matches) * self.GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]
        
        # Draw top matches
        #imMatches = cv2.drawMatches(im1, from_keypoints, im2, self.to_keypoints, matches, None)
        #show('Aligning', imMatches, resize=True)    
        
        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)
        
        for i, match in enumerate(matches):
            points1[i, :] = from_keypoints[match.queryIdx].pt
            points2[i, :] = self.to_keypoints[match.trainIdx].pt
        
        # Find homography
        height, width = self.ref_img.shape
        if self.TRANSFORM_2D:
            m, inliers = cv2.estimateAffine2D(points1, points2)
            return cv2.warpAffine(img, m, (width, height))
        else:        
            # Use homography
            h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
            return cv2.warpPerspective(img, h, (width, height))
        