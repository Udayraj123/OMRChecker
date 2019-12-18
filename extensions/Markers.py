import os
import cv2
from extension import ImagePreprocessor
import numpy as np
import imutils

import utils
import config

# defaults
MARKER_FILE = "omr_marker.jpg"
ERODE_SUB_OFF = 1

thresholdVar = 0.41    # max threshold difference for template matching
thresholdCircle = 0.3
marker_rescale_range = (35, 100)
marker_rescale_steps = 10

class Markers(ImagePreprocessor):
    def __init__(self, marker_ops, path):
        # process markers
        self.marker_path = os.path.join(
            os.path.dirname(path), marker_ops.get("RelativePath", MARKER_FILE))

        if(not os.path.exists(self.marker_path)):
            print(
                "Error: Marker not found at path provided in template:",
                self.marker_path)
            exit(31)

        marker = cv2.imread(self.marker_path, cv2.IMREAD_GRAYSCALE)

        if("SheetToMarkerWidthRatio" in marker_ops):
            marker = utils.resize_util(marker, config.uniform_width /
                                    int(marker_ops["SheetToMarkerWidthRatio"]))
        marker = cv2.GaussianBlur(marker, (5, 5), 0)
        marker = cv2.normalize(
            marker,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX)
        marker -= cv2.erode(marker, kernel=np.ones((5, 5)), iterations=5)
        
        self.marker = marker
        self.ERODE_SUB_OFF = marker_ops.get("ErodeSubOff", ERODE_SUB_OFF)
        self.thresholdCircles = []

    def __str__(self):
        return self.marker_path

    def exclude_files(self):
        return [self.marker_path]

    def apply_filter(self, image_norm, curr_filename):

        if self.ERODE_SUB_OFF:
            image_eroded_sub = utils.normalize_util(image_norm)
        else:
            image_eroded_sub = utils.normalize_util(image_norm
                                            - cv2.erode(image_norm,
                                                        kernel=np.ones((5, 5)),
                                                        iterations=5))
        # Quads on warped image
        quads = {}
        h1, w1 = image_eroded_sub.shape[:2]
        midh, midw = h1 // 3, w1 // 2
        origins = [[0, 0], [midw, 0], [0, midh], [midw, midh]]
        quads[0] = image_eroded_sub[0:midh, 0:midw]
        quads[1] = image_eroded_sub[0:midh, midw:w1]
        quads[2] = image_eroded_sub[midh:h1, 0:midw]
        quads[3] = image_eroded_sub[midh:h1, midw:w1]

        # Draw Quadlines
        image_eroded_sub[:, midw:midw + 2] = 255
        image_eroded_sub[midh:midh + 2, :] = 255

        best_scale, allMaxT = utils.getBestMatch(image_eroded_sub, self.marker)
        if(best_scale is None):
            # TODO: Plot and see performance of marker_rescale_range
            if(config.showimglvl >= 1):
                utils.show('Quads', image_eroded_sub)
            return None

        optimal_marker = imutils.resize_util_h(
            self.marker if self.ERODE_SUB_OFF else self.marker, u_height=int(
                self.marker.shape[0] * best_scale))
        h, w = optimal_marker.shape[:2]
        centres = []
        sumT, maxT = 0, 0
        print("Matching Marker:\t", end=" ")
        for k in range(0, 4):
            res = cv2.matchTemplate(quads[k], optimal_marker, cv2.TM_CCOEFF_NORMED)
            maxT = res.max()
            print("Q" + str(k + 1) + ": maxT", round(maxT, 3), end="\t")
            if(maxT < thresholdCircle or abs(allMaxT - maxT) >= thresholdVar):
                # Warning - code will stop in the middle. Keep Threshold low to
                # avoid.
                print(
                    curr_filename,
                    "\nError: No circle found in Quad",
                    k + 1,
                    "\n\tthresholdVar",
                    thresholdVar,
                    "maxT",
                    maxT,
                    "allMaxT",
                    allMaxT,
                    "Should you pass --noCropping flag?")
                if(config.showimglvl >= 1):
                    utils.show("no_pts_" + curr_filename, image_eroded_sub, 0)
                    utils.show("res_Q" + str(k + 1), res, 1)
                return None

            pt = np.argwhere(res == maxT)[0]
            pt = [pt[1], pt[0]]
            pt[0] += origins[k][0]
            pt[1] += origins[k][1]
            # print(">>",pt)
            image_norm = cv2.rectangle(image_norm, tuple(
                pt), (pt[0] + w, pt[1] + h), (150, 150, 150), 2)
            # display:
            image_eroded_sub = cv2.rectangle(
                image_eroded_sub,
                tuple(pt),
                (pt[0] + w,
                pt[1] + h),
                (50,
                50,
                50) if self.ERODE_SUB_OFF else (
                    155,
                    155,
                    155),
                4)
            centres.append([pt[0] + w / 2, pt[1] + h / 2])
            sumT += maxT
        print("Optimal Scale:", best_scale)
        # analysis data
        self.thresholdCircles.append(sumT / 4)

        image_norm = utils.four_point_transform(image_norm, np.array(centres))
        # appendSaveImg(1,image_eroded_sub)
        # appendSaveImg(1,image_norm)

        utils.appendSaveImg(2, image_eroded_sub)
        # Debugging image -
        # res = cv2.matchTemplate(image_eroded_sub,optimal_marker,cv2.TM_CCOEFF_NORMED)
        # res[ : , midw:midw+2] = 255
        # res[ midh:midh+2, : ] = 255
        # show("Markers Matching",res)
        if(config.showimglvl >= 2 and config.showimglvl < 4):
            image_eroded_sub = utils.resize_util_h(image_eroded_sub, image_norm.shape[0])
            image_eroded_sub[:, -5:] = 0
            h_stack = np.hstack((image_eroded_sub, image_norm))
            utils.show("Warped: " + curr_filename, utils.resize_util(h_stack,
                                                        int(config.display_width * 1.6)), 0, 0, [0, 0])
        # iterations : Tuned to 2.
        # image_eroded_sub = image_norm - cv2.erode(image_norm, kernel=np.ones((5,5)),iterations=2)
        return image_norm