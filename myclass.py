import cv2
import numpy as np
marker_rescale_range =  (35, 100)
marker_rescale_steps =  10
def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect
    #used this fucthon to crop based four point of four side of paper that devide based maker
def four_point_transform(image, pts):
        # obtain a consistent order of the points and unpack them
        # individually

        rect = order_points(pts)
        (tl, tr, br, bl) = rect

        # compute the width of the new image, which will be the
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

        max_width = max(int(width_a), int(width_b))
        # max_width = max(int(np.linalg.norm(br-bl)), int(np.linalg.norm(tr-tl)))

        # compute the height of the new image, which will be the
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        # max_height = max(int(np.linalg.norm(tr-br)), int(np.linalg.norm(tl-br)))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype="float32",
        )

        transform_matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, transform_matrix, (max_width, max_height))

        # return the warped image
        return warped
def normalize_util(img, alpha=0, beta=255):
        return cv2.normalize(img, alpha, beta, norm_type=cv2.NORM_MINMAX)
def resize_util_h(img, u_height, u_width=None):
            if u_width is None:
                h, w = img.shape[:2]
                u_width = int(w * u_height / h)
            return cv2.resize(img, (int(u_width), int(u_height)))    
def resize_util(img, u_width, u_height=None):
            if u_height is None:
                h, w = img.shape[:2]
                u_height = int(h * u_width / w)
            return cv2.resize(img, (int(u_width), int(u_height)))
marker = cv2.imread("savedmar.jpeg", cv2.IMREAD_GRAYSCALE)
def getBestMatch(image_eroded_sub):
            cv2.imwrite('image_eroded_sub.jpg', image_eroded_sub)    
            descent_per_step = (
                marker_rescale_range[1] - marker_rescale_range[0]
            ) // marker_rescale_steps

            _h, _w = marker.shape[:2]
            res, best_scale = None, None
            all_max_t = 0

            for r0 in np.arange(marker_rescale_range[1],marker_rescale_range[0],-1 * descent_per_step,):  # reverse order
                s = float(r0 * 1 / 100)
                if s == 0.0:
                    continue
                rescaled_marker = resize_util_h(marker, u_height=int(_h * s))
                # res is the black image with white dots
                res = cv2.matchTemplate(image_eroded_sub, rescaled_marker, cv2.TM_CCOEFF_NORMED )
                # cv2.imshow(str(5+r0),res)
                max_t = res.max()
                if all_max_t < max_t:
                   # print("-------------------------------------------------------------\n")
                    #print('Scale: '+str(s)+', Circle Match: '+str(round(max_t*100,2))+'%')
                    #print("-------------------------------------------------------------\n")
                    best_scale, all_max_t = s, max_t
            return best_scale, all_max_t
#-------------------------- find
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
def normalize(image):
    return cv2.normalize(image, 0, 255, norm_type=cv2.NORM_MINMAX)