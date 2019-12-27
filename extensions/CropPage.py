import cv2
import numpy as np
from extension import ImagePreprocessor

MIN_PAGE_AREA = 80000

# TODO: (remove noCropping bool) Automate the case of close up scan(incorrect page)-
# ^Note: App rejects croppeds along with others
def normalize(image):
    return cv2.normalize(image, 0, 255, norm_type=cv2.NORM_MINMAX)


def validateRect(approx):
    # TODO: add logic from app?!
    return len(approx) == 4 and checkMaxCosine(approx.reshape(4, 2))


def angle(p1, p2, p0):
    dx1 = float(p1[0] - p0[0])
    dy1 = float(p1[1] - p0[1])
    dx2 = float(p2[0] - p0[0])
    dy2 = float(p2[1] - p0[1])
    return (dx1 * dx2 + dy1 * dy2) / \
        np.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10)


def checkMaxCosine(approx):
    # assumes 4 pts present
    maxCosine = 0
    minCosine = 1.5
    for i in range(2, 5):
        cosine = abs(angle(approx[i % 4], approx[i - 2], approx[i - 1]))
        maxCosine = max(cosine, maxCosine)
        minCosine = min(cosine, minCosine)
    # TODO add to plot dict
    # print(maxCosine)
    if(maxCosine >= 0.35):
        print('Quadrilateral is not a rectangle.')
        return False
    return True


def findPage(image):
    # Done: find ORIGIN for the quadrants
    # Done, Auto tune! : Get canny parameters tuned
    # (https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/)

    image = normalize(image)
    ret, image = cv2.threshold(image, 200, 255, cv2.THRESH_TRUNC)
    image = normalize(image)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 10))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    """
    # Closing is reverse of Opening, Dilation followed by Erosion.
    A pixel in the original image (either 1 or 0) will be considered 1 only if all the pixels
    under the kernel is 1, otherwise it is eroded (made to zero).
    """
    # Close the small holes, i.e. Complete the edges on canny image
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    edge = cv2.Canny(closed, 185, 55)

    # findContours returns outer boundaries in CW and inner boundaries in ACW
    # order.
    cnts = imutils.grab_contours(
            cv2.findContours(
                edge,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE))
    # hullify to resolve disordered curves due to noise
    cnts = [cv2.convexHull(c) for c in cnts]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    sheet = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_PAGE_AREA:
            continue
        peri = cv2.arcLength(c, True)
        # ez algo -
        # https://en.wikipedia.org/wiki/Ramer–Douglas–Peucker_algorithm
        approx = cv2.approxPolyDP(c, epsilon=0.025 * peri, closed=True)
        # print("Area",cv2.contourArea(c), "Peri", peri)

        # check its rectangle-ness:
        if(validateRect(approx)):
            sheet = np.reshape(approx, (4, -1))
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
            cv2.drawContours(edge, [approx], -1, (255, 255, 255), 10)
            break
        # box = perspective.order_points(box)
    # sobel = cv2.addWeighted(cv2.Sobel(edge, cv2.CV_64F, 1, 0, ksize=3),0.5,cv2.Sobel(edge, cv2.CV_64F, 0, 1, ksize=3),0.5,0,edge)
    # ExcessDo : make it work on killer images
    # edge2 = auto_canny(image_norm)
    # show('Morphed Edges',np.hstack((closed,edge)),1,1)

    return sheet


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


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    maxWidth = max(int(widthA), int(widthB))
    # maxWidth = max(int(np.linalg.norm(br-bl)), int(np.linalg.norm(tr-tl)))

    # compute the height of the new image, which will be the
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # maxHeight = max(int(np.linalg.norm(tr-br)), int(np.linalg.norm(tl-br)))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


class CropPage(ImagePreprocessor):
    
    def apply_filter(self, image, filename):
        """
        TODO later Autorotate:
            - Rotate 90 : check page width:height, CW/ACW? - do CW, then pass to 180 check.
            - Rotate 180 :
                Nope, OMR specific, paper warping may be imperfect. - check markers centroid
                Nope - OCR check
                Match logo - can work, but 'lon' too big and may unnecessarily rotate? - but you know the scale
                Check roll field morphed
        """

        # TODO: need to detect if image is too blurry already! (M1: check
        # noCropping dimensions b4 resizing coz it won't be blurry otherwise _/)
        image = normalize(cv2.GaussianBlur(image, (3, 3), 0))

        if(not self.args['noCropping']):
            # Resize should be done with another preprocessor is needed
            sheet = findPage(image)
            if sheet == []:
                print("\tError: Paper boundary not found! Should you pass --noCropping flag?")
                return None
            else:
                print("Found page corners: \t", sheet.tolist())

            # Warp layer 1
            image = four_point_transform(image, sheet)

        # Return preprocessed image
        return image