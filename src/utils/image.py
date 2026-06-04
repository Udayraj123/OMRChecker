"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.logger import logger

plt.rcParams["figure.figsize"] = (10.0, 8.0)
CLAHE_HELPER = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))


class ImageUtils:
    """A Static-only Class to hold common image processing utilities & wrappers over OpenCV functions"""

    @staticmethod
    def save_img(path, final_marked):
        logger.info(f"Saving Image to '{path}'")
        cv2.imwrite(path, final_marked)

    @staticmethod
    def resize_util(img, u_width, u_height=None):
        if u_height is None:
            h, w = img.shape[:2]
            u_height = int(h * u_width / w)
        return cv2.resize(img, (int(u_width), int(u_height)))

    @staticmethod
    def resize_util_h(img, u_height, u_width=None):
        if u_width is None:
            h, w = img.shape[:2]
            u_width = int(w * u_height / h)
        return cv2.resize(img, (int(u_width), int(u_height)))

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
    def normalize_util(img, alpha=0, beta=255):
        return cv2.normalize(img, alpha, beta, norm_type=cv2.NORM_MINMAX)

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
    def four_point_transform(image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = ImageUtils.order_points(pts)
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

    @staticmethod
    def _resolve_pages(page_spec, doc_len):
        """Resolve user page spec into list of 1-based page numbers.

        None          → all pages [1, 2, ..., doc_len]
        int 3         → [3]
        str "3"       → [3]
        str "1-5"     → [1, 2, 3, 4, 5]
        str "3-"      → [3, 4, ..., doc_len]
        list [1,"3-5"]→ [1, 3, 4, 5]
        """
        if page_spec is None:
            return list(range(1, doc_len + 1))
        if isinstance(page_spec, int):
            return [page_spec]
        if isinstance(page_spec, str):
            if "-" in page_spec:
                parts = page_spec.split("-", 1)
                start = int(parts[0]) if parts[0] else 1
                end = int(parts[1]) if parts[1] else doc_len
                return list(range(start, end + 1))
            return [int(page_spec)]
        if isinstance(page_spec, list):
            result = []
            seen = set()
            for item in page_spec:
                for page in ImageUtils._resolve_pages(item, doc_len):
                    if page in seen:
                        continue
                    seen.add(page)
                    result.append(page)
            return result
        return []

    @staticmethod
    def _detect_pdf_native_dpi(doc, pages):
        """Auto-detect native DPI from embedded images in PDF pages.

        Compares embedded image pixel dimensions against page dimensions
        (in points, 1 pt = 1/72 inch) to find the DPI at which images
        would render 1:1. Falls back to 72 if no images are found.
        """
        dpis = []
        for p in pages:
            page = doc[p]
            pw, ph = page.rect.width, page.rect.height
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                img_data = doc.extract_image(xref)
                iw, ih = img_data["width"], img_data["height"]
                dpi_w = int(iw * 72 / pw)
                dpi_h = int(ih * 72 / ph)
                dpis.append(max(dpi_w, dpi_h))
        return max(dpis) if dpis else 72

    @staticmethod
    def load_omr_image(file_path, tuning_config):
        """Load OMR image from file. Returns list of (display_name, image_array) tuples."""
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            import fitz

            try:
                doc = fitz.open(str(file_path))
            except Exception as e:
                logger.error(f"Failed to open PDF: '{file_path}' - {e}")
                return []

            try:
                pdf_params = tuning_config.pdf_params
                doc_len = len(doc)
                logger.info(
                    f"Processing PDF '{file_path.name}' "
                    f"(pdf_dpi={pdf_params.pdf_dpi}, pdf_page={pdf_params.pdf_page})"
                )
                user_pages = ImageUtils._resolve_pages(
                    pdf_params.pdf_page, doc_len
                )
                # Convert 1-based user pages to 0-based fitz indices,
                # filter out-of-range pages with warning.
                pages = []
                for p in user_pages:
                    if p < 1 or p > doc_len:
                        logger.warning(
                            f"PDF page {p} out of range for '{file_path}' "
                            f"(has {doc_len} pages), skipping page."
                        )
                    else:
                        pages.append(p - 1)

                # Resolve rendering DPI
                dpi = pdf_params.pdf_dpi
                if dpi == "auto":
                    dpi = ImageUtils._detect_pdf_native_dpi(doc, pages)
                    logger.info(f"Auto-detected PDF native DPI: {dpi}")

                images = []
                for p in pages:
                    page = doc[p]
                    mat = fitz.Matrix(dpi / 72, dpi / 72)
                    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        pix.height, pix.width
                    )
                    name = (
                        f"{file_path.stem}_p{p + 1}.png"
                        if len(doc) > 1
                        else f"{file_path.stem}.png"
                    )
                    images.append((name, img))
            except Exception as e:
                logger.error(f"Failed to render PDF: '{file_path}' - {e}")
                return []
            finally:
                doc.close()
            return images
        else:
            img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            return [(file_path.name, img)]

    @staticmethod
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
