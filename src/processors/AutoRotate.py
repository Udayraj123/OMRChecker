import cv2
import numpy as np

from src.processors.interfaces.ImageTemplatePreprocessor import (
    ImageTemplatePreprocessor,
)
from src.utils.interaction import InteractionUtils
# from src.utils.image import ImageUtils
# from src.utils.interaction import InteractionUtils
from src.utils.logger import logger
from src.utils.image import ImageUtils 


class AutoRotate(ImageTemplatePreprocessor):
    __is_internal_preprocessor__ = False

    def get_class_name(self):
        return "AutoRotate"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        path = self.get_relative_path(self.options["referenceImage"])
        self.reference_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        self.marker_dimensions = self.options.get("markerDimensions",None)
        self.resized_reference=self.reference_image
        self.threshold=self.options.get("threshold",None)
        if self.marker_dimensions:
            self.resized_reference=ImageUtils.resize_to_dimensions(self.reference_image,self.marker_dimensions)

    def apply_filter(self, image, colored_image, _template, file_path):
        # for rotation in rotation rotate and match
        # methods = [
        #     "cv.TM_CCOEFF",
        #     "cv.TM_CCOEFF_NORMED",
        #     "cv.TM_CCORR",
        #     "cv.TM_CCORR_NORMED",
        #     "cv.TM_SQDIFF",
        #     "cv.TM_SQDIFF_NORMED",
        # ]
        img_dimensions=image.shape 
        best_val, best_rotation = -1, None
        rotations = [
            None,
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_180,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        ]
        for rotation in rotations:
            rotated_img=image
            if rotation is not None:
                rotated_img = cv2.rotate(image, rotation)
            rotated_img=ImageUtils.resize_to_shape(rotated_img,img_dimensions)
          
            res = cv2.matchTemplate(rotated_img, self.resized_reference, cv2.TM_CCOEFF)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            logger.info(rotation, max_val)
            if max_val > best_val:
                best_val = max_val
                best_rotation = rotation
        if self.threshold is not None:
            if self.threshold["value"]>best_val:
                if self.threshold["passthrough"]:
                    logger.warning("Best value for autorotate is below threshold")
                else:
                    logger.error("Best value for autorotate is below threshold")
                    raise Exception(
                        f"Autorotation score is below threshold, please check the reference marker and image"
                    )
        logger.info(
            "AutoRotate Applied with rotation",
            best_rotation,
            "best value",
            best_val
        )
        if best_rotation is None:
            return image, colored_image, _template
        image = cv2.rotate(image, best_rotation)
        image=ImageUtils.resize_to_shape(image,img_dimensions)
        if self.tuning_config.outputs.colored_outputs_enabled:
            colored_image = cv2.rotate(colored_image, best_rotation)
            colored_image=ImageUtils.resize_to_shape(colored_image,img_dimensions)
        return image, colored_image, _template

    def exclude_files(self):
        path = self.get_relative_path(self.options["referenceImage"])
        return [path]

    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(
            image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR
        )
        return result
