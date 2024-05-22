from src.processors.interfaces.ImageTemplatePreprocessor import (
    ImageTemplatePreprocessor,
)
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger
import cv2


class AutoAlign(ImageTemplatePreprocessor):
    __is_internal_preprocessor__ = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        path = self.get_relative_path(self.options["referenceImage"])
        self.reference_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    def apply_filter(self, image, colored_image, _template, file_path):
        # for rotation in rotation rotate and match
        methods = [
            "cv.TM_CCOEFF",
            "cv.TM_CCOEFF_NORMED",
            "cv.TM_CCORR",
            "cv.TM_CCORR_NORMED",
            "cv.TM_SQDIFF",
            "cv.TM_SQDIFF_NORMED",
        ]
        # resized_reference=self.reference_image
        res=cv2.matchTemplate(image,self.reference_image,cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        best_val,best_rotation=max_val,None
        rotations=[cv2.ROTATE_90_CLOCKWISE,cv2.ROTATE_180,cv2.ROTATE_90_COUNTERCLOCKWISE]
        values=[best_val]
        for rotation in rotations:
            rotated_img=cv2.rotate(image,rotation)
            res=cv2.matchTemplate(rotated_img,self.reference_image,cv2.TM_CCOEFF)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            logger.info(rotation,max_val)
            values.append(max_val)
            if max_val>best_val:
                best_val=max_val
                best_rotation=rotation
        logger.info("AutoRotate Applied with rotation",best_rotation,"having values",values)
        if best_rotation is None:
            return image,colored_image,_template
        
        image=cv2.rotate(image,best_rotation)
        colored_image=cv2.rotate(colored_image,best_rotation)
        return image,colored_image,_template
    
    def exclude_files(self):
        path = self.get_relative_path(self.options["referenceImage"])
        return [path]