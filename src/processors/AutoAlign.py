from src.processors.interfaces.ImageTemplatePreprocessor import (
    ImageTemplatePreprocessor,
)
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger
import cv2
from src.utils.image import ImageUtils
import numpy as np 
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
        for angle in range(0,20,5):
            logger.info(angle)
            logger.info("SHOW SOMETHING HERE")
            reference_image=self.rotate_image(self.reference_image,angle)
            InteractionUtils.show("reference",reference_image,pause=True)
            for rotation in rotations:
                rotated_img=cv2.rotate(image,rotation)
                res=cv2.matchTemplate(rotated_img,reference_image,cv2.TM_CCOEFF)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                logger.info(rotation,max_val)
                if max_val>best_val:
                    best_val=max_val
                    best_rotation=rotation
                    values.append(max_val)
        logger.info("AutoRotate Applied with rotation",best_rotation,"best value",best_val,"having values",values)
        # best_rotation=self.get_best_match(image)
        if best_rotation is None:
            return image,colored_image,_template
        
        image=cv2.rotate(image,best_rotation)
        colored_image=cv2.rotate(colored_image,best_rotation)
        return image,colored_image,_template
    
    def exclude_files(self):
        path = self.get_relative_path(self.options["referenceImage"])
        return [path]
    
    def rotate_image(self,image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result
    
    def get_best_match(self,image):
        res=cv2.matchTemplate(image,self.reference_image,cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        best_val,best_rotation=max_val,None
        rotations=[None,cv2.ROTATE_90_CLOCKWISE,cv2.ROTATE_180,cv2.ROTATE_90_COUNTERCLOCKWISE]
        values=[best_val]
        scale=2
         
        for applied_scale in np.arange(
            -scale,
            scale,
            0.5,
        ):
            if applied_scale<=0:
                continue
            if applied_scale>0:
                rescaled_marker = ImageUtils.resize_util(
                self.reference_image,
                u_width=int(min(self.reference_image.shape[0] * applied_scale,image.shape[0])),
                u_height=int(min(self.reference_image.shape[1] * applied_scale,image.shape[1])),
                )
            else:
                applied_scale=-1/applied_scale
                rescaled_marker = ImageUtils.resize_util(
                self.reference_image,
                u_width=int(self.reference_image.shape[0] * applied_scale),
                u_height=int(self.reference_image.shape[1] * applied_scale),
                )
            for rotation in rotations:
                rotated_img=image
                if rotation is not None:
                    rotated_img=cv2.rotate(image,rotation)
                res=cv2.matchTemplate(rotated_img,rescaled_marker,cv2.TM_CCOEFF)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if max_val>best_val:
                    best_val=max_val
                    best_rotation=rotation
                    values.append(max_val)
        logger.info("AutoRotate Applied with rotation",best_rotation,"having value",best_val,"having values",values)
        return best_rotation
            