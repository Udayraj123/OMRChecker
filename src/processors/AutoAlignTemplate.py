import cv2
import numpy as np

from src.processors.interfaces.ImageTemplatePreprocessor import (
    ImageTemplatePreprocessor,
)
from src.utils.image import CLAHE_HELPER, ImageUtils
from src.utils.interaction import InteractionUtils


class AutoAlignTemplate(ImageTemplatePreprocessor):
    # Note: 'auto_align' enables automatic template alignment, use if the scans show slight misalignments.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        default_options = {
            "match_col": 5,
            "max_steps": 20,
            "morph_threshold": 60,  # 60 for Mobile images, 40 for scanned Images
            "stride": 1,
            "thickness": 3,
        }
        self.options = {**default_options, **self.options}

    def __str__(self):
        return "AutoAlignTemplate"

    def exclude_files(self):
        return []

    def apply_filter(self, image, _colored_image, template, _file_path):
        config = self.tuning_config
        image_instance_ops = self.image_instance_ops
        morph = image.copy()
        image_instance_ops.append_save_img(3, morph)

        # Note: clahe is good for morphology, bad for thresholding
        morph = CLAHE_HELPER.apply(morph)
        image_instance_ops.append_save_img(3, morph)
        # Remove shadows further, make columns/boxes darker (less gamma)
        morph = ImageUtils.adjust_gamma(morph, config.threshold_params.GAMMA_LOW)
        # TODO: all numbers should come from either constants or config
        _, morph = cv2.threshold(morph, 220, 220, cv2.THRESH_TRUNC)
        morph = ImageUtils.normalize_util(morph)
        image_instance_ops.append_save_img(3, morph)
        if config.outputs.show_image_level >= 4:
            InteractionUtils.show("Auto align preparation", morph, 0, 1, config=config)

        # Step 1: Apply morphology, thresholding etc for processable output

        # Open : erode then dilate
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))
        morph_v = cv2.morphologyEx(morph, cv2.MORPH_OPEN, v_kernel, iterations=3)
        _, morph_v = cv2.threshold(morph_v, 200, 200, cv2.THRESH_TRUNC)
        morph_v = 255 - ImageUtils.normalize_util(morph_v)

        if config.outputs.show_image_level >= 3:
            InteractionUtils.show("morphed_vertical", morph_v, 1, 1, config=config)

        image_instance_ops.append_save_img(3, morph_v)

        match_col, max_steps, morph_threshold, align_stride, thickness = map(
            self.options.get,
            [
                "match_col",
                "max_steps",
                "morph_threshold",
                "stride",
                "thickness",
            ],
        )
        _, morph_v = cv2.threshold(morph_v, morph_threshold, 255, cv2.THRESH_BINARY)
        # kernel best tuned to 5x5 now
        morph_v = cv2.erode(morph_v, np.ones((5, 5), np.uint8), iterations=2)

        image_instance_ops.append_save_img(3, morph_v)
        # h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
        # morph_h = cv2.morphologyEx(morph, cv2.MORPH_OPEN, h_kernel, iterations=3)
        # ret, morph_h = cv2.threshold(morph_h,200,200,cv2.THRESH_TRUNC)
        # morph_h = 255 - normalize_util(morph_h)
        # InteractionUtils.show("morph_h",morph_h,0,1,config=config)
        # _, morph_h = cv2.threshold(morph_h,morph_threshold,255,cv2.THRESH_BINARY)
        # morph_h = cv2.erode(morph_h,  np.ones((5,5),np.uint8), iterations = 2)
        if config.outputs.show_image_level >= 3:
            InteractionUtils.show("morph_thr_eroded", morph_v, 0, 1, config=config)

        image_instance_ops.append_save_img(6, morph_v)

        # Step 2: Find Shifts for the field_blocks

        # template relative alignment code
        for field_block in template.field_blocks:
            s, d = field_block.origin, field_block.dimensions
            shift, steps = 0, 0
            while steps < max_steps:
                left_mean = np.mean(
                    morph_v[
                        s[1] : s[1] + d[1],
                        s[0]
                        + shift
                        - thickness : -thickness
                        + s[0]
                        + shift
                        + match_col,
                    ]
                )
                right_mean = np.mean(
                    morph_v[
                        s[1] : s[1] + d[1],
                        s[0]
                        + shift
                        - match_col
                        + d[0]
                        + thickness : thickness
                        + s[0]
                        + shift
                        + d[0],
                    ]
                )

                # For demonstration purposes-
                # if(field_block.name == "int1"):
                #     ret = morph_v.copy()
                #     cv2.rectangle(ret,
                #                   (s[0]+shift-thickness,s[1]),
                #                   (s[0]+shift+thickness+d[0],s[1]+d[1]),
                #                   constants.CLR_WHITE,
                #                   3)
                #     appendSaveImg(6,ret)
                # print(shift, left_mean, right_mean)
                left_shift, right_shift = left_mean > 100, right_mean > 100
                if left_shift:
                    if right_shift:
                        break
                    else:
                        shift -= align_stride
                else:
                    if right_shift:
                        shift += align_stride
                    else:
                        break
                steps += 1

            # Note: this mutating may not be compatible with parallelizing
            # TODO: support for vertical shifts too
            field_block.shifts = [shift, 0]

            # print("Aligned field_block: ",field_block.name,"Corrected Shift:",
            #   field_block.shift,", dimensions:", field_block.dimensions,
            #   "origin:", field_block.origin,'\n')

            # Note: Little debugging visualization - view the particular Qstrip
            # if(
            #     0
            #     # or "q17" in (field_block_bubbles[0].field_label)
            #     # or (field_block_bubbles[0].field_label+str(block_q_strip_no))=="q15"
            #  ):
            #     st, end = qStrip
            #     InteractionUtils.show("QStrip: "+key+"-"+str(block_q_strip_no),
            #     img[st[1] : end[1], st[0]+shift : end[0]+shift],0,config=config)
        # print("End Alignment")

        final_align = None
        if config.outputs.show_image_level >= 2:
            initial_align = image_instance_ops.draw_field_blocks_layout(
                image, template, shifted=False
            )
            final_align = image_instance_ops.draw_field_blocks_layout(
                image,
                template,
                shifted=True,
            )
            # appendSaveImg(4,mean_vals)
            image_instance_ops.append_save_img(2, initial_align)
            image_instance_ops.append_save_img(2, final_align)

            final_align = np.hstack((initial_align, final_align))
        image_instance_ops.append_save_img(5, image)

        if config.outputs.show_image_level >= 3 and final_align is not None:
            display_height, _display_width = config.dimensions.display_image_shape
            final_align = ImageUtils.resize_util_h(final_align, int(display_height))
            # [final_align.shape[1],0])
            InteractionUtils.show(
                "Template Alignment Adjustment", final_align, 0, 0, config=config
            )

        return image, _colored_image, template
