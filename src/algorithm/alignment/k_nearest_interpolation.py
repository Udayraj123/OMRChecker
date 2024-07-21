import numpy as np

from src.algorithm.alignment.sift_matcher import SiftMatcher
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger
from src.utils.math import MathUtils
from src.utils.template_drawing import TemplateDrawing


def find_k_nearest_anchors(origin, anchors_with_displacements, k):
    # TODO: any further optimization needed?
    sorted_by_distance = sorted(
        [
            [MathUtils.distance(origin, anchor_point), [anchor_point, displacement]]
            for anchor_point, displacement in anchors_with_displacements
        ],
        key=lambda item: item[0],
    )

    return [
        anchor_with_displacement
        for _distance, anchor_with_displacement in sorted_by_distance[:k]
    ]


def apply_k_nearest_interpolation_inplace(
    field_block,
    block_gray_image,
    block_gray_alignment_image,
    max_displacement,
    margins,
    config,
    k=4,
):
    field_block_name = field_block.name
    displacement_pairs = SiftMatcher.get_matches(
        field_block_name,
        block_gray_image,
        block_gray_alignment_image,
        max_displacement,
        config,
    )
    anchors_with_displacements = [
        [anchor_point, MathUtils.get_relative_position(displaced_point, anchor_point)]
        for anchor_point, displaced_point in displacement_pairs
    ]
    # modify field block level shifts

    nearest_anchors = find_k_nearest_anchors(
        field_block.origin, anchors_with_displacements, k
    )
    # Method 1: Get average displacement
    average_shifts = np.average(
        [displacement for _anchor_point, displacement in nearest_anchors], axis=0
    ).astype(np.int32)
    logger.info(field_block.name, average_shifts)

    if config.outputs.show_image_level >= 2:
        block_image_origin = MathUtils.get_relative_position(
            [margins["left"], margins["top"]], field_block.origin
        )
        block_gray_image_before = block_gray_image.copy()
        # Shift the coordinates to the field block's origin to draw on the cropped block
        field_block.shifts = MathUtils.get_relative_position(block_image_origin, [0, 0])

        TemplateDrawing.draw_field_block(
            field_block, block_gray_image_before, shifted=True, thickness=2
        )

        block_gray_image_after = block_gray_image.copy()
        field_block.shifts = MathUtils.get_relative_position(
            block_image_origin, average_shifts
        )

        TemplateDrawing.draw_field_block(
            field_block, block_gray_image_after, shifted=True, thickness=2
        )
        InteractionUtils.show(
            f"Field Block shifts: {average_shifts}",
            ImageUtils.get_padded_hstack(
                [block_gray_image_before, block_gray_image_after]
            ),
        )

    field_block.shifts = average_shifts
    # Method 2: Get affine transform
