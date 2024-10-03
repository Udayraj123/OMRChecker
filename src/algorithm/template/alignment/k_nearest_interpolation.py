import numpy as np

from src.algorithm.template.alignment.sift_matcher import SiftMatcher
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
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
        [anchor_point, MathUtils.subtract_points(anchor_point, displaced_point)]
        for anchor_point, displaced_point in displacement_pairs
    ]
    block_image_shifts = MathUtils.subtract_points(
        [margins["left"], margins["top"]], field_block.origin
    )
    if config.outputs.show_image_level >= 2:
        block_gray_image_before = block_gray_image.copy()

        # Shift the coordinates to the field block's origin to draw on the cropped block
        old_shifts = field_block.shifts
        field_block.shifts = block_image_shifts
        TemplateDrawing.draw_field_block(
            field_block, block_gray_image_before, shifted=True, thickness=2
        )
        field_block.shifts = old_shifts

    # modify bubble level shifts
    average_shifts = shift_by_field_blocks(
        field_block, block_image_shifts, anchors_with_displacements, k
    )
    # shift_by_fields(field_block, block_image_shifts, anchors_with_displacements, k)
    # shift_by_scan_boxes(field_block, block_image_shifts, anchors_with_displacements, k)

    if config.outputs.show_image_level >= 2:
        block_gray_image_after = block_gray_image.copy()
        old_shifts = field_block.shifts
        field_block.shifts = block_image_shifts  # MathUtils.add_points(block_image_shifts, average_shifts)
        TemplateDrawing.draw_field_block(
            field_block, block_gray_image_after, shifted=True, thickness=2
        )
        field_block.shifts = old_shifts

        InteractionUtils.show(
            f"Field Block shifts: {average_shifts}",
            ImageUtils.get_padded_hstack(
                [block_gray_image_before, block_gray_image_after]
            ),
        )

    # Method 2: Get affine transform on the bubble coordinates
    # field_block.shifts = average_shifts


def shift_by_field_blocks(
    field_block, block_image_shifts, anchors_with_displacements, k, centered=False
):
    # Take average position of all bubbles
    field_block_position = (
        np.average(
            [
                # field center
                np.average(
                    [
                        scan_box.get_shifted_position(block_image_shifts)
                        for scan_box in field.scan_boxes
                    ],
                    axis=0,
                )
                for field in field_block.fields
            ],
            axis=0,
        ).astype(np.int32)
        if centered
        else MathUtils.add_points(block_image_shifts, field_block.origin)
    )

    nearest_anchors = find_k_nearest_anchors(
        field_block_position, anchors_with_displacements, k
    )

    # Method 1: Get average displacement
    average_shifts = np.average(
        [displacement for _anchor_point, displacement in nearest_anchors],
        axis=0,
    ).astype(np.int32)

    # Shift all bubbles
    for field in field_block.fields:
        for scan_box in field.scan_boxes:
            scan_box.shifts = average_shifts

    return average_shifts


def shift_by_fields(field_block, block_image_shifts, anchors_with_displacements, k):
    # modify bubble level shifts
    for field in field_block.fields:
        # Take average position of all bubbles
        field_center_position = np.average(
            [
                scan_box.get_shifted_position(block_image_shifts)
                for scan_box in field.scan_boxes
            ],
            axis=0,
        ).astype(np.int32)

        nearest_anchors = find_k_nearest_anchors(
            field_center_position, anchors_with_displacements, k
        )

        # Method 1: Get average displacement
        average_shifts = np.average(
            [displacement for _anchor_point, displacement in nearest_anchors],
            axis=0,
        ).astype(np.int32)

        # Shift all bubbles
        for scan_box in field.scan_boxes:
            scan_box.shifts = average_shifts


def shift_by_scan_boxes(field_block, block_image_shifts, anchors_with_displacements, k):
    for field in field_block.fields:
        for scan_box in field.scan_boxes:
            scan_box.reset_shifts()
            relative_bubble_positions = scan_box.get_shifted_position(
                block_image_shifts
            )
            nearest_anchors = find_k_nearest_anchors(
                relative_bubble_positions, anchors_with_displacements, k
            )
            # Method 1: Get average displacement
            average_shifts = np.average(
                [displacement for _anchor_point, displacement in nearest_anchors],
                axis=0,
            ).astype(np.int32)

            scan_box.shifts = average_shifts
