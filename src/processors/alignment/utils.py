from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils


def show_displacement_overlay(
    block_gray_alignment_image, block_gray_image, shifted_block_image
) -> None:
    # ..
    overlay = ImageUtils.overlay_image(block_gray_alignment_image, block_gray_image)
    overlay_shifted = ImageUtils.overlay_image(
        block_gray_alignment_image, shifted_block_image
    )

    InteractionUtils.show(
        "Alignment For Field Block",
        ImageUtils.get_padded_hstack(
            [
                block_gray_alignment_image,
                block_gray_image,
                shifted_block_image,
                overlay,
                overlay_shifted,
            ]
        ),
    )
