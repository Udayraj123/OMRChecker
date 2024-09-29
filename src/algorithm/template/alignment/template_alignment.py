from src.algorithm.template.alignment.k_nearest_interpolation import (
    apply_k_nearest_interpolation_inplace,
)
from src.utils.image import ImageUtils
from src.utils.logger import logger


# TODO: move into template class
def apply_template_alignment(gray_image, colored_image, template, config):
    if "gray_alignment_image" not in template.alignment:
        logger.info(f"Note: Alignment not enabled for template {template}")
        return gray_image, colored_image, template

    # Parsed
    template_margins, template_max_displacement = (
        template.alignment["margins"],
        template.alignment["maxDisplacement"],
    )

    # Pre-processed
    gray_alignment_image, colored_alignment_image = (
        template.alignment["gray_alignment_image"],
        template.alignment["colored_alignment_image"],
    )

    # Note: resize also creates a copy
    (
        gray_image,
        colored_image,
        gray_alignment_image,
        colored_alignment_image,
    ) = ImageUtils.resize_to_dimensions(
        template.template_dimensions,
        gray_image,
        colored_image,
        gray_alignment_image,
        colored_alignment_image,
    )

    for field_block in template.field_blocks:
        field_block_name, origin, dimensions, field_block_alignment = map(
            lambda attr: getattr(field_block, attr),
            ["name", "origin", "dimensions", "alignment"],
        )
        # TODO: wrap this loop body into a function and generalize into passing *any* scanZone in this.

        margins = field_block_alignment.get("margins", template_margins)
        max_displacement = field_block_alignment.get(
            "maxDisplacement", template_max_displacement
        )

        if max_displacement == 0:
            # Skip alignment computation if allowed displacement is zero
            continue

        # compute zone and clip to image dimensions
        zone_start = [
            int(origin[0] - margins["left"]),
            int(origin[1] - margins["top"]),
        ]
        zone_end = [
            int(origin[0] + margins["right"] + dimensions[0]),
            int(origin[1] + margins["bottom"] + dimensions[1]),
        ]

        block_gray_image, block_colored_image, block_gray_alignment_image = map(
            lambda image: (
                None
                if image is None
                else image[zone_start[1] : zone_end[1], zone_start[0] : zone_end[0]]
            ),
            [gray_image, colored_image, gray_alignment_image],
        )

        # Method 1

        # TODO: move to a processor:
        # warped_block_image = apply_phase_correlation_shifts(
        #     field_block, block_gray_alignment_image, block_gray_image
        # )

        # Method 2

        # warped_block_image, warped_colored_image = apply_sift_shifts(
        #     field_block_name,
        #     block_gray_image,
        #     block_colored_image,
        #     block_gray_alignment_image,
        #     # block_colored_alignment_image,
        #     max_displacement,
        #     margins,
        #     dimensions,
        # )

        # # TODO: assignment outside loop?
        # # Set warped field block back into original image
        # gray_image[
        #     zone_start[1] : zone_end[1], zone_start[0] : zone_end[0]
        # ] = warped_block_image
        # if colored_image:
        #     colored_image[
        #         zone_start[1] : zone_end[1], zone_start[0] : zone_end[0]
        #     ] = warped_colored_image

        # Method 3
        # Warp bubble coordinates of the template
        apply_k_nearest_interpolation_inplace(
            field_block,
            block_gray_image,
            block_gray_alignment_image,
            max_displacement,
            margins,
            config,
        )

        # Method 4
        # Warp each field in the image
        # TODO: figure out how to apply detection on these copies to support overlapping field blocks!

    return gray_image, colored_image, template
