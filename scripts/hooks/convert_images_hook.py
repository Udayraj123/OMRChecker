#!/usr/bin/env python3
"""Convert png images within the repository."""

import argparse
import os
import sys

from scripts.utils.image_utils import convert_image, get_size_in_kb, get_size_reduction
from src.utils.logger import logger


def convert_images_in_tree(args):
    filenames = args.get("filenames", None)
    trigger_size = args.get("trigger_size", None)
    converted_count = 0
    for image_path in filenames:
        old_size = get_size_in_kb(image_path)
        if old_size <= trigger_size:
            continue

        # Note: the pre-commit hook takes care of ensuring only image files are passed here.
        new_image_path = convert_image(image_path)
        new_size = get_size_in_kb(new_image_path)
        if new_size <= old_size:
            logger.info(
                f"Converted png to jpg: {image_path}: {new_size:.2f}KB {get_size_reduction(old_size, new_size)}"
            )
            converted_count += 1
        else:
            logger.info(
                f"Skipping conversion for {image_path} as size is more than before ({new_size:.2f} KB > {old_size:.2f} KB)"
            )
            os.remove(new_image_path)

    return converted_count


def parse_args():
    # construct the argument parse and parse the arguments
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--trigger-size",
        default=200,
        required=True,
        type=int,
        dest="trigger_size",
        help="Specify minimum file size to trigger the hook.",
    )

    argparser.add_argument("filenames", nargs="*", help="Files to optimize.")

    (
        args,
        unknown,
    ) = argparser.parse_known_args()

    args = vars(args)

    if len(unknown) > 0:
        argparser.print_help()
        msg = f"\nError: Unknown arguments: {unknown}"
        raise Exception(msg)
    return args


if __name__ == "__main__":
    args = parse_args()

    converted_count = convert_images_in_tree(args)
    trigger_size = args["trigger_size"]
    if converted_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)
