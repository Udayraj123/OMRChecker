#!/usr/bin/env python3
"""Resize images within the repository."""

import argparse
import os
import shutil
import sys
from pathlib import Path

from scripts.utils.image_utils import get_size_in_kb, resize_image_and_save
from src.utils.logger import logger


def resize_images_in_tree(args):
    max_width = args.get("max_width", None)
    max_height = args.get("max_height", None)
    trigger_size = args.get("trigger_size", None)
    filenames = args.get("filenames", None)
    resized_count = 0
    for file_name in filenames:
        image_path = Path(file_name)
        if not image_path.is_file():
            logger.warning(f"Skipping non-file: {image_path}")
            continue
        old_size = get_size_in_kb(image_path)
        if old_size <= trigger_size:
            continue
        (
            resized_and_saved,
            temp_image_path,
            _old_image_size,
            _new_image_size,
        ) = resize_image_and_save(image_path, max_width, max_height)
        if resized_and_saved:
            new_size = get_size_in_kb(temp_image_path)
            if new_size >= old_size:
                os.remove(temp_image_path)
            else:
                shutil.move(temp_image_path, image_path)
                resized_count += 1
    return resized_count


def parse_args():
    # construct the argument parse and parse the arguments
    argparser = argparse.ArgumentParser()

    argparser.add_argument("filenames", nargs="*", help="Files to optimize.")

    argparser.add_argument(
        "--trigger-size",
        default=200,
        required=True,
        type=int,
        dest="trigger_size",
        help="Specify minimum file size to trigger the hook.",
    )

    argparser.add_argument(
        "--max-width",
        default=1000,
        required=True,
        type=int,
        dest="max_width",
        help="Specify maximum width of the resized images.",
    )
    argparser.add_argument(
        "--max-height",
        default=1000,
        required=False,
        type=int,
        dest="max_height",
        help="Specify maximum height of the resized images.",
    )

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
    resized_count = resize_images_in_tree(args)
    if resized_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)
