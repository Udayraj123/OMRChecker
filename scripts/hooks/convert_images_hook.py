#!/usr/bin/env python3
"""Convert png images within the repository."""


import argparse
import os

import argparse
import os
from scripts.local.utils.bulk_ops_common import convert_image, convert_pdf_to_jpg

from scripts.utils.image_utils import convert_image, get_size_in_kb, get_size_reduction


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
            print(
                f"Converted png to jpg: {image_path}: {new_size:.2f}KB {get_size_reduction(old_size, new_size)}"
            )
            converted_count += 1
        else:
            print(
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
        raise Exception(f"\nError: Unknown arguments: {unknown}")
    return args


if __name__ == "__main__":
    args = parse_args()

    converted_count = convert_images_in_tree(args)
    trigger_size = args["trigger_size"]
    if converted_count > 0:
        print(
            f"Note: {converted_count} png images above {trigger_size}KB were converted to jpg.\nPlease manually remove the png files and add your commit again."
        )
        exit(1)
    else:
        # print("All sample images are jpgs. Commit accepted.")
        exit(0)