#!/usr/bin/env python3
"""Convert png images within the repository."""


import argparse
import os

from PIL import Image


def get_size_in_kb(path):
    return os.path.getsize(path) / 1000


def get_size_reduction(old_size, new_size):
    percent = 100 * (new_size - old_size) / old_size
    return f"({percent:.2f}%)"


def convert_image(image_path):
    with Image.open(image_path) as image:
        # Note: using hardcoded -4 as we're expected to receive .png or .PNG files only
        new_image_path = f"{image_path[:-4]}.jpg"
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image.save(new_image_path, "JPEG", quality=90, optimize=True)

        return new_image_path


def convert_images_in_tree(args):
    filenames = args.get("filenames", None)
    trigger_size = args.get("trigger_size", None)
    converted_count = 0
    for image_path in filenames:
        old_size = get_size_in_kb(image_path)
        if old_size <= trigger_size:
            continue

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
