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
        new_image_path = f"{image_path[:-4]}.jpg"
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image.save(new_image_path, "JPEG", quality=90, optimize=True)

        return new_image_path


def convert_images_in_tree(args):
    filenames = args.get("filenames", None)
    converted_any = False
    for path in filenames:
        old_size = get_size_in_kb(path)
        new_image_path = convert_image(path)
        new_size = get_size_in_kb(new_image_path)
        print(
            f"Converted png to jpg: {path} : {new_size:.2f}KB {get_size_reduction(old_size, new_size)}"
        )
        converted_any = True
    return converted_any


def parse_args():
    # construct the argument parse and parse the arguments
    argparser = argparse.ArgumentParser()

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

    if convert_images_in_tree(args):
        print(
            "Note: Some png images were converted to jpg. Please manually remove the png files and add your commit again."
        )
        exit(1)
    else:
        print("All sample images are jpgs. Commit accepted.")
        exit(0)
