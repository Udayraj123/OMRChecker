#!/usr/bin/env python3
"""Resize images within the repository."""


import argparse
import os

from PIL import Image


def get_size_in_kb(path):
    return os.path.getsize(path) / 1000


def get_size_reduction(old_size, new_size):
    percent = 100 * (new_size - old_size) / old_size
    return f"({percent:.2f}%)"


def resize_util(image, u_width=None, u_height=None):
    w, h = image.size[:2]
    if u_height is None:
        u_height = int(h * u_width / w)
    if u_width is None:
        u_width = int(w * u_height / h)

    if u_height == h and u_width == w:
        # No need to resize
        return image
    return image.resize((int(u_width), int(u_height)), Image.LANCZOS)


def resize_image_and_save(image_path, max_width, max_height):
    with Image.open(image_path) as image:
        w, h = image.size[:2]

        if w > max_width:
            image = resize_util(image, u_width=max_width)
        if h > max_height:
            image = resize_util(image, u_height=max_height)
        if w > max_width or h > max_height:
            image.save(image_path)
            return True, image.size
        return False, image.size


def resize_images_in_tree(args):
    max_width = args.get("max_width", None)
    max_height = args.get("max_height", None)
    filenames = args.get("filenames", None)
    resized_any = False
    for path in filenames:
        old_size = get_size_in_kb(path)
        resized_and_saved, new_image_size = resize_image_and_save(
            path, max_width, max_height
        )
        if resized_and_saved:
            new_size = get_size_in_kb(path)
            print(
                f"Resized: {path} to {new_image_size} with size {new_size:.2f}KB {get_size_reduction(old_size, new_size)}"
            )
            resized_any = True
    return resized_any


def parse_args():
    # construct the argument parse and parse the arguments
    argparser = argparse.ArgumentParser()

    argparser.add_argument("filenames", nargs="*", help="Files to optimize.")

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
        raise Exception(f"\nError: Unknown arguments: {unknown}")

    return args


if __name__ == "__main__":
    args = parse_args()

    if resize_images_in_tree(args):
        print("Note: Some images were resized. Please check, add and commit again.")
        exit(1)
    else:
        print("All images are of the appropriate size. Commit accepted.")
        exit(0)
