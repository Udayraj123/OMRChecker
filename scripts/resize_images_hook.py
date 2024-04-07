#!/usr/bin/env python3
"""Resize images within the repository."""


import argparse
import os
import shutil

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
    without_extension, extension = os.path.splitext(image_path)
    temp_image_path = f"{without_extension}-tmp{extension}"
    with Image.open(image_path) as image:
        old_image_size = image.size[:2]
        w, h = old_image_size
        resized = False

        if h > max_height:
            image = resize_util(image, u_height=max_height)
            resized = True

        if w > max_width:
            image = resize_util(image, u_width=max_width)
            w, h = image.size[:2]
            resized = True

        if resized:
            image.save(temp_image_path)
            return True, temp_image_path, old_image_size, image.size

        return False, temp_image_path, old_image_size, image.size


def resize_images_in_tree(args):
    max_width = args.get("max_width", None)
    max_height = args.get("max_height", None)
    trigger_size = args.get("trigger_size", None)
    filenames = args.get("filenames", None)
    resized_count = 0
    for image_path in filenames:
        old_size = get_size_in_kb(image_path)
        if old_size <= trigger_size:
            continue
        (
            resized_and_saved,
            temp_image_path,
            old_image_size,
            new_image_size,
        ) = resize_image_and_save(image_path, max_width, max_height)
        if resized_and_saved:
            new_size = get_size_in_kb(temp_image_path)
            if new_size >= old_size:
                print(
                    f"Skipping resize for {image_path} as size is more than before ({new_size:.2f} KB > {old_size:.2f} KB)"
                )
                os.remove(temp_image_path)
            else:
                shutil.move(temp_image_path, image_path)
                print(
                    f"Resized: {image_path} {old_image_size} -> {new_image_size} with file size {new_size:.2f}KB {get_size_reduction(old_size, new_size)}"
                )
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
        raise Exception(f"\nError: Unknown arguments: {unknown}")

    return args


if __name__ == "__main__":
    args = parse_args()
    resized_count = resize_images_in_tree(args)
    if resized_count > 0:
        print(
            f"Note: {resized_count} images were resized. Please check, add and commit again."
        )
        exit(1)
    else:
        # print("All images are of the appropriate size. Commit accepted.")
        exit(0)
