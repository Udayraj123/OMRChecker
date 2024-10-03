#!/usr/bin/env python3
"""Convert png and other images within the repository."""

import argparse
import os
import sys
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.image_utils import get_size_in_kb, get_size_reduction

def convert_image(input_path, output_path, output_format):
    with Image.open(input_path) as img:
        if output_format == 'JPG':
            output_format = 'JPEG'
        img.save(output_path, output_format)

def bulk_convert(input_dir, output_dir, output_format, trigger_size):
    os.makedirs(output_dir, exist_ok=True)
    converted_count = 0

    for root, _, files in os.walk(input_dir):
        for filename in files:
            input_path = os.path.join(root, filename)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
                old_size = get_size_in_kb(input_path)

                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}.{output_format.lower()}")

                convert_image(input_path, output_path, output_format.upper())
                new_size = get_size_in_kb(output_path)

                if old_size > trigger_size and new_size <= old_size:
                    print(
                        f"Converted {filename} to {output_format.upper()}: {new_size:.2f}KB "
                        f"{get_size_reduction(old_size, new_size)}"
                    )
                    converted_count += 1
                elif old_size <= trigger_size:
                    print(
                        f"Converted {filename} to {output_format.upper()}: {new_size:.2f}KB"
                    )
                else:
                    print(
                        f"Skipping conversion for {filename} as size increased "
                        f"({new_size:.2f} KB > {old_size:.2f} KB)"
                    )
                    # os.remove(output_path)

    return converted_count

def parse_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--trigger-size",
        default=200,
        required=True,
        type=int,
        dest="trigger_size",
        help="Specify minimum file size to trigger the hook.",
    )
    argparser.add_argument(
        "--input-dir",
        default=None,
        required=False,
        help="Specify the input directory for bulk conversion.",
    )
    argparser.add_argument(
        "--output-dir",
        default=None,
        required=True,
        help="Specify the output directory for converted files.",
    )
    argparser.add_argument(
        "--format",
        choices=['jpg', 'jpeg', 'png'],
        default='jpeg',
        help="Specify the output format (default: jpeg).",
    )
    argparser.add_argument("filenames", nargs="*", help="Files to optimize.")

    args, unknown = argparser.parse_known_args()

    if len(unknown) > 0:
        argparser.print_help()
        raise Exception(f"\nError: Unknown arguments: {unknown}")
    return vars(args)

if __name__ == "__main__":
    args = parse_args()

    trigger_size = args["trigger_size"]
    output_dir = args["output_dir"]
    output_format = args["format"]

    if args.get("input_dir"):
        converted_count = bulk_convert(args["input_dir"], output_dir, output_format, trigger_size)
    else:
        print("No input directory specified. Please provide an input directory.")
        exit(1)

    if converted_count > 0:
        print(
            f"Note: {converted_count} images above {trigger_size}KB were converted to {output_format.upper()}.\n"
        )
        exit(1)
    else:
        print("All images are optimized. Commit accepted.")
        exit(0)