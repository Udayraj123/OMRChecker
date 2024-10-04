#!/usr/bin/env python3
import argparse
import os
from scripts.local.utils.bulk_ops_common import convert_image

def convert_images(filenames, output_format):
    converted_count = 0
    for input_path in filenames:
        if input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
            name, ext = os.path.splitext(input_path)
            output_path = f"{name}.{output_format.lower()}"
            convert_image(input_path, output_path, output_format)
            print(f"Converted {input_path} to {output_format}")
            converted_count += 1
        else:
            print(f"Skipping unsupported file: {input_path}")
    return converted_count

def parse_args():
    parser = argparse.ArgumentParser(description="Convert images for pre-commit hook")
    parser.add_argument(
        "--format", 
        choices=['jpg', 'png', 'jpeg'], 
        default='jpg', 
        help="Output format for images (default: jpg)"
    )
    parser.add_argument("filenames", nargs="*", help="Files to convert.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    converted_count = convert_images(args.filenames, args.format.upper())
    if converted_count > 0:
        print(f"Note: {converted_count} images were converted.")
        exit(1)
    else:
        exit(0)