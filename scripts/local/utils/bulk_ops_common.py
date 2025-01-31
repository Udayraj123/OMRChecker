import argparse
import functools
import glob
import operator
import os
from PIL import Image
from pdf2image import convert_from_path
from src.utils.file import PathUtils

# TODO: add shell utilities for bulk image processing, resizing, watermarking, etc.

def walk_and_extract_files(input_dir, file_extensions):
    """
    Walks through the directory to extract files with specified extensions.
    """
    extracted_files = []
    for _dir, _subdir, _files in os.walk(input_dir):
        matching_globs = [
            glob.glob(os.path.join(_dir, f"*.{file_extension}"))
            for file_extension in file_extensions
        ]
        matching_files = functools.reduce(operator.iconcat, matching_globs, [])
        for file_path in matching_files:
            extracted_files.append(PathUtils.sep_based_posix_path(file_path))
    return extracted_files


def get_local_argparser():
    """
    Returns an argument parser with common input, output, and optional recursive processing flags.
    """
    local_argparser = argparse.ArgumentParser()
    
    local_argparser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        dest="input",
        help="Specify the path to your input directory",
    )

    local_argparser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        dest="output",
        help="Specify the path to your output directory",
    )

    local_argparser.add_argument(
        "-r",
        "--recursive",
        action='store_true',
        dest="recursive",
        help="Specify whether to process subdirectories recursively",
    )

    local_argparser.add_argument(
        "--trigger-size",
        default=200,
        type=int,
        dest="trigger_size",
        help="Specify minimum file size (KB) to trigger the hook.",
    )

    return local_argparser


def convert_image(input_path, output_path, output_format):
    """
    Converts an image to the specified output format.
    """
    with Image.open(input_path) as img:
        if output_format == 'JPG':
            output_format = 'JPEG'
        img.save(output_path, output_format)


def convert_pdf_to_jpg(input_path, output_dir):
    """
    Converts a PDF to a series of JPG images, one per page.
    """
    pages = convert_from_path(input_path)
    for i, page in enumerate(pages):
        output_path = os.path.join(output_dir, f"page_{i + 1}.jpg")
        page.save(output_path, 'JPEG')


def bulk_convert(input_dir, output_dir, output_format, in_place=False):
    """
    Bulk converts images and PDFs to the specified format.
    """
    os.makedirs(output_dir, exist_ok=True)
    extensions = ['png', 'jpg', 'jpeg', 'tiff', 'tif', 'pdf']

    filepaths = walk_and_extract_files(input_dir, extensions)

    for input_path in filepaths:
        relative_path = os.path.relpath(os.path.dirname(input_path), input_dir)
        output_subdir = os.path.join(output_dir, relative_path) if not in_place else os.path.dirname(input_path)
        os.makedirs(output_subdir, exist_ok=True)

        filename = os.path.basename(input_path)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
            name, _ = os.path.splitext(filename)
            output_path = os.path.join(output_subdir, f"{name}.{output_format.lower()}")
            convert_image(input_path, output_path, output_format)
            print(f"Converted {filename} to {output_format}")
        elif filename.lower().endswith('.pdf'):
            pdf_output_dir = os.path.join(output_subdir, os.path.splitext(filename)[0])
            os.makedirs(pdf_output_dir, exist_ok=True)
            convert_pdf_to_jpg(input_path, pdf_output_dir)
            print(f"Converted {filename} to JPG")
        else:
            print(f"Skipping unsupported file: {filename}")


def add_common_args(argparser, arguments):
    """
    Adds arguments from the local argparser to the main argument parser.
    """
    local_argparser = get_local_argparser()
    for argument in arguments:
        for action in local_argparser._actions:
            if argument in action.option_strings:
                argparser.add_argument(
                    *action.option_strings,
                    dest=action.dest,
                    type=action.type,
                    default=action.default,
                    required=action.required,
                    help=action.help,
                )
                break  # Move to the next argument


def run_argparser(argparser):
    """
    Runs the argument parser and returns parsed arguments.
    """
    args, unknown = argparser.parse_known_args()
    args = vars(args)

    if unknown:
        argparser.print_help()
        raise Exception(f"\nError: Unknown arguments: {unknown}")

    return args


def main():
    """
    Main entry point for the script. Handles argument parsing and starts the bulk conversion process.
    """
    parser = argparse.ArgumentParser(description="Bulk image and PDF converter")

    # Add standard arguments
    add_common_args(parser, ['-i', '--input', '-o', '--output', '--recursive', '--trigger-size'])

    parser.add_argument(
        "--format", 
        choices=['jpg', 'png', 'jpeg'], 
        default='jpg', 
        help="Output format for images (default: jpg)"
    )
    parser.add_argument(
        "--in-place", 
        action='store_true', 
        help="Modify files in place"
    )

    args = run_argparser(parser)

    bulk_convert(args['input'], args['output'], args['format'].upper(), args['in_place'])


if __name__ == "__main__":
    main()
