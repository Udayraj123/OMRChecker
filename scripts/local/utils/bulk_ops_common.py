import argparse
import functools
import glob
import operator
import os
from PIL import Image, UnidentifiedImageError
from src.utils.file import PathUtils

from src.utils.file import PathUtils

# TODO: add shell utilities for simple local images processing such as:
# From issue: https://github.com/Udayraj123/OMRChecker/issues/213
# - bulk resize,
#     - clip to max width (or height)
#     -  with a conditional trigger if the file size exceeds a provided value
# - bulk convert :
#     - pdf to jpg
#     - png to jpg or vice versa
#     - tiff
# - bulk rename files
#     - adding folder name to file name
#     - removing non-utf characters from filename (to avoid processing errors)
# - add watermark to all images
# - blur a particular section of the images (e.g. student names and signatures)
# - create a gif from a folder of images
# - Save output of cropped pages to avoid cropping in each run (and merge with manually cropped images)
# - Save output of cropped markers to avoid cropping in each run (and merge with manually cropped images)

# Make sure to be cross-os compatible i.e. use Posix paths wherever possible


# Maybe have a common util file for bulk ops and then create one file for each of the above util.


# Usual pre-processing commands for speedups (useful during re-runs)
# python3 scripts/local/convert_images.py -i inputs/ --replace [--filter-ext png,jpg] --output-ext jpg
# python3 scripts/local/resize_images.py -i inputs/ -o outputs --max-width=1500


def walk_and_extract_files(input_dir, file_extensions):
    extracted_files = []
    for _dir, _subdir, _files in os.walk(input_dir):
        matching_globs = [
            glob(os.path.join(_dir, f"*.{file_extension}"))
            for file_extension in file_extensions
        ]
        matching_files = functools.reduce(operator.iconcat, matching_globs, [])
        for file_path in matching_files:
            extracted_files.append(PathUtils.sep_based_posix_path(file_path))
    return extracted_files


def get_local_argparser():
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
        required=True,
        type=bool,
        dest="recursive",
        help="Specify whether to process subdirectories recursively",
    )

    local_argparser.add_argument(
        "--trigger-size",
        default=200,
        required=True,
        type=int,
        dest="trigger_size",
        help="Specify minimum file size to trigger the hook.",
    )
    return local_argparser


def add_common_args(argparser, arguments):
    local_argparser = get_local_argparser()
    for argument in arguments:
        for action in local_argparser._actions:
            if argument in action.option_strings:
                # Copy the argument from local_argparser to argparser
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
    (
        args,
        unknown,
    ) = argparser.parse_known_args()

    args = vars(args)

    if len(unknown) > 0:
        argparser.print_help()
        raise Exception(f"\nError: Unknown arguments: {unknown}")

    return args


def resize_image(
    input_path, output_path, max_width=None, max_height=None, trigger_size=0
):
    """
    Resize an image based on given width, height, and trigger size.

    :param input_path: Path to the input image.
    :param output_path: Path to save the resized image.
    :param max_width: Maximum width for resizing.
    :param max_height: Maximum height for resizing.
    :param trigger_size: Minimum file size in bytes to trigger resizing.
    """
    try:
        if os.path.getsize(input_path) > trigger_size:
            with Image.open(input_path) as img:
                width, height = img.size

                # Resize based on max width or height while keeping the aspect ratio
                if max_width and width > max_width:
                    new_height = int((max_width / width) * height)
                    img = img.resize((max_width, new_height), Image.ANTIALIAS)

                if max_height and height > max_height:
                    new_width = int((max_height / height) * width)
                    img = img.resize((new_width, max_height), Image.ANTIALIAS)

                # Save the resized image
                img.save(output_path)
                print(f"Resized image saved at: {output_path}")
    except UnidentifiedImageError:
        print(f"Skipping corrupt image: {input_path}")
    except Exception as e:
        print(f"Error resizing image {input_path}: {e}")
