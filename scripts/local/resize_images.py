import os
from utils.bulk_ops_common import (
    get_local_argparser,
    run_argparser,
    walk_and_extract_files,
    resize_image,
)


def resize_images(input_dir, output_dir, max_width, max_height, trigger_size):
    """
    Resize images in the input directory and save them in the output directory.

    :param input_dir: Directory containing images to be resized.
    :param output_dir: Directory to save resized images.
    :param max_width: Maximum width for resizing.
    :param max_height: Maximum height for resizing.
    :param trigger_size: Minimum file size in bytes to trigger resizing.
    """
    # Extract image files from the input directory
    file_extensions = ["jpg", "jpeg", "png", "tiff"]
    files = walk_and_extract_files(input_dir, file_extensions)

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Resize each image and save it to the output directory
    for file_path in files:
        output_file_path = os.path.join(output_dir, os.path.basename(file_path))
        resize_image(file_path, output_file_path, max_width, max_height, trigger_size)


if __name__ == "__main__":
    # Set up argument parser and parse arguments
    argparser = get_local_argparser()
    args = run_argparser(argparser)

    # Call the resize_images function with parsed arguments
    resize_images(
        input_dir=args["input"],
        output_dir=args["output"],
        max_width=args["max_width"],
        max_height=args["max_height"],
        trigger_size=args["trigger_size"],
    )
