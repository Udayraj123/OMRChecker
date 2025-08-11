import hashlib
import json
import os
import string
from collections import defaultdict
from pathlib import Path, PureWindowsPath
from typing import Any, ClassVar

from src.utils.image import ImageUtils
from src.utils.logger import logger
from src.utils.math import MathUtils


def load_json(path, **rest) -> dict[str, Any]:
    try:
        # TODO: see if non-utf characters need to be handled
        with Path.open(path) as f:
            loaded = json.load(f, **rest)
    except json.decoder.JSONDecodeError as error:
        msg = f"Error when loading json file at: '{path}'"
        logger.critical(msg, error)
        raise Exception(msg) from None

    return loaded


def calculate_file_checksum(file_path: Path | str, algorithm: str = "sha256") -> str:
    """
    Calculate the checksum of a file using the specified hashing algorithm.

    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use (md5, sha1, sha256, sha512)

    Returns:
        Hexadecimal string representation of the file's checksum

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the algorithm is not supported
    """
    file_path = Path(file_path)

    if not file_path.exists():
        msg = f"File not found: {file_path}"
        raise FileNotFoundError(msg)

    # Validate algorithm
    try:
        hasher = hashlib.new(algorithm)
    except ValueError as e:
        msg = f"Unsupported hash algorithm: {algorithm}"
        raise ValueError(msg) from e

    # Read file in chunks to handle large files efficiently
    with file_path.open("rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()


def print_file_checksum(file_path: Path | str, algorithm: str = "md5") -> None:
    """
    Calculate and print the checksum of a file.

    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use (md5, sha1, sha256, sha512)
    """
    try:
        checksum = calculate_file_checksum(file_path, algorithm)
        print(f"{algorithm.upper()} ({file_path}): {checksum}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")


class PathUtils:
    printable_chars: ClassVar = set(string.printable)

    @staticmethod
    def remove_non_utf_characters(path_string):
        return "".join(x for x in path_string if x in PathUtils.printable_chars)

    # TODO: add util to skip duplicate files after removing utf-8 encoding

    @staticmethod
    def sep_based_posix_path(path):
        path = os.path.normpath(path)
        # TODO: check for this second condition
        if os.path.sep == "\\" or "\\" in path:
            path = PureWindowsPath(path).as_posix()

        return PathUtils.remove_non_utf_characters(path)

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.save_marked_dir = output_dir.joinpath("CheckedOMRs")
        self.image_metrics_dir = output_dir.joinpath("ImageMetrics")
        self.results_dir = output_dir.joinpath("Results")
        self.manual_dir = output_dir.joinpath("Manual")
        self.errors_dir = self.manual_dir.joinpath("ErrorFiles")
        self.multi_marked_dir = self.manual_dir.joinpath("MultiMarkedFiles")
        self.evaluations_dir = output_dir.joinpath("Evaluations")
        self.debug_dir = output_dir.joinpath("Debug")

    def create_output_directories(self) -> None:
        logger.info("Checking Directories...")

        if not self.save_marked_dir.exists():
            Path.mkdir(self.save_marked_dir, parents=True)
        # Main output directories
        for save_output_dir in [
            self.save_marked_dir.joinpath("colored"),
            self.save_marked_dir.joinpath("stack"),
            self.save_marked_dir.joinpath("stack", "colored"),
            self.save_marked_dir.joinpath("_MULTI_"),
            self.save_marked_dir.joinpath("_MULTI_", "colored"),
        ]:
            if not save_output_dir.exists():
                Path.mkdir(save_output_dir)
                logger.info(f"Created : {save_output_dir}")

        # Image buckets
        for save_output_dir in [
            self.manual_dir,
            self.multi_marked_dir,
            self.errors_dir,
        ]:
            if not save_output_dir.exists():
                logger.info(f"Created : {save_output_dir}")
                Path.mkdir(save_output_dir, parents=True)
                Path.mkdir(save_output_dir.joinpath("colored"))

        # Non-image directories
        for save_output_dir in [
            self.results_dir,
            self.image_metrics_dir,
            self.evaluations_dir,
        ]:
            if not save_output_dir.exists():
                logger.info(f"Created : {save_output_dir}")
                Path.mkdir(save_output_dir, parents=True)


class SaveImageOps:
    def __init__(self, tuning_config) -> None:
        self.gray_images = defaultdict(list)
        self.colored_images = defaultdict(list)
        self.tuning_config = tuning_config
        self.save_image_level = tuning_config.outputs.save_image_level

    def append_save_image(
        self, title, keys, gray_image=None, colored_image=None
    ) -> None:
        if not isinstance(title, str):
            msg = f"title={title} is not a string"
            raise TypeError(msg)
        if isinstance(keys, int):
            keys = [keys]
        gray_image_copy, colored_image_copy = None, None
        if gray_image is not None:
            gray_image_copy = gray_image.copy()
        if colored_image is not None:
            colored_image_copy = colored_image.copy()
        for key in keys:
            if int(key) > self.save_image_level:
                continue
            if gray_image_copy is not None:
                self.gray_images[key].append([title, gray_image_copy])
            if colored_image_copy is not None:
                self.colored_images[key].append([title, colored_image_copy])

    def save_image_stacks(
        self, file_path: Path, save_marked_dir: Path, key=None, images_per_row=4
    ) -> None:
        key = self.save_image_level if key is None else key
        if self.save_image_level >= int(key):
            stem = file_path.stem

            if len(self.gray_images[key]) > 0:
                logger.info(
                    f"Gray Stack level: {key}",
                    [title for title, _ in self.gray_images[key]],
                )
                result = self.get_result_hstack(self.gray_images[key], images_per_row)
                stack_path = f"{save_marked_dir}/stack/{stem}_{key!s}_stack.jpg"
                logger.info(f"Saved stack image to: {stack_path}")
                ImageUtils.save_img(stack_path, result)
            else:
                logger.info(
                    f"Note: Nothing to save for gray image. Stack level: {self.save_image_level}"
                )

            if len(self.colored_images[key]) > 0:
                logger.info(
                    f"Colored Stack level: {key}",
                    [title for title, _ in self.colored_images[key]],
                )
                colored_result = self.get_result_hstack(
                    self.colored_images[key], images_per_row
                )
                colored_stack_path = (
                    f"{save_marked_dir}/stack/colored/{stem}_{key!s}_stack.jpg"
                )
                logger.info(f"Saved colored stack image to: {colored_stack_path}")

                ImageUtils.save_img(
                    colored_stack_path,
                    colored_result,
                )
            else:
                logger.info(
                    f"Note: Nothing to save for colored image. Stack level: {self.save_image_level}"
                )

    def get_result_hstack(self, titles_and_images, images_per_row):
        config = self.tuning_config
        _display_height, display_width = config.outputs.display_image_dimensions

        # TODO: attach title text as header to each stack image!
        images = ImageUtils.resize_multiple(
            [image for _title, image in titles_and_images], display_width
        )
        grid_images = MathUtils.chunks(images, images_per_row)
        result = ImageUtils.get_vstack_image_grid(grid_images)
        return ImageUtils.resize_single(
            result,
            min(
                len(titles_and_images) * display_width // 3,
                int(display_width * 2.5),
            ),
        )

    def reset_all_save_img(self) -> None:
        # Max save image level is 6
        for i in range(7):
            self.gray_images[i + 1] = []
            self.colored_images[i + 1] = []
