import json
import os
import string
from collections import defaultdict
from pathlib import PureWindowsPath

from src.utils.image import ImageUtils
from src.utils.logger import logger
from src.utils.math import MathUtils


def load_json(path, **rest):
    try:
        # TODO: see if non-utf characters need to be handled
        with open(path, "r") as f:
            loaded = json.load(f, **rest)
    except json.decoder.JSONDecodeError as error:
        logger.critical(error)
        raise Exception(f"Error when loading json file at: '{path}'")

    return loaded


class PathUtils:
    printable_chars = set(string.printable)

    @staticmethod
    def remove_non_utf_characters(path_string):
        return "".join(x for x in path_string if x in PathUtils.printable_chars)

    # @staticmethod
    # def filter_omr_files(omr_files):
    #     filtered_omr_files = []
    #     omr_files_set = set()
    #     for omr_file in omr_files:
    #         omr_file_string = omr_file.as_posix()
    #         filtered_omr_file = PathUtils.remove_non_utf_characters(omr_file_string)
    #         if omr_file_string in omr_files_set:
    #             logger.warning(
    #                 f"Skipping duplicate file after utf-8 encoding: {omr_file_string}"
    #             )
    #         omr_files_set.add(omr_file_string)
    #         filtered_omr_files.append(Path(filtered_omr_file))
    #     return filtered_omr_files

    @staticmethod
    def sep_based_posix_path(path):
        path = os.path.normpath(path)
        # TODO: check for this second condition
        if os.path.sep == "\\" or "\\" in path:
            path = PureWindowsPath(path).as_posix()

        path = PathUtils.remove_non_utf_characters(path)

        return path

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.save_marked_dir = output_dir.joinpath("CheckedOMRs")
        self.image_metrics_dir = output_dir.joinpath("ImageMetrics")
        self.results_dir = output_dir.joinpath("Results")
        self.manual_dir = output_dir.joinpath("Manual")
        self.errors_dir = self.manual_dir.joinpath("ErrorFiles")
        self.multi_marked_dir = self.manual_dir.joinpath("MultiMarkedFiles")
        self.evaluations_dir = output_dir.joinpath("Evaluations")
        self.debug_dir = output_dir.joinpath("Debug")

    def create_output_directories(self):
        logger.info("Checking Directories...")

        if not os.path.exists(self.save_marked_dir):
            os.makedirs(self.save_marked_dir)
        # Main output directories
        for save_output_dir in [
            self.save_marked_dir.joinpath("colored"),
            self.save_marked_dir.joinpath("stack"),
            self.save_marked_dir.joinpath("stack", "colored"),
            self.save_marked_dir.joinpath("_MULTI_"),
            self.save_marked_dir.joinpath("_MULTI_", "colored"),
        ]:
            if not os.path.exists(save_output_dir):
                os.mkdir(save_output_dir)
                logger.info(f"Created : {save_output_dir}")

        # Image buckets
        for save_output_dir in [
            self.manual_dir,
            self.multi_marked_dir,
            self.errors_dir,
        ]:
            if not os.path.exists(save_output_dir):
                logger.info(f"Created : {save_output_dir}")
                os.makedirs(save_output_dir)
                os.mkdir(save_output_dir.joinpath("colored"))

        # Non-image directories
        for save_output_dir in [
            self.results_dir,
            self.image_metrics_dir,
            self.evaluations_dir,
        ]:
            if not os.path.exists(save_output_dir):
                logger.info(f"Created : {save_output_dir}")
                os.makedirs(save_output_dir)


class SaveImageOps:
    def __init__(self, tuning_config):
        self.gray_images = defaultdict(list)
        self.colored_images = defaultdict(list)
        self.tuning_config = tuning_config
        self.save_image_level = tuning_config.outputs.save_image_level

    def append_save_image(self, title, keys, gray_image=None, colored_image=None):
        assert type(title) == str
        if type(keys) == int:
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

    def save_image_stacks(self, filename, save_marked_dir, key=None, images_per_row=4):
        key = self.save_image_level if key is None else key
        if self.save_image_level >= int(key):
            name, _ext = os.path.splitext(filename)
            logger.info(
                f"Stack level: {key}", [title for title, _ in self.gray_images[key]]
            )

            if len(self.gray_images[key]) > 0:
                result = self.get_result_hstack(self.gray_images[key], images_per_row)
                stack_path = f"{save_marked_dir}/stack/{name}_{str(key)}_stack.jpg"
                logger.info(f"Saved stack image to: {stack_path}")
                ImageUtils.save_img(stack_path, result)
            else:
                logger.info(
                    f"Note: Nothing to save for gray image. Stack level: {self.save_image_level}"
                )

            if len(self.colored_images[key]) > 0:
                colored_result = self.get_result_hstack(
                    self.colored_images[key], images_per_row
                )
                colored_stack_path = (
                    f"{save_marked_dir}/stack/colored/{name}_{str(key)}_stack.jpg"
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
        result = ImageUtils.resize_single(
            result,
            min(
                len(titles_and_images) * display_width // 3,
                int(display_width * 2.5),
            ),
        )
        return result

    def reset_all_save_img(self):
        # Max save image level is 6
        for i in range(7):
            self.gray_images[i + 1] = []
            self.colored_images[i + 1] = []
