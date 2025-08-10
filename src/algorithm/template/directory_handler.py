from csv import QUOTE_NONNUMERIC
from pathlib import Path
from time import localtime, strftime
from typing import TYPE_CHECKING

import pandas as pd

from src.utils.constants import OUTPUT_MODES
from src.utils.file import PathUtils
from src.utils.logger import logger

if TYPE_CHECKING:
    from io import TextIOWrapper


# Process files + directory
class TemplateDirectoryHandler:
    def __init__(self, template) -> None:
        self.template = template

    def reset_path_utils(self, output_dir: Path, output_mode) -> None:
        if output_mode == OUTPUT_MODES.SET_LAYOUT:
            logger.info("Note: Skipped files creation in setLayout mode")
            return

        # Override the paths utils
        self.path_utils = PathUtils(output_dir)
        self.path_utils.create_output_directories()
        self.setup_outputs_namespace(output_mode)

    def setup_outputs_namespace(self, output_mode) -> None:
        logger.info("Checking Files...")
        self.omr_response_columns = (
            list(self.template.all_parsed_labels)
            if output_mode == OUTPUT_MODES.MODERATION
            else self.template.output_columns
        )

        # Include current output paths
        self.empty_response_array = [""] * len(self.omr_response_columns)

        self.sheet_columns = [
            "file_id",
            "input_path",
            "output_path",
            "score",
            *self.omr_response_columns,
        ]

        self.output_files: dict[str, TextIOWrapper] = {}
        time_now_hrs = strftime("%I%p", localtime())
        files_map = {
            "Results": self.path_utils.results_dir / f"Results_{time_now_hrs}.csv",
            "MultiMarked": self.path_utils.manual_dir / "MultiMarkedFiles.csv",
            "Errors": self.path_utils.manual_dir / "ErrorFiles.csv",
        }

        for file_key, file_path in files_map.items():
            if not file_path.exists():
                logger.info(f"Created new file: '{file_path}'")
                # Create Header Columns
                pd.DataFrame([self.sheet_columns], dtype=str).to_csv(
                    file_path,
                    mode="a",
                    quoting=QUOTE_NONNUMERIC,
                    header=False,
                    index=False,
                )
            else:
                logger.info(f"Present : appending to '{file_path}'")

            self.output_files[file_key] = Path.open(file_path, "a")

    def finish_processing_directory(self):
        for file_key, file_handler in self.output_files.items():
            file_handler.close()
            logger.debug(f"Closed file {file_key}")

    def get_empty_response_array(self):
        return self.empty_response_array
