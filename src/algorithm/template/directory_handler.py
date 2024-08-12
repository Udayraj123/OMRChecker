import json
import os
from csv import QUOTE_NONNUMERIC
from time import localtime, strftime

import pandas as pd

from src.utils.file import PathUtils
from src.utils.logger import logger


class DirectoryHandler:
    def __init__(self):
        self.path_utils = None

    def reset_path_utils(self, output_dir, output_mode):
        # Override the paths utils
        self.path_utils = PathUtils(output_dir)
        self.path_utils.create_output_directories()
        self.setup_outputs_namespace(self, output_mode)

    def setup_outputs_namespace(self, output_mode):
        logger.info("Checking Files...")
        self.omr_response_columns = (
            list(self.all_parsed_labels)
            if output_mode == "moderation"
            else self.output_columns
        )

        # Include current output paths
        self.empty_response_array = [""] * len(self.omr_response_columns)

        self.sheet_columns = [
            "file_id",
            "input_path",
            "output_path",
            "score",
        ] + self.omr_response_columns

        self.output_files = {}
        TIME_NOW_HRS = strftime("%I%p", localtime())
        files_map = {
            "Results": os.path.join(
                self.path_utils.results_dir, f"Results_{TIME_NOW_HRS}.csv"
            ),
            "MultiMarked": os.path.join(
                self.path_utils.manual_dir, "MultiMarkedFiles.csv"
            ),
            "Errors": os.path.join(self.path_utils.manual_dir, "ErrorFiles.csv"),
        }

        for file_key, file_name in files_map.items():
            if not os.path.exists(file_name):
                logger.info(f"Created new file: '{file_name}'")
                # moved handling of files to pandas csv writer
                self.output_files[file_key] = file_name
                # Create Header Columns
                pd.DataFrame([self.sheet_columns], dtype=str).to_csv(
                    self.output_files[file_key],
                    mode="a",
                    quoting=QUOTE_NONNUMERIC,
                    header=False,
                    index=False,
                )
            else:
                logger.info(f"Present : appending to '{file_name}'")
                self.output_files[file_key] = open(file_name, "a")

    # TODO: figure out a structure to output directory metrics apart from from this file one.
    # directory_metrics_path = self.path_utils.image_metrics_dir.joinpath()
    # def export_omr_metrics_for_directory()
    def export_omr_metrics_for_file(
        self,
        file_name,
        # TODO: get these args from self instead
        image,
        final_marked,
        colored_final_marked,
        template,
        field_number_to_field_bubble_means,
        all_fields_threshold_for_file,
        confidence_metrics_for_file,
        evaluation_meta,
    ):
        field_wise_means_and_refs = []
        # TODO: loop over using a list of field_labels now
        for field_bubble_means in field_number_to_field_bubble_means:
            field_wise_means_and_refs.extend(field_bubble_means)
        # sorted_global_bubble_means_and_refs = sorted(field_wise_means_and_refs)

        image_metrics_path = self.path_utils.image_metrics_dir.joinpath(
            f"{os.path.splitext(file_name)[0]}.js"
        )

        template_meta = template.template_detector.get_omr_metrics_for_file(
            # TODO: args here
        )
        # evaluation_meta = evaluation.get_omr_metrics_for_file()

        image_metrics = {
            "template_meta": template_meta,
            "evaluation_meta": (evaluation_meta if evaluation_meta is not None else {}),
        }
        with open(
            image_metrics_path,
            "w",
        ) as f:
            json_string = json.dumps(
                image_metrics,
                default=lambda x: x.to_json(),
                indent=4,
            )
            f.write(f"export default {json_string}")
            logger.info(f"Exported image metrics to: {image_metrics_path}")
