from pathlib import Path

from src.algorithm.template.directory_handler import DirectoryHandler
from src.algorithm.template.template_detector import TemplateDetector
from src.algorithm.template.template_layout import TemplateLayout
from src.algorithm.template.template_preprocessing import TemplatePreprocessing
from src.utils.file import SaveImageOps
from src.utils.image import ImageUtils

"""
The main interface for interacting with all template json related operations
"""


class Template:
    def __init__(self, template_path, tuning_config):
        # template_json =
        self.save_image_ops = SaveImageOps(tuning_config)
        self.template_layout = TemplateLayout(template_path, tuning_config)
        self.template_detector = TemplateDetector(self, tuning_config)
        self.directory_handler = DirectoryHandler(self)
        
        # re-export references for external use
        self.all_field_detection_types = self.template_layout.all_field_detection_types
        self.all_fields = self.template_layout.all_fields
        self.path = self.template_layout.path
        self.apply_preprocessors = self.template_layout.apply_preprocessors
        self.export_omr_metrics_for_file = self.template_layout.export_omr_metrics_for_file
        # TODO: move some other functions here

    def get_exclude_files(self):
        excluded_files = []
        if self.template_layout.alignment["reference_image_path"] is not None:
            # Note: reference_image_path is already Path()
            excluded_files.extend(
                self.template_layout.alignment["reference_image_path"]
            )

        for pp in self.get_pre_processors():
            excluded_files.extend(Path(p) for p in pp.exclude_files())

        return excluded_files

    # TODO: move consumers of this function inside
    def get_pre_processors(self):
        # return self.template_preprocessing.pre_processors
        return self.template_layout.pre_processors

    # TODO: reduce the number of these getter
    def get_processing_image_shape(self):
        return self.template_preprocessing.processing_image_shape

    def get_empty_response_array(self):
        return self.directory_handler.empty_response_array

    def append_output_omr_response(self, file_name, output_omr_response):
        omr_response_array = []
        for field in self.directory_handler.omr_response_columns:
            omr_response_array.append(output_omr_response[field])

        self.directory_handler.OUTPUT_SET.append([file_name] + omr_response_array)
        return omr_response_array

    def get_results_file(self):
        return self.directory_handler.output_files["Results"]

    # TODO: replace these utils with more dynamic flagged files
    def get_multimarked_file(self):
        return self.directory_handler.output_files["MultiMarked"]

    def get_errors_file(self):
        return self.directory_handler.output_files["Errors"]

    def finalize_directory_metrics(self):
        return self.template_detector.finalize_directory_metrics()

    def get_save_marked_dir(self):
        return self.directory_handler.path_utils.save_marked_dir

    def get_multi_marked_dir(self):
        return self.directory_handler.path_utils.multi_marked_dir

    def get_errors_dir(self):
        return self.directory_handler.path_utils.errors_dir

    def read_omr_response(self, input_gray_image, colored_image, file_path):
        # Note: resize also creates a copy
        gray_image, colored_image = ImageUtils.resize_to_dimensions(
            self.template_dimensions, input_gray_image, colored_image
        )
        # Resize to template dimensions for saved outputs
        self.save_image_ops.append_save_image(
            f"Resized Image", range(3, 7), gray_image, colored_image
        )

        gray_image, colored_image = ImageUtils.normalize(gray_image, colored_image)

        (
            omr_response,
            file_aggregate_params,
        ) = self.template_detector.read_omr_and_update_metrics(
            file_path, gray_image, colored_image
        )

        # file_aggregate_params would be used for drawing the debug layout
        return omr_response, file_aggregate_params
