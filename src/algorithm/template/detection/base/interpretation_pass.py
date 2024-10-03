from abc import abstractmethod

from src.algorithm.template.detection.base.common_pass import FilePassAggregates
from src.algorithm.template.detection.base.interpretation import FieldInterpretation
from src.algorithm.template.template_layout import Field
from src.utils.stats import StatsByLabel


class FieldTypeInterpretationPass(FilePassAggregates):
    def __init__(self, tuning_config, field_detection_type):
        self.field_detection_type = field_detection_type
        super().__init__(tuning_config)

    @abstractmethod
    def get_field_interpretation(
        self,
        field: Field,
        file_level_detection_aggregates,
        file_level_aggregates,
    ) -> FieldInterpretation:
        raise Exception("Not implemented")

    def initialize_field_level_aggregates(self, field):
        super().initialize_field_level_aggregates(field)
        self.insert_field_level_aggregates(
            {
                "field_level_confidence_metrics": {},
            }
        )

    def initialize_file_level_aggregates(
        self,
        file_path,
        field_detection_type_wise_detection_aggregates,
        field_label_wise_detection_aggregates,
    ):
        super().initialize_file_level_aggregates(file_path)
        self.insert_file_level_aggregates(
            {
                "confidence_metrics_for_file": {},
            }
        )

    def update_aggregates_on_processed_field_interpretation(
        self, field: Field, field_interpretation: FieldInterpretation
    ):
        self.update_field_level_aggregates_on_processed_field_interpretation(
            field, field_interpretation
        )
        field_level_aggregates = self.get_field_level_aggregates()
        self.update_file_level_aggregates_on_processed_field_interpretation(
            field, field_interpretation, field_level_aggregates
        )
        self.update_directory_level_aggregates_on_processed_field_interpretation(
            field, field_interpretation, field_level_aggregates
        )

    def update_field_level_aggregates_on_processed_field_interpretation(
        self, field: Field, field_interpretation: FieldInterpretation
    ):
        self.insert_field_level_aggregates(
            {
                "field_level_confidence_metrics": field_interpretation.get_field_level_confidence_metrics(),
            }
        )
        super().update_field_level_aggregates_on_processed_field(field)

    def update_file_level_aggregates_on_processed_field_interpretation(
        self,
        field: Field,
        field_interpretation: FieldInterpretation,
        field_level_aggregates,
    ):
        self.file_level_aggregates["confidence_metrics_for_file"][
            field.field_label
        ] = field_interpretation.get_field_level_confidence_metrics()

        super().update_file_level_aggregates_on_processed_field(
            field, field_level_aggregates
        )

    def update_directory_level_aggregates_on_processed_field_interpretation(
        self,
        field: Field,
        field_interpretation: FieldInterpretation,
        field_level_aggregates,
    ):
        super().update_directory_level_aggregates_on_processed_field(
            field, field_level_aggregates
        )


class TemplateInterpretationPass(FilePassAggregates):
    def initialize_file_level_aggregates(
        self,
        file_path,
        all_field_detection_types,
        field_detection_type_wise_detection_aggregates,
        field_label_wise_detection_aggregates,
    ):
        super().initialize_file_level_aggregates(file_path)
        self.insert_file_level_aggregates(
            {
                "confidence_metrics_for_file": {},
                "files_by_label_count": StatsByLabel("processed", "multi_marked"),
                "read_response_flags": {
                    "is_multi_marked": False,
                    "multi_marked_fields": [],
                    "is_identifier_multi_marked": False,
                },
                "field_detection_type_wise_aggregates": {
                    key: {"fields_count": StatsByLabel("processed")}
                    for key in all_field_detection_types
                },
            }
        )

    def initialize_directory_level_aggregates(
        self, initial_directory_path, all_field_detection_types
    ):
        super().initialize_directory_level_aggregates(initial_directory_path)
        self.insert_directory_level_aggregates(
            {
                "files_by_label_count": StatsByLabel("processed", "multi_marked"),
                "field_detection_type_wise_aggregates": {
                    key: {"fields_count": StatsByLabel("processed")}
                    for key in all_field_detection_types
                },
            }
        )

    # This overrides parent definition -
    def update_aggregates_on_processed_field_interpretation(
        self,
        current_omr_response,
        field: Field,
        field_interpretation: FieldInterpretation,
        # TODO: see if detection also needs this arg (field_type_runner_field_level_aggregates)
        field_type_runner_field_level_aggregates,
    ):
        self.update_field_level_aggregates_on_processed_field_interpretation(
            current_omr_response,
            field,
            field_interpretation,
            field_type_runner_field_level_aggregates,
        )
        template_field_level_aggregates = self.get_field_level_aggregates()

        self.update_file_level_aggregates_on_processed_field_interpretation(
            field, field_interpretation, template_field_level_aggregates
        )
        self.update_directory_level_aggregates_on_processed_field_interpretation(
            field, field_interpretation, template_field_level_aggregates
        )

    def update_field_level_aggregates_on_processed_field_interpretation(
        self,
        current_omr_response,
        field: Field,
        field_interpretation: FieldInterpretation,
        field_type_runner_field_level_aggregates,
    ):
        read_response_flags = self.file_level_aggregates["read_response_flags"]

        if field_type_runner_field_level_aggregates["is_multi_marked"]:
            read_response_flags["is_multi_marked"] = True
            read_response_flags["multi_marked_fields"].append(field)
            # TODO: define identifier_labels
            # if field.is_part_of_identifier():
            # if field_label in self.template.identifier_labels:
            #     read_response_flags["is_identifier_multi_marked"] = True

        # TODO: is there a better way for this?
        self.insert_field_level_aggregates(
            {"from_field_type_runner": field_type_runner_field_level_aggregates}
        )
        super().update_field_level_aggregates_on_processed_field(field)
        # TODO: support for more validations here?

    def update_file_level_aggregates_on_processed_field_interpretation(
        self,
        field: Field,
        field_interpretation: FieldInterpretation,
        template_field_level_aggregates,
    ):
        super().update_file_level_aggregates_on_processed_field(
            field, template_field_level_aggregates
        )

        self.file_level_aggregates["confidence_metrics_for_file"][
            field.field_label
        ] = field_interpretation.get_field_level_confidence_metrics()

    def update_directory_level_aggregates_on_processed_field_interpretation(
        self, field: Field, field_interpretation, template_field_level_aggregates
    ):
        super().update_directory_level_aggregates_on_processed_field(
            field, template_field_level_aggregates
        )
        field_detection_type = field.field_detection_type

        field_detection_type_wise_aggregates = self.directory_level_aggregates[
            "field_detection_type_wise_aggregates"
        ][field_detection_type]
        # Update the processed field count for that runner
        field_detection_type_wise_aggregates["fields_count"].push("processed")

    # TODO: check if passing runners is really needed or not here (can move inside field_detection_type_runner?) -
    def update_aggregates_on_processed_file(
        self, file_path, field_detection_type_runners
    ):
        super().update_aggregates_on_processed_file(file_path)

        field_detection_type_wise_aggregates = self.file_level_aggregates[
            "field_detection_type_wise_aggregates"
        ]
        for field_detection_type_runner in field_detection_type_runners.values():
            field_detection_type_wise_aggregates[
                field_detection_type_runner.field_detection_type
            ] = field_detection_type_runner.get_file_level_interpretation_aggregates()

        # Update read_response_flags
        read_response_flags = self.file_level_aggregates["read_response_flags"]

        if read_response_flags["is_multi_marked"]:
            print("self.directory_level_aggregates", self.directory_level_aggregates)
            self.directory_level_aggregates["files_by_label_count"].push("multi_marked")
