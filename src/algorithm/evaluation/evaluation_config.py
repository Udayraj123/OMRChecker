import re

from src.algorithm.evaluation.evaluation_config_for_set import EvaluationConfigForSet
from src.schemas.constants import DEFAULT_SET_NAME
from src.utils.logger import logger
from src.utils.parsing import OVERRIDE_MERGER, open_evaluation_with_defaults


class EvaluationConfig:
    def __init__(self, curr_dir, local_evaluation_path, template, tuning_config):
        self.path = local_evaluation_path
        default_evaluation_json = open_evaluation_with_defaults(local_evaluation_path)
        # .pop() will delete the conditional_sets key from the default json if it exists
        conditional_sets = default_evaluation_json.pop("conditional_sets", [])

        self.conditional_sets = []
        for conditional_set in conditional_sets:
            name, matcher = conditional_set["name"], conditional_set["matcher"]
            self.conditional_sets.append([name, matcher])
        self.validate_conditional_sets()

        # TODO: allow default evaluation to not contain any question/answer/config
        self.default_evaluation_config = EvaluationConfigForSet(
            DEFAULT_SET_NAME, curr_dir, default_evaluation_json, template, tuning_config
        )
        # Currently source_type + options + marking_schemes need to be completely parsed before merging
        # This is due to:
        # 1. multiple ways to denote same questions i.e. field strings
        # 2. The child marking_schemes may want to override parent's questions partially over different schemes.
        # 3. merging schemas in the case of different source_type in child can result in an unexpected state in child schema.
        partial_default_evaluation_json = {
            "outputs_configuration": default_evaluation_json["outputs_configuration"],
        }
        self.exclude_files = self.default_evaluation_config.get_exclude_files()
        self.set_mapping = {}
        for conditional_set in conditional_sets:
            set_name, evaluation_json_for_set = map(
                conditional_set.get, ["name", "evaluation"]
            )

            logger.debug("evaluation_json_for_set", set_name, evaluation_json_for_set)
            # Merge two jsons, override the arrays(if any) instead of appending
            merged_evaluation_json = OVERRIDE_MERGER.merge(
                partial_default_evaluation_json, evaluation_json_for_set
            )
            logger.debug("merged_evaluation_json", merged_evaluation_json)
            evaluation_config_for_set = EvaluationConfigForSet(
                set_name,
                curr_dir,
                merged_evaluation_json,
                template,
                tuning_config,
                self.default_evaluation_config,
            )
            self.set_mapping[set_name] = evaluation_config_for_set
            self.exclude_files += evaluation_config_for_set.get_exclude_files()

    def __str__(self):
        return str(self.path)

    def get_exclude_files(self):
        return self.exclude_files

    def validate_conditional_sets(self):
        all_names = set()
        for name, _ in self.conditional_sets:
            if name in all_names:
                raise Exception(
                    f"Repeated set name {name} in conditional_sets in the given evaluation.json: {self.path}"
                )
            all_names.add(name)

    # Public function
    def get_evaluation_config_for_response(self, concatenated_response, file_path):
        matched_key = self.get_matching_set(concatenated_response, file_path)
        if matched_key is None:
            return self.default_evaluation_config
        return self.set_mapping[matched_key]

    def get_matching_set(self, concatenated_response, file_path):
        formatting_fields = {
            **concatenated_response,
            "file_path": str(file_path),
            "file_name": str(file_path.name),
        }
        # loop on all sets and return first matched set
        for name, matcher in self.conditional_sets:
            format_string, match_regex = matcher["formatString"], matcher["matchRegex"]
            try:
                formatted_string = format_string.format(**formatting_fields)
                if re.search(match_regex, formatted_string) is not None:
                    return name
            except:  # NOQA
                return None
        return None
