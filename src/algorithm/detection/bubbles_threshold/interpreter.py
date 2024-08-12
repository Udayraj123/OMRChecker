from src.algorithm.detection.field import FieldInterpreter
from src.processors.constants import FieldDetectionType
from src.utils.logger import logger


class BubblesFieldInterpreter(FieldInterpreter):
    def __init__(self, field, field_block):
        self.type = FieldDetectionType.BUBBLES_THRESHOLD
        super().__init__(field, field_block)
        # TODO: copy field_block shifts into field in during init
        # TODO: remove field_block coupling as all coordinates and bubble dimensions should be parsed at this point
        self.shifts = field_block.shifts
        self.bubble_dimensions = field.bubble_dimensions
        self.empty_value = field.empty_value

    def get_detected_string(self):
        detected_bubbles = [
            bubble.bubble_value for bubble in self.bubble_detections if bubble.is_marked
        ]
        # Empty value logic
        if len(detected_bubbles) == 0:
            return self.empty_value

        # Concatenation logic
        return "".join(detected_bubbles)

    def update_field_level_aggregates(self, file_aggregate_params, omr_response):
        field_label, local_threshold_for_field = (
            self.field_label,
            self.local_threshold_for_field,
        )
        local_thresholds = file_aggregate_params["local_thresholds"]
        local_thresholds["running_total"] += local_threshold_for_field
        local_thresholds["processed_fields_count"] += 1
        local_thresholds["running_average"] = (
            local_thresholds["running_total"]
            / local_thresholds["processed_fields_count"]
        )

        read_response_flags = file_aggregate_params["read_response_flags"]

        # Check for multi_marked for the before-concatenations case
        if len(self.detected_bubbles) > 1:
            read_response_flags["multi_marked"] = True
            read_response_flags["multi_marked_fields"][
                field_label
            ] = self.detected_bubbles
        # TODO: more field level validations here

    @staticmethod
    def get_field_level_confidence_metrics(
        field,
        field_bubble_means,
        config,
        local_threshold_for_field,
        all_fields_threshold_for_file,
        local_max_jump,
        global_max_jump,
    ):
        if not config.outputs.show_confidence_metrics:
            return {}

        (
            MIN_JUMP,
            GLOBAL_THRESHOLD_MARGIN,
            MIN_JUMP_SURPLUS_FOR_GLOBAL_FALLBACK,
            CONFIDENT_JUMP_SURPLUS_FOR_DISPARITY,
        ) = map(
            config.thresholding.get,
            [
                "MIN_JUMP",
                "GLOBAL_THRESHOLD_MARGIN",
                "MIN_JUMP_SURPLUS_FOR_GLOBAL_FALLBACK",
                "CONFIDENT_JUMP_SURPLUS_FOR_DISPARITY",
            ],
        )
        bubbles_in_doubt = {
            "by_disparity": [],
            "by_jump": [],
            "global_higher": [],
            "global_lower": [],
        }

        # Main detection logic:
        for bubble in field_bubble_means:
            global_bubble_is_marked = all_fields_threshold_for_file > bubble.mean_value
            local_bubble_is_marked = local_threshold_for_field > bubble.mean_value
            # 1. Disparity in global/local threshold output
            if global_bubble_is_marked != local_bubble_is_marked:
                bubbles_in_doubt["by_disparity"].append(bubble)
        # 5. High confidence if the gap is very large compared to MIN_JUMP
        is_global_jump_confident = (
            global_max_jump > MIN_JUMP + CONFIDENT_JUMP_SURPLUS_FOR_DISPARITY
        )
        is_local_jump_confident = (
            local_max_jump > MIN_JUMP + CONFIDENT_JUMP_SURPLUS_FOR_DISPARITY
        )

        # TODO: FieldInterpreter.bubbles = field_bubble_means
        thresholds_string = f"global={round(all_fields_threshold_for_file, 2)} local={round(local_threshold_for_field, 2)} global_margin={GLOBAL_THRESHOLD_MARGIN}"
        jumps_string = f"global_max_jump={round(global_max_jump, 2)} local_max_jump={round(local_max_jump, 2)} MIN_JUMP={MIN_JUMP} SURPLUS={CONFIDENT_JUMP_SURPLUS_FOR_DISPARITY}"
        if len(bubbles_in_doubt["by_disparity"]) > 0:
            logger.warning(
                f"found disparity in field: {field.field_label}",
                list(map(str, bubbles_in_doubt["by_disparity"])),
                thresholds_string,
            )
            # 5.2 if the gap is very large compared to MIN_JUMP, but still there is disparity
            if is_global_jump_confident:
                logger.warning(
                    f"is_global_jump_confident but still has disparity",
                    jumps_string,
                )
            elif is_local_jump_confident:
                logger.warning(
                    f"is_local_jump_confident but still has disparity",
                    jumps_string,
                )
        else:
            # Note: debug logs are disabled by default
            logger.debug(
                f"party_matched for field: {field.field_label}",
                thresholds_string,
            )

            # 5.1 High confidence if the gap is very large compared to MIN_JUMP
            if is_local_jump_confident:
                # Higher weightage for confidence
                logger.debug(
                    f"is_local_jump_confident => increased confidence",
                    jumps_string,
                )
            # No output disparity, but -
            # 2.1 global threshold is "too close" to lower bubbles
            bubbles_in_doubt["global_lower"] = [
                bubble
                for bubble in field_bubble_means
                if GLOBAL_THRESHOLD_MARGIN
                > max(
                    GLOBAL_THRESHOLD_MARGIN,
                    all_fields_threshold_for_file - bubble.mean_value,
                )
            ]

            if len(bubbles_in_doubt["global_lower"]) > 0:
                logger.warning(
                    'bubbles_in_doubt["global_lower"]',
                    list(map(str, bubbles_in_doubt["global_lower"])),
                )
            # 2.2 global threshold is "too close" to higher bubbles
            bubbles_in_doubt["global_higher"] = [
                bubble
                for bubble in field_bubble_means
                if GLOBAL_THRESHOLD_MARGIN
                > max(
                    GLOBAL_THRESHOLD_MARGIN,
                    bubble.mean_value - all_fields_threshold_for_file,
                )
            ]

            if len(bubbles_in_doubt["global_higher"]) > 0:
                logger.warning(
                    'bubbles_in_doubt["global_higher"]',
                    list(map(str, bubbles_in_doubt["global_higher"])),
                )

            # 3. local jump outliers are close to the configured min_jump but below it.
            # Note: This factor indicates presence of cases like partially filled bubbles,
            # mis-aligned box boundaries or some form of unintentional marking over the bubble
            if len(field_bubble_means) > 1:
                # TODO: move util
                def get_jumps_in_bubble_means(field_bubble_means):
                    # get sorted array
                    sorted_field_bubble_means = sorted(
                        field_bubble_means,
                    )
                    # get jumps
                    jumps_in_bubble_means = []
                    previous_bubble = sorted_field_bubble_means[0]
                    previous_mean = previous_bubble.mean_value
                    for i in range(1, len(sorted_field_bubble_means)):
                        bubble = sorted_field_bubble_means[i]
                        current_mean = bubble.mean_value
                        jumps_in_bubble_means.append(
                            [
                                round(current_mean - previous_mean, 2),
                                previous_bubble,
                            ]
                        )
                        previous_bubble = bubble
                        previous_mean = current_mean
                    return jumps_in_bubble_means

                jumps_in_bubble_means = get_jumps_in_bubble_means(field_bubble_means)
                bubbles_in_doubt["by_jump"] = [
                    bubble
                    for jump, bubble in jumps_in_bubble_means
                    if MIN_JUMP_SURPLUS_FOR_GLOBAL_FALLBACK
                    > max(
                        MIN_JUMP_SURPLUS_FOR_GLOBAL_FALLBACK,
                        MIN_JUMP - jump,
                    )
                ]

                if len(bubbles_in_doubt["by_jump"]) > 0:
                    logger.warning(
                        'bubbles_in_doubt["by_jump"]',
                        list(map(str, bubbles_in_doubt["by_jump"])),
                    )
                    logger.warning(
                        list(map(str, jumps_in_bubble_means)),
                    )

                # TODO: aggregate the bubble metrics into the Field objects
                # collect_bubbles_in_doubt(bubbles_in_doubt["by_disparity"], bubbles_in_doubt["global_higher"], bubbles_in_doubt["global_lower"], bubbles_in_doubt["by_jump"])
        confidence_metrics = {
            "bubbles_in_doubt": bubbles_in_doubt,
            "is_global_jump_confident": is_global_jump_confident,
            "is_local_jump_confident": is_local_jump_confident,
            "local_max_jump": local_max_jump,
            "field_label": field.field_label,
        }
        return confidence_metrics
