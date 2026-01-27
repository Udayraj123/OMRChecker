"""Typed dataclass models for evaluation configuration."""

from dataclasses import dataclass, field

from src.utils.json_conversion import convert_dict_keys_to_snake
from src.utils.serialization import dataclass_to_dict


@dataclass
class DrawScoreConfig:
    """Configuration for drawing score on output images."""

    enabled: bool = False
    position: list[int] = field(default_factory=lambda: [200, 200])
    score_format_string: str = "Score: {score}"
    size: float = 1.5

    @classmethod
    def from_dict(cls, data: dict) -> "DrawScoreConfig":
        """Create DrawScoreConfig from dictionary with camelCase keys."""
        data = convert_dict_keys_to_snake(data)
        return cls(**data)

    def to_dict(self) -> dict:
        """Convert DrawScoreConfig to dictionary."""
        return dataclass_to_dict(self)


@dataclass
class DrawAnswersSummaryConfig:
    """Configuration for drawing answers summary on output images."""

    enabled: bool = False
    position: list[int] = field(default_factory=lambda: [200, 600])
    answers_summary_format_string: str = (
        "Correct: {correct} Incorrect: {incorrect} Unmarked: {unmarked}"
    )
    size: float = 1.0

    @classmethod
    def from_dict(cls, data: dict) -> "DrawAnswersSummaryConfig":
        """Create DrawAnswersSummaryConfig from dictionary with camelCase keys."""
        data = convert_dict_keys_to_snake(data)
        return cls(**data)

    def to_dict(self) -> dict:
        """Convert DrawAnswersSummaryConfig to dictionary."""
        return dataclass_to_dict(self)


@dataclass
class DrawAnswerGroupsConfig:
    """Configuration for drawing answer groups."""

    enabled: bool = True
    color_sequence: list[str] = field(
        default_factory=lambda: ["#8DFBC4", "#F7FB8D", "#8D9EFB", "#EA666F"]
    )

    @classmethod
    def from_dict(cls, data: dict) -> "DrawAnswerGroupsConfig":
        """Create DrawAnswerGroupsConfig from dictionary with camelCase keys."""
        data = convert_dict_keys_to_snake(data)
        return cls(**data)

    def to_dict(self) -> dict:
        """Convert DrawAnswerGroupsConfig to dictionary."""
        return dataclass_to_dict(self)


@dataclass
class DrawQuestionVerdictsConfig:
    """Configuration for drawing question verdicts on output images."""

    enabled: bool = True
    verdict_colors: dict[str, str | None] = field(
        default_factory=lambda: {
            "correct": "#00FF00",
            "neutral": None,
            "incorrect": "#FF0000",
            "bonus": "#00DDDD",
        }
    )
    verdict_symbol_colors: dict[str, str] = field(
        default_factory=lambda: {
            "positive": "#000000",
            "neutral": "#000000",
            "negative": "#000000",
            "bonus": "#000000",
        }
    )
    draw_answer_groups: DrawAnswerGroupsConfig = field(
        default_factory=DrawAnswerGroupsConfig
    )

    @classmethod
    def from_dict(cls, data: dict) -> "DrawQuestionVerdictsConfig":
        """Create DrawQuestionVerdictsConfig from dictionary with camelCase keys."""
        data = convert_dict_keys_to_snake(data)
        if "draw_answer_groups" in data:
            data["draw_answer_groups"] = DrawAnswerGroupsConfig.from_dict(
                data["draw_answer_groups"]
            )
        return cls(**data)

    def to_dict(self) -> dict:
        """Convert DrawQuestionVerdictsConfig to dictionary."""
        return dataclass_to_dict(self)


@dataclass
class DrawDetectedBubbleTextsConfig:
    """Configuration for drawing detected bubble texts."""

    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "DrawDetectedBubbleTextsConfig":
        """Create DrawDetectedBubbleTextsConfig from dictionary with camelCase keys."""
        data = convert_dict_keys_to_snake(data)
        return cls(**data)

    def to_dict(self) -> dict:
        """Convert DrawDetectedBubbleTextsConfig to dictionary."""
        return dataclass_to_dict(self)


@dataclass
class OutputsConfiguration:
    """Configuration for evaluation outputs and visualization."""

    should_explain_scoring: bool = False
    should_export_explanation_csv: bool = False
    draw_score: DrawScoreConfig = field(default_factory=DrawScoreConfig)
    draw_answers_summary: DrawAnswersSummaryConfig = field(
        default_factory=DrawAnswersSummaryConfig
    )
    draw_question_verdicts: DrawQuestionVerdictsConfig = field(
        default_factory=DrawQuestionVerdictsConfig
    )
    draw_detected_bubble_texts: DrawDetectedBubbleTextsConfig = field(
        default_factory=DrawDetectedBubbleTextsConfig
    )

    @classmethod
    def from_dict(cls, data: dict) -> "OutputsConfiguration":
        """Create OutputsConfiguration from dictionary with camelCase keys."""
        data = convert_dict_keys_to_snake(data)
        return cls(
            should_explain_scoring=data.get("should_explain_scoring", False),
            should_export_explanation_csv=data.get(
                "should_export_explanation_csv", False
            ),
            draw_score=DrawScoreConfig.from_dict(data.get("draw_score", {})),
            draw_answers_summary=DrawAnswersSummaryConfig.from_dict(
                data.get("draw_answers_summary", {})
            ),
            draw_question_verdicts=DrawQuestionVerdictsConfig.from_dict(
                data.get("draw_question_verdicts", {})
            ),
            draw_detected_bubble_texts=DrawDetectedBubbleTextsConfig.from_dict(
                data.get("draw_detected_bubble_texts", {})
            ),
        )

    def to_dict(self) -> dict:
        """Convert OutputsConfiguration to dictionary."""
        return dataclass_to_dict(self)


@dataclass
class EvaluationConfig:
    """Main evaluation configuration object.

    This represents the structure of evaluation.json files used for answer key
    matching and scoring configuration.
    """

    options: dict = field(default_factory=dict)
    marking_schemes: dict = field(default_factory=dict)
    conditional_sets: list = field(default_factory=list)
    outputs_configuration: OutputsConfiguration = field(
        default_factory=OutputsConfiguration
    )

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationConfig":
        """Create EvaluationConfig from dictionary (typically from JSON).

        Converts camelCase keys from JSON to snake_case for Python dataclass fields.

        Args:
            data: Dictionary containing evaluation configuration data (with camelCase keys)

        Returns:
            EvaluationConfig instance with nested dataclasses
        """
        # Convert all keys from camelCase to snake_case
        data = convert_dict_keys_to_snake(data)

        return cls(
            options=data.get("options", {}),
            marking_schemes=data.get("marking_schemes", {}),
            conditional_sets=data.get("conditional_sets", []),
            outputs_configuration=OutputsConfiguration.from_dict(
                data.get("outputs_configuration", {})
            ),
        )

    def to_dict(self) -> dict:
        """Convert EvaluationConfig to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the evaluation config
        """
        return dataclass_to_dict(self)
