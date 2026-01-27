"""Typed dataclass models for template configuration."""

from dataclasses import dataclass, field

from src.utils.json_conversion import convert_dict_keys_to_snake
from src.utils.serialization import dataclass_to_dict


@dataclass
class AlignmentMarginsConfig:
    """Configuration for alignment margins."""

    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "AlignmentMarginsConfig":
        """Create AlignmentMarginsConfig from dictionary with camelCase keys."""
        data = convert_dict_keys_to_snake(data)
        return cls(**data)

    def to_dict(self) -> dict:
        """Convert AlignmentMarginsConfig to dictionary."""
        return dataclass_to_dict(self)


@dataclass
class AlignmentConfig:
    """Configuration for template alignment."""

    margins: AlignmentMarginsConfig = field(default_factory=AlignmentMarginsConfig)
    maxDisplacement: int = 10

    @classmethod
    def from_dict(cls, data: dict) -> "AlignmentConfig":
        """Create AlignmentConfig from dictionary with camelCase keys."""
        data = convert_dict_keys_to_snake(data)
        if "margins" in data:
            data["margins"] = AlignmentMarginsConfig.from_dict(data["margins"])
        return cls(**data)

    def to_dict(self) -> dict:
        """Convert AlignmentConfig to dictionary."""
        return dataclass_to_dict(self)


@dataclass
class OutputColumnsConfig:
    """Configuration for output columns ordering and sorting."""

    customOrder: list[str] = field(default_factory=list)
    sortType: str = "ALPHANUMERIC"
    sortOrder: str = "ASC"

    @classmethod
    def from_dict(cls, data: dict) -> "OutputColumnsConfig":
        """Create OutputColumnsConfig from dictionary with camelCase keys."""
        data = convert_dict_keys_to_snake(data)
        return cls(**data)

    def to_dict(self) -> dict:
        """Convert OutputColumnsConfig to dictionary."""
        return dataclass_to_dict(self)


@dataclass
class SortFilesConfig:
    """Configuration for file sorting."""

    enabled: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "SortFilesConfig":
        """Create SortFilesConfig from dictionary with camelCase keys."""
        data = convert_dict_keys_to_snake(data)
        return cls(**data)

    def to_dict(self) -> dict:
        """Convert SortFilesConfig to dictionary."""
        return dataclass_to_dict(self)


@dataclass
class TemplateConfig:
    """Main template configuration object.

    This represents the structure of template.json files used for OMR sheet
    layout definition and field detection.
    """

    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    conditionalSets: list = field(default_factory=list)
    customLabels: dict = field(default_factory=dict)
    customBubbleFieldTypes: dict = field(default_factory=dict)
    emptyValue: str = ""
    fieldBlocks: dict = field(default_factory=dict)
    fieldBlocksOffset: list[int] = field(default_factory=lambda: [0, 0])
    outputColumns: OutputColumnsConfig = field(default_factory=OutputColumnsConfig)
    preProcessors: list = field(default_factory=list)
    processingImageShape: list[int] = field(default_factory=lambda: [900, 650])
    sortFiles: SortFilesConfig = field(default_factory=SortFilesConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "TemplateConfig":
        """Create TemplateConfig from dictionary (typically from JSON).

        Converts camelCase keys from JSON to snake_case for Python dataclass fields.

        Args:
            data: Dictionary containing template configuration data (with camelCase keys)

        Returns:
            TemplateConfig instance with nested dataclasses
        """
        # Convert all keys from camelCase to snake_case
        data = convert_dict_keys_to_snake(data)

        return cls(
            alignment=AlignmentConfig.from_dict(data.get("alignment", {})),
            conditionalSets=data.get("conditional_sets", []),
            customLabels=data.get("custom_labels", {}),
            customBubbleFieldTypes=data.get("custom_bubble_field_types", {}),
            emptyValue=data.get("empty_value", ""),
            fieldBlocks=data.get("field_blocks", {}),
            fieldBlocksOffset=data.get("field_blocks_offset", [0, 0]),
            outputColumns=OutputColumnsConfig.from_dict(data.get("output_columns", {})),
            preProcessors=data.get("pre_processors", []),
            processingImageShape=data.get("processing_image_shape", [900, 650]),
            sortFiles=SortFilesConfig.from_dict(data.get("sort_files", {})),
        )

    def to_dict(self) -> dict:
        """Convert TemplateConfig to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the template config
        """
        return dataclass_to_dict(self)
