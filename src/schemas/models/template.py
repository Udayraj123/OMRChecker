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
    max_displacement: int = 10

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

    custom_order: list[str] = field(default_factory=list)
    sort_type: str = "ALPHANUMERIC"
    sort_order: str = "ASC"

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

    # Required template properties
    bubble_dimensions: list[int] = field(default_factory=lambda: [10, 10])
    template_dimensions: list[int] = field(default_factory=lambda: [1200, 1600])

    # Configuration properties
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    conditional_sets: list = field(default_factory=list)
    custom_labels: dict = field(default_factory=dict)
    custom_bubble_field_types: dict = field(default_factory=dict)
    empty_value: str = ""
    field_blocks: dict = field(default_factory=dict)
    field_blocks_offset: list[int] = field(default_factory=lambda: [0, 0])
    output_columns: OutputColumnsConfig = field(default_factory=OutputColumnsConfig)
    pre_processors: list = field(default_factory=list)
    processing_image_shape: list[int] = field(default_factory=lambda: [900, 650])
    sort_files: SortFilesConfig = field(default_factory=SortFilesConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "TemplateConfig":
        """Create TemplateConfig from dictionary (typically from JSON).

        Converts camelCase keys from JSON to snake_case for Python dataclass fields.
        Preserves user-defined names in customBubbleFieldTypes and fieldBlocks.

        Args:
            data: Dictionary containing template configuration data (with camelCase keys)

        Returns:
            TemplateConfig instance with nested dataclasses
        """
        # Preserve and convert customBubbleFieldTypes
        # Keep type names (keys), but convert keys within each type definition
        custom_bubble_field_types_raw = data.get("customBubbleFieldTypes", {})
        custom_bubble_field_types_converted = {}
        for type_name, type_data in custom_bubble_field_types_raw.items():
            # Preserve type name, convert keys within type definition
            custom_bubble_field_types_converted[type_name] = convert_dict_keys_to_snake(
                type_data
            )

        # Preserve and convert fieldBlocks
        # Keep block names (keys), but convert keys within each block
        field_blocks_raw = data.get("fieldBlocks", {})
        field_blocks_converted = {}
        for block_name, block_data in field_blocks_raw.items():
            # Preserve block name, convert keys within block
            field_blocks_converted[block_name] = convert_dict_keys_to_snake(block_data)

        # Convert all other top-level keys from camelCase to snake_case
        data = convert_dict_keys_to_snake(data)

        return cls(
            bubble_dimensions=data.get("bubble_dimensions", [10, 10]),
            template_dimensions=data.get("template_dimensions", [1200, 1600]),
            alignment=AlignmentConfig.from_dict(data.get("alignment", {})),
            conditional_sets=data.get("conditional_sets", []),
            custom_labels=data.get("custom_labels", {}),
            # Use converted custom bubble field types (type names preserved, keys converted)
            custom_bubble_field_types=custom_bubble_field_types_converted,
            empty_value=data.get("empty_value", ""),
            # Use converted field blocks (block names preserved, keys converted)
            field_blocks=field_blocks_converted,
            field_blocks_offset=data.get("field_blocks_offset", [0, 0]),
            output_columns=OutputColumnsConfig.from_dict(
                data.get("output_columns", {})
            ),
            pre_processors=data.get("pre_processors", []),
            processing_image_shape=data.get("processing_image_shape", [900, 650]),
            sort_files=SortFilesConfig.from_dict(data.get("sort_files", {})),
        )

    def to_dict(self) -> dict:
        """Convert TemplateConfig to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the template config
        """
        return dataclass_to_dict(self)
