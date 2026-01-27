from src.schemas.models.template import (
    AlignmentConfig,
    AlignmentMarginsConfig,
    OutputColumnsConfig,
    SortFilesConfig,
    TemplateConfig,
)

# Create default template config instance
TEMPLATE_DEFAULTS = TemplateConfig(
    bubble_dimensions=[10, 10],
    template_dimensions=[1200, 1600],
    alignment=AlignmentConfig(
        margins=AlignmentMarginsConfig(top=0, bottom=0, left=0, right=0),
        max_displacement=10,
    ),
    conditional_sets=[],
    custom_labels={},
    custom_bubble_field_types={},
    empty_value="",
    field_blocks={},
    field_blocks_offset=[0, 0],
    output_columns=OutputColumnsConfig(
        custom_order=[],
        sort_type="ALPHANUMERIC",
        sort_order="ASC",
    ),
    pre_processors=[],
    processing_image_shape=[900, 650],
    sort_files=SortFilesConfig(enabled=False),
)
