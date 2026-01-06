"""Core OMRProcessor - Pure library interface for OMR processing.

This module provides a clean, reusable interface for OMR processing that can be
used both by CLI applications and as a library API. It extracts the core logic
from entry.py without CLI-specific dependencies.
"""

from collections.abc import Callable
from pathlib import Path
from time import time

from src.core.types import DirectoryProcessingResult, OMRResult, ProcessorConfig
from src.exceptions import InputDirectoryNotFoundError, TemplateNotFoundError
from src.processors.evaluation.evaluation_config import EvaluationConfig
from src.processors.template.template import Template
from src.schemas.defaults import CONFIG_DEFAULTS
from src.schemas.models.config import Config
from src.utils import constants
from src.utils.image import ImageUtils
from src.utils.logger import logger
from src.utils.parsing import open_config_with_defaults


class OMRProcessor:
    """Main processor for OMR sheet processing.

    This class provides the core OMR processing functionality without CLI
    dependencies, making it suitable for both command-line and library usage.

    Example (Library usage):
        ```python
        from pathlib import Path
        from src.core import OMRProcessor, ProcessorConfig

        config = ProcessorConfig(
            input_dir=Path("inputs/sample"),
            output_dir=Path("outputs"),
            debug=False
        )
        processor = OMRProcessor(config)
        results = processor.process_directory()

        for result in results.results:
            if result.status == "success":
                print(f"{result.file_name}: {result.omr_response}")
        ```

    Example (Single image):
        ```python
        result = processor.process_image(Path("inputs/sample/sheet1.jpg"))
        print(result.omr_response)
        ```
    """

    def __init__(self, config: ProcessorConfig) -> None:
        """Initialize the OMR processor with configuration.

        Args:
            config: Processor configuration (can be created from CLI args)
        """
        self.config = config
        self.args = config.to_args()  # For backward compatibility with existing code

    def process_directory(
        self,
        input_dir: Path | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> DirectoryProcessingResult:
        """Process all OMR sheets in a directory tree.

        This method recursively processes all images in the input directory,
        respecting the template hierarchy and configuration files.

        Args:
            input_dir: Directory to process (uses config.input_dir if None)
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            DirectoryProcessingResult containing all processing results

        Raises:
            InputDirectoryNotFoundError: If the input directory doesn't exist
        """
        input_dir = input_dir or self.config.input_dir
        if input_dir is None:
            msg = "Input directory not specified in config or arguments"
            raise ValueError(msg)

        if not input_dir.exists():
            raise InputDirectoryNotFoundError(input_dir)

        start_time = time()
        result = DirectoryProcessingResult()

        # Process directory tree recursively
        self._process_directory_recursive(
            root_dir=input_dir,
            curr_dir=input_dir,
            result=result,
            progress_callback=progress_callback,
        )

        result.processing_time = time() - start_time
        return result

    def process_image(
        self,
        image_path: Path,
        template: Template | None = None,
        tuning_config: Config | None = None,
        evaluation_config: EvaluationConfig | None = None,  # noqa: ARG002
    ) -> OMRResult:
        """Process a single OMR image.

        This is useful for library usage when you want to process individual
        images rather than directory trees.

        Args:
            image_path: Path to the OMR image
            template: Optional template (will search for template.json if None)
            tuning_config: Optional config (will use defaults if None)
            evaluation_config: Optional evaluation config

        Returns:
            OMRResult containing the processing results
        """
        start_time = time()

        # Load template and config if not provided
        if template is None or tuning_config is None:
            template, tuning_config = self._load_template_and_config(image_path.parent)

        if template is None:
            return OMRResult(
                file_name=image_path.name,
                file_path=image_path,
                status="error",
                error="No template found",
            )

        # Read the image
        try:
            gray_image, colored_image = ImageUtils.read_image_util(
                image_path, tuning_config
            )
        except Exception as e:
            return OMRResult(
                file_name=image_path.name,
                file_path=image_path,
                status="error",
                error=f"Failed to read image: {e}",
            )

        # Process the image through the pipeline
        try:
            context = template.process_file(image_path, gray_image, colored_image)

            # Create result from context
            return OMRResult(
                file_name=image_path.name,
                file_path=image_path,
                status="success" if context.gray_image is not None else "error",
                error="NO_MARKER_ERR" if context.gray_image is None else None,
                omr_response=context.omr_response,
                raw_omr_response=context.metadata.get("raw_omr_response", {}),
                score=context.score,
                evaluation_meta=context.evaluation_meta,
                is_multi_marked=context.is_multi_marked,
                field_interpretations=context.field_id_to_interpretation,
                processing_time=time() - start_time,
            )
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return OMRResult(
                file_name=image_path.name,
                file_path=image_path,
                status="error",
                error=str(e),
                processing_time=time() - start_time,
            )

    def _process_directory_recursive(  # noqa: PLR0913
        self,
        root_dir: Path,
        curr_dir: Path,
        result: DirectoryProcessingResult,
        template: Template | None = None,
        tuning_config: Config = CONFIG_DEFAULTS,
        evaluation_config: EvaluationConfig | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """Recursively process directories (internal method).

        This mirrors the logic from entry.py's process_directory_wise but
        uses the cleaner OMRResult-based interface.
        """
        # Update local tuning_config if config file exists
        local_config_path = curr_dir.joinpath(constants.CONFIG_FILENAME)
        if local_config_path.exists():
            tuning_config = open_config_with_defaults(local_config_path, self.args)
            logger.set_log_levels(tuning_config.outputs.show_logs_by_type)

        # Update local template if template file exists
        local_template_path = curr_dir.joinpath(constants.TEMPLATE_FILENAME)
        if local_template_path.exists():
            template = Template(local_template_path, tuning_config, self.args)

        # Find subdirectories
        subdirs = [d for d in curr_dir.iterdir() if d.is_dir()]

        # Find images in current directory
        exts = ("*.[pP][nN][gG]", "*.[jJ][pP][gG]", "*.[jJ][pP][eE][gG]")
        omr_files = sorted([f for ext in exts for f in curr_dir.glob(ext)])

        # Exclude files if template specifies
        excluded_files: set[str] = set()
        if template:
            excluded_files.update(
                str(exclude_file) for exclude_file in template.get_exclude_files()
            )

        # Load evaluation config if present
        local_evaluation_path = curr_dir.joinpath(constants.EVALUATION_FILENAME)
        if (
            not self.config.set_layout
            and local_evaluation_path.exists()
            and local_template_path.exists()
        ):
            evaluation_config = EvaluationConfig(
                curr_dir,
                local_evaluation_path,
                template,
                tuning_config,
            )
            excluded_files.update(
                str(exclude_file)
                for exclude_file in evaluation_config.get_exclude_files()
            )

        omr_files = [f for f in omr_files if str(f) not in excluded_files]

        # Process files if any exist
        if omr_files:
            if not template:
                logger.error(
                    f"Found images but no template in '{curr_dir}'. "
                    f"Place {constants.TEMPLATE_FILENAME} in the appropriate directory."
                )
                raise TemplateNotFoundError(curr_dir)

            output_dir = Path(self.args["output_dir"], curr_dir.relative_to(root_dir))
            template.reset_and_setup_for_directory(output_dir)

            # Process each file
            for idx, file_path in enumerate(omr_files, start=1):
                file_result = self.process_image(
                    file_path, template, tuning_config, evaluation_config
                )
                result.add_result(file_result)

                if progress_callback:
                    progress_callback(idx, len(omr_files))

        # Recursively process subdirectories
        for subdir in subdirs:
            self._process_directory_recursive(
                root_dir,
                subdir,
                result,
                template,
                tuning_config,
                evaluation_config,
                progress_callback,
            )

    def _load_template_and_config(
        self, directory: Path
    ) -> tuple[Template | None, Config]:
        """Load template and config from a directory.

        Searches up the directory tree for template.json and config.json files.
        """
        curr_dir = directory
        template = None
        tuning_config = CONFIG_DEFAULTS

        # Search up the tree for config and template
        while curr_dir != curr_dir.parent:
            config_path = curr_dir.joinpath(constants.CONFIG_FILENAME)
            if config_path.exists() and tuning_config == CONFIG_DEFAULTS:
                tuning_config = open_config_with_defaults(config_path, self.args)

            template_path = curr_dir.joinpath(constants.TEMPLATE_FILENAME)
            if template_path.exists() and template is None:
                template = Template(template_path, tuning_config, self.args)
                break  # Found template, stop searching

            curr_dir = curr_dir.parent

        return template, tuning_config
