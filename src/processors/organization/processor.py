"""File organization processor for grouping OMR files."""

import os
import re
import shutil
from pathlib import Path
from threading import Lock

from src.processors.base import ProcessingContext, Processor
from src.schemas.models.config import GroupingRule
from src.utils.file_pattern_resolver import FilePatternResolver
from src.utils.logger import logger


class FileOrganizerProcessor(Processor):
    """Processor that organizes files into folders with dynamic patterns.

    CONCURRENCY-SAFE: Runs AFTER all parallel processing completes.
    Creates symlinks/copies with dynamic names using FilePatternResolver.

    This processor:
    1. Collects results during parallel processing (thread-safe)
    2. Runs in finish_processing_directory() after all files processed
    3. Uses FilePatternResolver to format dynamic file paths
    4. Creates symlinks or copies with collision handling
    5. Maintains original CheckedOMRs/ structure intact

    Example patterns:
    - "booklet_{code}/roll_{roll}.jpg" → "booklet_A/roll_12345.jpg"
    - "scores/{score_bucket}/{name}" → "scores/90-100/John_Doe.jpg" (extension preserved)
    - "failed/{batch}/student_{id}.png" → "failed/morning/student_456.png"
    """

    def __init__(self, file_grouping_config, output_dir: Path) -> None:
        """Initialize the FileOrganizer processor.

        Args:
            file_grouping_config: FileGroupingConfig with rules
            output_dir: Base output directory (creates organized/ subdirectory)
        """
        self.config = file_grouping_config
        self.output_dir = output_dir
        self.organized_dir = output_dir / "organized"
        self.results = []  # Store results from each file processing
        self.file_operations = []  # Track all operations
        self._lock = Lock()  # For thread-safe result collection

        # Initialize pattern resolver
        self.pattern_resolver = FilePatternResolver(base_dir=self.organized_dir)

        # Sort rules by priority (lower number = higher priority)
        if self.config.enabled and self.config.rules:
            self.config.rules.sort(key=lambda r: r.priority)

    def get_name(self) -> str:
        """Get the name of this processor."""
        return "FileOrganizer"

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Collect processing results for later organization.

        NOTE: Does NOT organize files here - stores context for batch processing.

        Args:
            context: Processing context with all detected data

        Returns:
            Unchanged context
        """
        if not self.config.enabled:
            return context

        # Thread-safe collection of results
        with self._lock:
            self.results.append(
                {
                    "context": context,
                    "output_path": context.metadata.get("output_path"),
                    "score": context.score if hasattr(context, "score") else 0,
                    "omr_response": context.omr_response.copy(),
                    "is_multi_marked": context.is_multi_marked,
                }
            )

        return context

    def finish_processing_directory(self):
        """Organize all files sequentially after processing completes.

        This runs AFTER all parallel processing is done, eliminating race conditions.
        """
        if not self.config.enabled or not self.results:
            return

        logger.info(f"\n{'=' * 60}")
        logger.info("Starting file organization...")
        logger.info(f"{'=' * 60}")

        # Create organized base directory
        self.organized_dir.mkdir(parents=True, exist_ok=True)

        # Process each result sequentially (no concurrency issues!)
        for result in self.results:
            self._organize_single_file(result)

        # Print summary
        self._print_summary()

    def _organize_single_file(self, result: dict) -> None:
        """Organize a single file using pattern resolver."""
        context = result["context"]
        output_path = result.get("output_path")

        if not output_path or not Path(output_path).exists():
            logger.warning(f"Output file not found, skipping: {output_path}")
            return

        # Build formatting fields
        formatting_fields = {
            "file_path": str(context.file_path),
            "file_name": Path(context.file_path).name,
            "file_stem": Path(context.file_path).stem,
            "original_name": Path(output_path).name,
            "score": str(result["score"]),
            "is_multi_marked": str(result["is_multi_marked"]),
            **result["omr_response"],
        }

        # Find matching rule
        matched_rule = self._find_matching_rule(formatting_fields)

        if matched_rule:
            pattern = matched_rule.destination_pattern
            action = matched_rule.action
            collision_strategy = matched_rule.collision_strategy
            rule_name = matched_rule.name
        else:
            pattern = self.config.default_pattern
            action = "symlink"
            collision_strategy = "skip"
            rule_name = "default"

        # Resolve pattern to path
        source_path = Path(output_path)
        dest_path = self.pattern_resolver.resolve_pattern(
            pattern=pattern,
            fields=formatting_fields,
            original_path=source_path,
            collision_strategy=collision_strategy,
        )

        # If collision with skip strategy, dest_path will be None
        if dest_path is None:
            logger.warning(
                f"Skipping due to collision or pattern error: {source_path.name}"
            )
            self.file_operations.append(
                {
                    "source": str(source_path),
                    "destination": "N/A",
                    "action": "skipped",
                    "rule": rule_name,
                    "reason": "collision or pattern error",
                }
            )
            return

        # Create parent directories
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Perform operation
        try:
            if action == "symlink":
                # Create relative symlink if possible
                try:
                    rel_source = os.path.relpath(source_path, dest_path.parent)
                    dest_path.symlink_to(rel_source)
                except (OSError, NotImplementedError):
                    # Fallback to absolute symlink or copy on Windows
                    if os.name == "nt":
                        shutil.copy2(str(source_path), str(dest_path))
                        action = "copy"
                    else:
                        dest_path.symlink_to(source_path.absolute())

            elif action == "copy":
                shutil.copy2(str(source_path), str(dest_path))

            # Get relative path for logging
            rel_dest = dest_path.relative_to(self.organized_dir)
            logger.debug(f"{action.capitalize()}ed to '{rel_dest}': {source_path.name}")

            self.file_operations.append(
                {
                    "source": str(source_path),
                    "destination": str(dest_path),
                    "action": action,
                    "rule": rule_name,
                }
            )

        except Exception as e:
            logger.error(f"Failed to {action} {source_path.name}: {e}")
            self.file_operations.append(
                {
                    "source": str(source_path),
                    "destination": str(dest_path) if dest_path else "N/A",
                    "action": "failed",
                    "rule": rule_name,
                    "error": str(e),
                }
            )

    def _find_matching_rule(self, formatting_fields: dict) -> GroupingRule | None:
        """Find the first matching rule based on priority."""
        for rule in self.config.rules:
            try:
                format_string = rule.matcher["format_string"]
                match_regex = rule.matcher["match_regex"]

                formatted_string = format_string.format(**formatting_fields)

                if re.search(match_regex, formatted_string):
                    return rule

            except KeyError as e:
                logger.warning(f"Rule '{rule.name}' references undefined field: {e}")
            except Exception as e:
                logger.warning(f"Error evaluating rule '{rule.name}': {e}")

        return None

    def _print_summary(self) -> None:
        """Print organization summary."""
        logger.info(f"\n{'=' * 60}")
        logger.info("File Organization Summary")
        logger.info(f"{'=' * 60}")
        logger.info(f"Total files processed: {len(self.file_operations)}")

        # Group by rule and action
        by_rule = {}
        skipped = 0
        failed = 0

        for op in self.file_operations:
            rule = op["rule"]
            action = op["action"]

            if action == "skipped":
                skipped += 1
            elif action == "failed":
                failed += 1
            else:
                by_rule.setdefault(rule, []).append(op)

        for rule_name, ops in by_rule.items():
            logger.info(f"  {rule_name}: {len(ops)} files")

        if skipped > 0:
            logger.warning(f"  Skipped (collisions/errors): {skipped} files")
        if failed > 0:
            logger.error(f"  Failed: {failed} files")

        logger.info(f"\nOrganized files location: {self.organized_dir}")
        logger.info(f"{'=' * 60}\n")
