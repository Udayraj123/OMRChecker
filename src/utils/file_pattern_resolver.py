"""Utility for resolving file path patterns using OMR field data."""

import re
from pathlib import Path
from typing import Any

from src.utils.logger import logger


class FilePatternResolver:
    """Resolves file path/name patterns using field data.

    This utility can be used by any processor that needs to generate
    dynamic file paths based on OMR detected values or other fields.

    Features:
    - Format patterns with {field} placeholders
    - Auto-preserve original file extensions if not specified
    - Sanitize paths (remove invalid characters)
    - Handle path collisions with configurable strategies

    Usage:
        resolver = FilePatternResolver()
        path = resolver.resolve_pattern(
            "booklet_{code}/{roll}_{score}",
            {"code": "A", "roll": "12345", "score": "95"},
            original_path="/path/to/image.jpg",
            collision_strategy="increment"
        )
        # Result: Path("booklet_A/12345_95.jpg")
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        """Initialize the pattern resolver.

        Args:
            base_dir: Optional base directory for all resolved paths
        """
        self.base_dir = base_dir

    def resolve_pattern(
        self,
        pattern: str,
        fields: dict[str, Any],
        original_path: Path | str | None = None,
        collision_strategy: str = "skip",
    ) -> Path | None:
        """Resolve a file path pattern using provided fields.

        Args:
            pattern: Pattern string with {field} placeholders
                    e.g., "folder_{booklet}/{roll}_{score}"
                    If extension not in pattern, preserves original extension
            fields: Dictionary of field values for substitution
            original_path: Original file path (for extension preservation)
            collision_strategy: How to handle existing files:
                - "skip": Return None if file exists
                - "increment": Append _001, _002, etc.
                - "overwrite": Allow overwriting

        Returns:
            Resolved Path object, or None if collision with "skip" strategy
        """
        try:
            # Format the pattern with fields
            formatted = pattern.format(**fields)

            # Sanitize the path (remove invalid characters)
            sanitized = self._sanitize_path(formatted)

            # Create Path object
            resolved_path = Path(sanitized)

            # Handle extension preservation
            if original_path and not resolved_path.suffix:
                original_ext = Path(original_path).suffix
                resolved_path = resolved_path.with_suffix(original_ext)

            # Apply base directory if set
            if self.base_dir:
                resolved_path = self.base_dir / resolved_path

            # Handle collisions
            return self._handle_collision(resolved_path, collision_strategy)

        except KeyError as e:
            logger.warning(f"Pattern references undefined field: {e}")
            return None
        except Exception as e:
            logger.error(f"Error resolving pattern '{pattern}': {e}")
            return None

    def _sanitize_path(self, path_str: str) -> str:
        """Sanitize path string by removing invalid characters.

        Args:
            path_str: Path string to sanitize

        Returns:
            Sanitized path string
        """
        # Replace invalid filename characters with underscore
        # Invalid: < > : " / \ | ? *
        # Note: We need to preserve / for directory separators
        # Split by / to handle directory parts separately
        parts = path_str.split("/")
        sanitized_parts = []

        for part in parts:
            # Sanitize each path component
            # Remove/replace invalid chars except forward slash
            sanitized = re.sub(r'[<>:"|?*\\]', "_", part)
            # Remove any double underscores
            sanitized = re.sub(r"_+", "_", sanitized)
            # Strip leading/trailing underscores and spaces
            sanitized = sanitized.strip("_ ")
            if sanitized:  # Only add non-empty parts
                sanitized_parts.append(sanitized)

        return "/".join(sanitized_parts)

    def _handle_collision(self, path: Path, strategy: str) -> Path | None:
        """Handle file path collisions based on strategy.

        Args:
            path: The path to check
            strategy: Collision handling strategy

        Returns:
            Final path, or None if skipping collision
        """
        if not path.exists():
            return path

        if strategy == "skip":
            logger.debug(f"File exists, skipping: {path.name}")
            return None

        if strategy == "overwrite":
            logger.debug(f"File exists, will overwrite: {path.name}")
            return path

        if strategy == "increment":
            # Find available incremented filename
            stem = path.stem
            suffix = path.suffix
            parent = path.parent
            counter = 1

            while counter < 9999:
                new_name = f"{stem}_{counter:03d}{suffix}"
                new_path = parent / new_name
                if not new_path.exists():
                    logger.debug(f"File exists, using incremented name: {new_name}")
                    return new_path
                counter += 1

                # Safety limit
                if counter == 9999:
                    logger.error(f"Too many collisions for {stem}, giving up")
                    return None

        logger.warning(f"Unknown collision strategy '{strategy}', skipping")
        return None

    def resolve_batch(
        self,
        patterns_and_fields: list[tuple[str, dict, Path]],
        collision_strategy: str = "skip",
    ) -> list[tuple[Path | None, dict]]:
        """Resolve multiple patterns in batch.

        Args:
            patterns_and_fields: List of (pattern, fields, original_path) tuples
            collision_strategy: Collision handling strategy for all

        Returns:
            List of (resolved_path, fields) tuples
        """
        results = []
        for pattern, fields, original_path in patterns_and_fields:
            resolved = self.resolve_pattern(
                pattern, fields, original_path, collision_strategy
            )
            results.append((resolved, fields))
        return results
