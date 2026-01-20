"""Repository for managing detection results.

Replaces nested dictionary-based aggregate management with clean repository pattern.
"""

from src.processors.detection.models.detection_results import (
    BarcodeFieldDetectionResult,
    BubbleFieldDetectionResult,
    BubbleMeanValue,
    FileDetectionResults,
    OCRFieldDetectionResult,
)


class DetectionRepository:
    """Repository for managing detection results at different levels.

    Replaces the complex nested dictionary structure in FilePassAggregates
    with a clean, queryable interface.
    """

    def __init__(self) -> None:
        """Initialize empty repository."""
        self._current_file_results: FileDetectionResults | None = None
        self._file_results: dict[str, FileDetectionResults] = {}
        self._directory_path: str | None = None

    # File-level operations
    def initialize_file(self, file_path: str) -> None:
        """Initialize a new file for detection results.

        Args:
            file_path: Path to the file being processed
        """
        # Normalize file_path to string for consistent storage
        file_path_str = str(file_path)
        self._current_file_results = FileDetectionResults(file_path=file_path_str)

    def finalize_file(self) -> None:
        """Finalize current file and store results."""
        if self._current_file_results is not None:
            self._file_results[self._current_file_results.file_path] = (
                self._current_file_results
            )

    def get_current_file_results(self) -> FileDetectionResults:
        """Get results for current file being processed.

        Returns:
            FileDetectionResults for current file

        Raises:
            RuntimeError: If no file is currently being processed
        """
        if self._current_file_results is None:
            msg = "No file currently being processed"
            raise RuntimeError(msg)
        return self._current_file_results

    def get_file_results(self, file_path: str) -> FileDetectionResults:
        """Get results for a specific file.

        Args:
            file_path: Path to the file

        Returns:
            FileDetectionResults for the file

        Raises:
            KeyError: If file has not been processed
        """
        # Normalize file_path to string for consistent lookup
        file_path_str = str(file_path)

        # Try exact match first
        if file_path_str in self._file_results:
            return self._file_results[file_path_str]

        # Try to find by matching stored paths (handle Path vs str differences)
        for stored_path, results in self._file_results.items():
            if str(stored_path) == file_path_str or stored_path == file_path_str:
                return results

        # If still not found, raise error with helpful message
        available_paths = list(self._file_results.keys())
        msg = (
            f"No results found for file: {file_path_str}. "
            f"Available files: {available_paths}"
        )
        raise KeyError(msg)

    # Field-level operations
    def save_bubble_field(
        self, field_id: str, result: BubbleFieldDetectionResult
    ) -> None:
        """Save bubble field detection result.

        Args:
            field_id: Unique field identifier
            result: Detection result for the field
        """
        current = self.get_current_file_results()
        current.bubble_fields[field_id] = result

    def save_ocr_field(self, field_id: str, result: OCRFieldDetectionResult) -> None:
        """Save OCR field detection result.

        Args:
            field_id: Unique field identifier
            result: Detection result for the field
        """
        current = self.get_current_file_results()
        current.ocr_fields[field_id] = result

    def save_barcode_field(
        self, field_id: str, result: BarcodeFieldDetectionResult
    ) -> None:
        """Save barcode field detection result.

        Args:
            field_id: Unique field identifier
            result: Detection result for the field
        """
        current = self.get_current_file_results()
        current.barcode_fields[field_id] = result

    def get_bubble_field(self, field_id: str) -> BubbleFieldDetectionResult:
        """Get bubble field result from current file.

        Args:
            field_id: Field identifier

        Returns:
            BubbleFieldDetectionResult

        Raises:
            KeyError: If field not found
        """
        current = self.get_current_file_results()
        if field_id not in current.bubble_fields:
            msg = f"Bubble field not found: {field_id}"
            raise KeyError(msg)
        return current.bubble_fields[field_id]

    def get_ocr_field(self, field_id: str) -> OCRFieldDetectionResult:
        """Get OCR field result from current file.

        Args:
            field_id: Field identifier

        Returns:
            OCRFieldDetectionResult

        Raises:
            KeyError: If field not found
        """
        current = self.get_current_file_results()
        if field_id not in current.ocr_fields:
            msg = f"OCR field not found: {field_id}"
            raise KeyError(msg)
        return current.ocr_fields[field_id]

    def get_barcode_field(self, field_id: str) -> BarcodeFieldDetectionResult:
        """Get barcode field result from current file.

        Args:
            field_id: Field identifier

        Returns:
            BarcodeFieldDetectionResult

        Raises:
            KeyError: If field not found
        """
        current = self.get_current_file_results()
        if field_id not in current.barcode_fields:
            msg = f"Barcode field not found: {field_id}"
            raise KeyError(msg)
        return current.barcode_fields[field_id]

    # Query operations
    def get_all_bubble_means_for_current_file(self) -> list[BubbleMeanValue]:
        """Get all bubble means across all bubble fields in current file.

        Returns:
            List of all BubbleMeanValue objects

        Replaces nested dictionary access:
        all_bubble_means = [
            mean
            for field_agg in file_level_aggregates["field_label_wise_aggregates"].values()
            for mean in field_agg.get("field_bubble_means", [])
        ]
        """
        current = self.get_current_file_results()
        return current.all_bubble_means

    def get_all_bubble_mean_values_for_current_file(self) -> list[float]:
        """Get all bubble mean values as floats for current file.

        Returns:
            List of mean values
        """
        current = self.get_current_file_results()
        return current.all_bubble_mean_values

    def get_all_bubble_fields_for_current_file(
        self,
    ) -> dict[str, BubbleFieldDetectionResult]:
        """Get all bubble field results for current file.

        Returns:
            Dictionary mapping field_id to BubbleFieldDetectionResult
        """
        current = self.get_current_file_results()
        return current.bubble_fields

    # Directory-level operations
    def initialize_directory(self, directory_path: str) -> None:
        """Initialize repository for a directory.

        Args:
            directory_path: Path to the directory being processed
        """
        self._directory_path = directory_path
        self._file_results.clear()
        self._current_file_results = None

    def get_all_file_results(self) -> dict[str, FileDetectionResults]:
        """Get results for all processed files in directory.

        Returns:
            Dictionary mapping file_path to FileDetectionResults
        """
        return self._file_results.copy()

    def clear(self) -> None:
        """Clear all stored results."""
        self._current_file_results = None
        self._file_results.clear()
        self._directory_path = None

    # Statistics
    def get_total_files_processed(self) -> int:
        """Get total number of files processed."""
        return len(self._file_results)

    def get_total_fields_in_current_file(self) -> int:
        """Get total number of fields in current file."""
        current = self.get_current_file_results()
        return current.num_fields

    def __repr__(self) -> str:
        """Readable representation."""
        return (
            f"DetectionRepository("
            f"directory={self._directory_path}, "
            f"files={len(self._file_results)}, "
            f"current_file={self._current_file_results.file_path if self._current_file_results else None})"
        )
