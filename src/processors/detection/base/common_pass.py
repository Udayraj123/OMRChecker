import threading
from threading import local
from typing import Any

from src.processors.layout.field.base import Field
from src.utils.stats import StatsByLabel


class FilePassAggregates:
    """Interface defining methods for collecting and managing data during file processing at various levels (field, file, directory).

    Thread-safe: Maintains separate state per file_path using a dictionary with locks.
    Each file gets its own file-level and field-level aggregates, avoiding conflicts in parallel processing.

    Uses a simple approach:
    - Per-file state stored in dict[file_path, state] with a lock
    - Thread-local storage for current file_path (just a string, minimal overhead)
    - Simple lock for directory-level aggregates
    """

    def __init__(self, tuning_config) -> None:
        self.tuning_config = tuning_config
        # Per-file state: file_path -> {file_level_aggregates, field_level_aggregates}
        self._file_states: dict[str, dict[str, Any]] = {}
        self._file_states_lock = threading.Lock()
        # Thread-local storage for current file_path (just a string, minimal overhead)
        self._thread_local = local()
        # Shared directory-level aggregates (protected by lock)
        self._directory_level_aggregates: dict | None = None
        self._directory_lock = threading.Lock()

    def initialize_directory_level_aggregates(self, initial_directory_path) -> None:
        """Initialize directory-level aggregates (shared across all threads)."""
        with self._directory_lock:
            self._directory_level_aggregates = {
                "initial_directory_path": initial_directory_path,
                "file_wise_aggregates": {},
                "files_count": StatsByLabel("processed"),
            }

    def get_directory_level_aggregates(self):
        """Get directory-level aggregates (thread-safe read).

        Note: Returns a reference to the aggregates dict. For thread-safe writes,
        access the dict directly within a lock context (see update_aggregates_on_processed_file
        for an example).
        """
        with self._directory_lock:
            return self._directory_level_aggregates

    def insert_directory_level_aggregates(
        self, next_directory_level_aggregates
    ) -> None:
        """Insert directory-level aggregates (thread-safe write)."""
        with self._directory_lock:
            self._directory_level_aggregates = {
                **self._directory_level_aggregates,
                **next_directory_level_aggregates,
            }

    def initialize_file_level_aggregates(self, file_path) -> None:
        """Initialize file-level aggregates for a specific file."""
        file_path_str = str(file_path)
        # Store current file_path in thread-local (just a string, minimal overhead)
        self._thread_local.current_file_path = file_path_str

        with self._file_states_lock:
            # Always a new file, so create new entry
            self._file_states[file_path_str] = {
                "file_level_aggregates": {
                    "file_path": file_path_str,
                    "fields_count": StatsByLabel("processed"),
                    # field_label_wise_aggregates removed - all field types now use repository
                }
            }

    def get_file_level_aggregates(self):
        """Get file-level aggregates for the current file (from thread-local file_path)."""
        file_path_str = getattr(self._thread_local, "current_file_path", None)
        if file_path_str is None:
            raise RuntimeError(
                "No current file_path set. "
                "Call initialize_file_level_aggregates() first."
            )

        with self._file_states_lock:
            if file_path_str not in self._file_states:
                raise RuntimeError(
                    f"File level aggregates not initialized for {file_path_str}. "
                    "Call initialize_file_level_aggregates() first."
                )
            return self._file_states[file_path_str]["file_level_aggregates"]

    def insert_file_level_aggregates(self, next_file_level_aggregates) -> None:
        """Insert file-level aggregates for the current file."""
        file_level_aggregates = self.get_file_level_aggregates()
        file_level_aggregates.update(next_file_level_aggregates)

    def update_aggregates_on_processed_file(self, file_path) -> None:
        """Update directory-level aggregates with file results (thread-safe)."""
        file_path_str = str(file_path)
        # Get file-level aggregates from the per-file state dict
        # initialize_file_level_aggregates() is always called before this, so it's guaranteed to exist
        with self._file_states_lock:
            file_level_aggregates = self._file_states[file_path_str][
                "file_level_aggregates"
            ]

        with self._directory_lock:
            # Thread-safe write to shared directory-level aggregates
            self._directory_level_aggregates["file_wise_aggregates"][file_path_str] = (
                file_level_aggregates
            )
            self._directory_level_aggregates["files_count"].push("processed")

    def initialize_field_level_aggregates(self, field: Field) -> None:
        """Initialize field-level aggregates for the current file."""
        file_path_str = getattr(self._thread_local, "current_file_path", None)
        if file_path_str is None:
            raise RuntimeError(
                "No current file_path set. "
                "Call initialize_file_level_aggregates() first."
            )

        with self._file_states_lock:
            # initialize_file_level_aggregates() is always called before field processing,
            # so file-level aggregates are guaranteed to exist
            self._file_states[file_path_str]["field_level_aggregates"] = {
                "field": field,
            }

    def get_field_level_aggregates(self):
        """Get field-level aggregates for the current file."""
        file_path_str = getattr(self._thread_local, "current_file_path", None)
        if file_path_str is None:
            raise RuntimeError(
                "No current file_path set. "
                "Call initialize_file_level_aggregates() first."
            )

        with self._file_states_lock:
            if file_path_str not in self._file_states:
                raise RuntimeError(
                    f"File level aggregates not initialized for {file_path_str}. "
                    "Call initialize_file_level_aggregates() first."
                )
            if "field_level_aggregates" not in self._file_states[file_path_str]:
                raise RuntimeError(
                    f"Field level aggregates not initialized for {file_path_str}. "
                    "Call initialize_field_level_aggregates() first."
                )
            return self._file_states[file_path_str]["field_level_aggregates"]

    def insert_field_level_aggregates(self, next_field_level_aggregates) -> None:
        """Insert field-level aggregates for the current file."""
        field_level_aggregates = self.get_field_level_aggregates()
        field_level_aggregates.update(next_field_level_aggregates)

    # To be called by the child classes as per consumer needs
    def update_field_level_aggregates_on_processed_field(self, field: Field) -> None:
        pass

    # To be called by the child classes as per consumer needs
    def update_file_level_aggregates_on_processed_field(
        self, field: Field, field_level_aggregates
    ) -> None:
        """Update file-level aggregates on processed field."""
        file_level_aggregates = self.get_file_level_aggregates()
        # Just update fields_count for statistics
        file_level_aggregates["fields_count"].push("processed")

    def update_directory_level_aggregates_on_processed_field(
        self, field: Field, field_level_aggregates
    ) -> None:
        pass
