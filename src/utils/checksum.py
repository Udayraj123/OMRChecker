import contextlib
import hashlib
from pathlib import Path


def calculate_file_checksum(file_path: Path | str, algorithm: str = "sha256") -> str:
    """Calculate the checksum of a file using the specified hashing algorithm.

    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use (md5, sha1, sha256, sha512)

    Returns:
        Hexadecimal string representation of the file's checksum

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the algorithm is not supported
    """
    file_path = Path(file_path)

    if not file_path.exists():
        msg = f"File not found: {file_path}"
        raise FileNotFoundError(msg)

    # Validate algorithm
    try:
        hasher = hashlib.new(algorithm)
    except ValueError as e:
        msg = f"Unsupported hash algorithm: {algorithm}"
        raise ValueError(msg) from e

    # Read file in chunks to handle large files efficiently
    with file_path.open("rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()


def print_file_checksum(file_path: Path | str, algorithm: str = "md5") -> None:
    """Calculate and print the checksum of a file.

    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use (md5, sha1, sha256, sha512)
    """
    with contextlib.suppress(FileNotFoundError, ValueError):
        calculate_file_checksum(file_path, algorithm)
