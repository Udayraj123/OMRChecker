from pathlib import Path, PurePosixPath, PureWindowsPath

from src.utils.file import to_csv_str


def test_to_csv_str_accepts_windows_path():
    """Windows paths must be normalized to forward slashes regardless of host OS."""
    path = PureWindowsPath(r"samples\answer-key\using-csv\adrian_omr.png")
    assert to_csv_str(path) == "samples/answer-key/using-csv/adrian_omr.png"


def test_to_csv_str_accepts_posix_path():
    """POSIX paths must pass through unchanged."""
    path = PurePosixPath("samples/answer-key/using-csv/adrian_omr.png")
    assert to_csv_str(path) == "samples/answer-key/using-csv/adrian_omr.png"


def test_to_csv_str_accepts_host_path():
    """A host-native Path must be normalized to forward slashes."""
    path = Path("outputs") / "CheckedOMRs" / "adrian_omr.png"
    assert to_csv_str(path) == "outputs/CheckedOMRs/adrian_omr.png"


def test_to_csv_str_accepts_string():
    """
    Plain string paths must be normalized on every host OS.

    The naive ``Path(p).as_posix()`` implementation parses the string with the
    host's path rules, so on a POSIX host a Windows-style string keeps its
    backslashes and this assertion fails.
    """
    assert to_csv_str(r"outputs\Manual\ErrorFiles.csv") == "outputs/Manual/ErrorFiles.csv"


def test_to_csv_str_keeps_posix_absolute_string():
    """An already POSIX string must not be rewritten or otherwise mangled."""
    assert to_csv_str("/home/user/outputs/Results.csv") == "/home/user/outputs/Results.csv"


def test_to_csv_str_handles_mixed_separators_in_string():
    """Mixed separators in one string must converge to a single style."""
    assert to_csv_str(r"outputs\Manual/MultiMarkedFiles.csv") == "outputs/Manual/MultiMarkedFiles.csv"
