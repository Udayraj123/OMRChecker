"""Shared constants, helpers, and Qt utilities for the OMR template editor."""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PyQt6 import QtGui, QtWidgets

# ---------------------------------------------------------------------------
# Project root (two parents up from src/ui/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
try:
    from src.logger import logger as LOG  # type: ignore
except Exception:

    class _DummyLog:
        def info(self, *a, **k):
            print(*a)

        def warning(self, *a, **k):
            print(*a)

        def error(self, *a, **k):
            print(*a)

    LOG = _DummyLog()  # type: ignore

# ---------------------------------------------------------------------------
# Field types (from src.constants, with fallback)
# ---------------------------------------------------------------------------
FIELD_TYPES: Dict[str, Dict[str, Any]] = {}
try:
    from src.constants import FIELD_TYPES as _FT  # type: ignore

    FIELD_TYPES = dict(_FT)
except Exception:
    FIELD_TYPES = {
        "QTYPE_INT": {
            "bubbleValues": [str(i) for i in range(10)],
            "direction": "vertical",
        },
        "QTYPE_INT_FROM_1": {
            "bubbleValues": [str(i) for i in range(1, 10)] + ["0"],
            "direction": "vertical",
        },
        "QTYPE_MCQ4": {"bubbleValues": ["A", "B", "C", "D"], "direction": "horizontal"},
        "QTYPE_MCQ5": {
            "bubbleValues": ["A", "B", "C", "D", "E"],
            "direction": "horizontal",
        },
    }

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
RECENT_FOLDERS_PATH = Path.home() / ".config" / "omrchecker" / "recent_folders.json"

PROCESSOR_TYPES = [
    "CropPage",
    "CropOnMarkers",
    "FeatureBasedAlignment",
    "GaussianBlur",
    "MedianBlur",
    "Levels",
]

PROCESSOR_DESCRIPTIONS = {
    "CropPage": "Detects paper boundary via edge detection and perspective-crops the sheet. Best for flat, clean scans.",
    "CropOnMarkers": "Finds corner markers (fiducial dots) printed on the sheet and warps it to exactly match the template. Best for sheets printed with omr_marker.jpg at corners.",
    "FeatureBasedAlignment": "Uses ORB keypoint matching to align a photo of the sheet to a reference image. Best for mobile camera photos of complex forms.",
    "GaussianBlur": "Smooths the image to reduce noise. Usually applied before thresholding.",
    "MedianBlur": "Salt-and-pepper noise reduction. Good for scanned sheets with ink specks.",
    "Levels": "Adjusts brightness (low/high) and gamma — brightens dark scans or increases bubble contrast.",
}

# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------


def find_first_image_under(folder: Path) -> Optional[Path]:
    if not folder.exists():
        return None
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            return p
    return None


def find_all_images_under(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_first_template_under(folder: Path) -> Optional[Path]:
    if not folder.exists():
        return None
    for p in sorted(folder.rglob("template.json")):
        if p.is_file():
            return p
    return None


# ---------------------------------------------------------------------------
# String / list helpers
# ---------------------------------------------------------------------------


def parse_csv_or_range(text: str) -> List[str]:
    """Parse 'A,B,C' or 'q1..4' into a list of strings."""
    import re

    t = text.strip()
    if ".." in t and "," not in t:
        m = re.match(r"^([^\d]+)(\d+)\.\.(\d+)$", t)
        if m:
            pref, s, e = m.group(1), int(m.group(2)), int(m.group(3))
            step = 1 if e >= s else -1
            return [f"{pref}{i}" for i in range(s, e + step, step)]
    if t == "":
        return []
    return [x.strip() for x in t.split(",") if x.strip()]


def to_csv(values: List[str]) -> str:
    return ",".join(values)


# ---------------------------------------------------------------------------
# Image / Qt helpers
# ---------------------------------------------------------------------------


def load_image_as_pixmap(img_path: Path) -> QtGui.QPixmap:
    img = QtGui.QImage(str(img_path))
    if img.isNull():
        raise RuntimeError(f"Cannot load image: {img_path}")
    return QtGui.QPixmap.fromImage(img)


def np_to_qpixmap(img: np.ndarray) -> QtGui.QPixmap:
    """Convert grayscale or BGR numpy array to QPixmap."""
    h, w = img.shape[:2]
    if len(img.shape) == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(
            img_rgb.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888
        )
    else:
        qimg = QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)
    return QtGui.QPixmap.fromImage(qimg.copy())


# ---------------------------------------------------------------------------
# Recent folders (persisted to ~/.config/omrchecker/recent_folders.json)
# ---------------------------------------------------------------------------


def get_recent_folders() -> List[Path]:
    try:
        if RECENT_FOLDERS_PATH.exists():
            data = json.loads(RECENT_FOLDERS_PATH.read_text(encoding="utf-8"))
            return [Path(p) for p in data if Path(p).exists()]
    except Exception:
        pass
    return []


def add_recent_folder(folder: Path) -> None:
    try:
        RECENT_FOLDERS_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing = [
            str(p) for p in get_recent_folders() if p.resolve() != folder.resolve()
        ]
        recent = [str(folder.resolve())] + existing
        RECENT_FOLDERS_PATH.write_text(json.dumps(recent[:10]), encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Dark palette helper
# ---------------------------------------------------------------------------


def apply_dark_palette(app: QtWidgets.QApplication) -> None:
    p = app.palette()
    p.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(30, 30, 30))
    p.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(37, 37, 37))
    p.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(45, 45, 45))
    p.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(230, 230, 230))
    p.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(230, 230, 230))
    p.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(53, 53, 53))
    p.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(230, 230, 230))
    p.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(42, 130, 218))
    p.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(255, 255, 255))
    app.setPalette(p)
