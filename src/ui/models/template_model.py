"""TemplateModel — in-memory template.json with undo/redo and full settings access."""
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from PyQt6 import QtCore

from ..common import FIELD_TYPES, LOG, parse_csv_or_range


class TemplateModel(QtCore.QObject):
    """
    Mutable in-memory representation of a template.json file.

    Derived-default rules (render-time only, not persisted):
      • direction  → derived from fieldType if absent
      • bubbleValues → derived from fieldType via FIELD_TYPES if absent
      • bubbleDimensions per-block → falls back to template-global if absent

    Persistence rules (on save / save_as_edited):
      • Omit 'direction' when it equals the type default
      • Omit 'bubbleValues' when it equals the type default
    """

    changed = QtCore.pyqtSignal()

    def __init__(
        self,
        template_path: Path,
        image_path: Optional[Path] = None,
    ):
        super().__init__()
        self.template_path = Path(template_path)
        self.image_path = Path(image_path) if image_path else None
        self.template: Dict[str, Any] = self._load_or_empty(self.template_path)
        self._ensure_defaults()
        self._history: List[Dict[str, Any]] = []
        self._future: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_or_empty(self, p: Path) -> Dict[str, Any]:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                LOG.error(f"Failed to load template '{p}': {e}")
        return {}

    def _ensure_defaults(self) -> None:
        self.template.setdefault("fieldBlocks", {})
        self.template.setdefault("bubbleDimensions", [32, 32])
        self.template.setdefault("pageDimensions", self._infer_page_dims())
        self.template.setdefault("preProcessors", [])

    def _infer_page_dims(self) -> List[int]:
        if self.image_path and self.image_path.exists():
            try:
                img = cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    h, w = img.shape[:2]
                    return [int(w), int(h)]
            except Exception:
                pass
        return [0, 0]

    # ------------------------------------------------------------------
    # Load / reload
    # ------------------------------------------------------------------

    def load_from_path(
        self, template_path: Path, image_path: Optional[Path] = None
    ) -> None:
        """Reload model from a different template file (clears history)."""
        self.template_path = Path(template_path)
        if image_path:
            self.image_path = Path(image_path)
        self.template = self._load_or_empty(self.template_path)
        self._ensure_defaults()
        self._history.clear()
        self._future.clear()
        self.changed.emit()

    def new_template(self, image_path: Optional[Path] = None) -> None:
        """Reset to a blank template (clears history)."""
        if image_path:
            self.image_path = Path(image_path)
        w, h = 0, 0
        if self.image_path and self.image_path.exists():
            try:
                img = cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    h, w = img.shape[:2]
            except Exception:
                pass
        self.template = {
            "pageDimensions": [int(w), int(h)],
            "bubbleDimensions": [32, 32],
            "preProcessors": [],
            "fieldBlocks": {},
        }
        self._history.clear()
        self._future.clear()
        self.changed.emit()

    # ------------------------------------------------------------------
    # Undo / redo
    # ------------------------------------------------------------------

    def push_state(self, reason: str = "") -> None:
        self._history.append(copy.deepcopy(self.template))
        self._future.clear()

    def undo(self) -> bool:
        if not self._history:
            return False
        self._future.append(copy.deepcopy(self.template))
        self.template = self._history.pop()
        self.changed.emit()
        return True

    def redo(self) -> bool:
        if not self._future:
            return False
        self._history.append(copy.deepcopy(self.template))
        self.template = self._future.pop()
        self.changed.emit()
        return True

    @property
    def is_dirty(self) -> bool:
        return bool(self._history)

    # ------------------------------------------------------------------
    # Field-block access
    # ------------------------------------------------------------------

    def field_blocks(self):
        return list(self.template.get("fieldBlocks", {}).items())

    @staticmethod
    def default_dir_for_type(field_type: Optional[str]) -> str:
        if field_type in ("QTYPE_INT", "QTYPE_INT_FROM_1"):
            return "vertical"
        return "horizontal"

    def get_block(self, name: str) -> Dict[str, Any]:
        """Render view with derived defaults. Do NOT persist this dict."""
        base = self.template["fieldBlocks"].get(name, {})
        fb = dict(base)
        ft = base.get("fieldType")
        if "direction" not in fb or fb.get("direction") is None:
            fb["direction"] = self.default_dir_for_type(ft)
        if not isinstance(fb.get("bubbleDimensions"), list):
            fb["bubbleDimensions"] = list(
                self.template.get("bubbleDimensions", [32, 32])
            )
        if not fb.get("bubbleValues") and ft in FIELD_TYPES:
            fb["bubbleValues"] = list(FIELD_TYPES[ft].get("bubbleValues", []))
        # Expand range-notation labels (e.g. ["roll1..9"] → ["roll1",…,"roll9"])
        # so that dimension calculations use the true label count.
        raw_labels = fb.get("fieldLabels", [])
        if isinstance(raw_labels, list):
            expanded: List[str] = []
            for lbl in raw_labels:
                s = str(lbl)
                expanded.extend(parse_csv_or_range(s) if ".." in s else [s])
            fb["fieldLabels"] = expanded
        return fb

    def get_block_base(self, name: str) -> Dict[str, Any]:
        """Direct reference to the stored (persistent) block dict."""
        return self.template["fieldBlocks"].setdefault(name, {})

    def next_block_name(self) -> str:
        base = "FieldBlock"
        n = 1
        existing = set(self.template["fieldBlocks"].keys())
        while f"{base}_{n}" in existing:
            n += 1
        return f"{base}_{n}"

    def add_block(self, name: str, rect: "QtCore.QRectF") -> None:
        self.push_state("add_block")
        x, y = int(rect.left()), int(rect.top())
        w, h = max(30, int(rect.width())), max(30, int(rect.height()))
        self.template["fieldBlocks"][name] = {
            "origin": [x, y],
            "bubblesGap": w,
            "labelsGap": h,
            "fieldLabels": ["q1..1"],
        }
        self.changed.emit()

    def remove_block(self, name: str) -> None:
        if name in self.template["fieldBlocks"]:
            self.push_state("remove_block")
            del self.template["fieldBlocks"][name]
            self.changed.emit()

    # ------------------------------------------------------------------
    # Global settings
    # ------------------------------------------------------------------

    def get_page_dimensions(self) -> List[int]:
        return list(self.template.get("pageDimensions", [0, 0]))

    def set_page_dimensions(self, w: int, h: int) -> None:
        self.push_state("page_dims")
        self.template["pageDimensions"] = [w, h]
        self.changed.emit()

    def get_bubble_dimensions(self) -> List[int]:
        return list(self.template.get("bubbleDimensions", [32, 32]))

    def set_bubble_dimensions(self, w: int, h: int) -> None:
        self.push_state("bubble_dims")
        self.template["bubbleDimensions"] = [w, h]
        self.changed.emit()

    def get_preprocessors(self) -> List[Dict[str, Any]]:
        return list(self.template.get("preProcessors", []))

    def set_preprocessors(self, procs: List[Dict[str, Any]]) -> None:
        self.push_state("set_preprocessors")
        self.template["preProcessors"] = copy.deepcopy(procs)
        self.changed.emit()

    def get_custom_labels(self) -> Dict[str, Any]:
        return dict(self.template.get("customLabels", {}))

    def set_custom_labels(self, labels: Dict[str, Any]) -> None:
        self.push_state("custom_labels")
        if labels:
            self.template["customLabels"] = labels
        else:
            self.template.pop("customLabels", None)
        self.changed.emit()

    def get_output_columns(self) -> List[str]:
        return list(self.template.get("outputColumns", []))

    def set_output_columns(self, cols: List[str]) -> None:
        self.push_state("output_cols")
        if cols:
            self.template["outputColumns"] = cols
        else:
            self.template.pop("outputColumns", None)
        self.changed.emit()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _build_clean_template(self) -> Dict[str, Any]:
        cleaned = copy.deepcopy(self.template)
        for _name, fb in cleaned.get("fieldBlocks", {}).items():
            ft = fb.get("fieldType")
            if "direction" in fb:
                if fb["direction"] == self.default_dir_for_type(ft):
                    fb.pop("direction", None)
            if "bubbleValues" in fb and ft in FIELD_TYPES:
                if fb["bubbleValues"] == FIELD_TYPES[ft].get("bubbleValues", []):
                    fb.pop("bubbleValues", None)
        return cleaned

    def save(self, path: Optional[Path] = None) -> Path:
        """Save to path (defaults to original template_path)."""
        out = Path(path) if path else self.template_path
        cleaned = self._build_clean_template()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
        self.template_path = out
        return out

    def save_as_edited(self) -> Path:
        """Legacy: save to <name>.edited.json next to template."""
        out = self.template_path.with_name(self.template_path.stem + ".edited.json")
        return self.save(out)
