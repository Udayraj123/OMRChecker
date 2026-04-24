import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
# New imports for preprocessing
import cv2
import numpy as np
import copy

from PyQt6 import QtCore, QtGui, QtWidgets

# Try importing project modules; fall back to simple defaults if not available.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Logger from repo (for explicit error messages)
try:
    from src.logger import logger as LOG  # type: ignore
except Exception:
    class _DummyLog:
        def info(self,*a,**k): print(*a)
        def warning(self,*a,**k): print(*a)
        def error(self,*a,**k): print(*a)
    LOG = _DummyLog()  # type: ignore

FIELD_TYPES: Dict[str, Dict[str, Any]] = {}
try:
    from src.constants import FIELD_TYPES as _FT  # type: ignore
    FIELD_TYPES = dict(_FT)
except Exception:
    FIELD_TYPES = {
        "QTYPE_INT": {"bubbleValues": [str(i) for i in range(10)], "direction": "vertical"},
        "QTYPE_INT_FROM_1": {"bubbleValues": [str(i) for i in range(1, 10)] + ["0"], "direction": "vertical"},
        "QTYPE_MCQ4": {"bubbleValues": ["A", "B", "C", "D"], "direction": "horizontal"},
        "QTYPE_MCQ5": {"bubbleValues": ["A", "B", "C", "D", "E"], "direction": "horizontal"},
    }

def load_image_as_pixmap(img_path: Path) -> QtGui.QPixmap:
    img = QtGui.QImage(str(img_path))
    if img.isNull():
        raise SystemExit(f"Cannot load image: {img_path}")
    return QtGui.QPixmap.fromImage(img)

def find_first_image_under(inputs_dir: Path) -> Optional[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if not inputs_dir.exists():
        return None
    for p in sorted(inputs_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            return p
    return None

def find_first_template_under(inputs_dir: Path) -> Optional[Path]:
    """Find the first template.json under the given inputs dir (recursively)."""
    if not inputs_dir.exists():
        return None
    for p in sorted(inputs_dir.rglob("template.json")):
        if p.is_file():
            return p
    return None

def parse_csv_or_range(text: str) -> List[str]:
    # e.g., "A,B,C" or "q1..4" -> ["q1","q2","q3","q4"] (kept simple for labels)
    t = text.strip()
    if ".." in t and "," not in t:
        # very simple range expansion: prefix + start..end (numbers only)
        # Example: q1..4 -> q1, q2, q3, q4
        import re
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

# New: import defaults and pipeline (EXACT same modules as main pipeline)
try:
    from src.defaults.config import CONFIG_DEFAULTS  # EXACT import as main
    from src.utils.parsing import open_config_with_defaults  # EXACT import as main
    from src.template import Template  # EXACT import as main
    from src.utils.image import ImageUtils  # EXACT import as main
    PIPELINE_IMPORT_ERROR: Optional[str] = None
except Exception as e:
    # Do NOT silently continue: record why pipeline is unavailable so we can show a clear error
    PIPELINE_IMPORT_ERROR = f"{type(e).__name__}: {e}"
    LOG.error(
        "Qt editor failed to import pipeline modules. "
        f"Reason: {PIPELINE_IMPORT_ERROR}\n"
        f"Tip: run from the repo root (python -m src.ui.qt_editor ...) and ensure dependencies are installed."
    )
    CONFIG_DEFAULTS = None  # type: ignore
    open_config_with_defaults = None  # type: ignore
    Template = None  # type: ignore
    ImageUtils = None  # type: ignore

# ---------------- In-memory template model with defaults, undo/redo ----------------
class TemplateModel(QtCore.QObject):
    """
    Keeps a working copy of template.json during editing.
    - Derived defaults (render-time only, not persisted):
        * direction default by fieldType:
            - QTYPE_MCQ4/QTYPE_MCQ5 -> horizontal
            - QTYPE_INT/QTYPE_INT_FROM_1 -> vertical
            - otherwise -> horizontal
        * bubbleValues derived from fieldType (constants.FIELD_TYPES) if absent.
        * bubbleDimensions defaults to global template bubbleDimensions if absent.
    - Persistence rules on save:
        * Do NOT persist 'direction' if it matches the type-default above.
        * Do NOT persist 'bubbleValues' when it matches type defaults or was never explicitly set.
    - Undo/Redo stacks for create/delete/move/resize/panel edits.
    """
    changed = QtCore.pyqtSignal()

    def __init__(self, template_path: Path, image_path: Path):
        super().__init__()
        self.template_path = Path(template_path)
        self.image_path = Path(image_path)
        self.template: Dict[str, Any] = self._load_template(self.template_path)
        # Ensure required keys
        self.template.setdefault("fieldBlocks", {})
        self.template.setdefault("bubbleDimensions", [20, 20])
        self.template.setdefault("pageDimensions", self._infer_page_dims())
        # Undo/redo stacks
        self._history: List[Dict[str, Any]] = []
        self._future: List[Dict[str, Any]] = []

    def _load_template(self, p: Path) -> Dict[str, Any]:
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise SystemExit(f"Failed to load template: {p}\n{e}")

    def _infer_page_dims(self) -> List[int]:
        try:
            img = cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                h, w = img.shape[:2]
                return [int(w), int(h)]
        except Exception:
            pass
        return [0, 0]

    def field_blocks(self):
        return list(self.template["fieldBlocks"].items())

    def default_dir_for_type(self, field_type: Optional[str]) -> str:
        if field_type in ("QTYPE_INT", "QTYPE_INT_FROM_1"):
            return "vertical"
        if field_type in ("QTYPE_MCQ4", "QTYPE_MCQ5"):
            return "horizontal"
        return "horizontal"

    def get_block(self, name: str) -> Dict[str, Any]:
        """
        Render view of a block with derived defaults. Do not modify/persist this dict.
        """
        base = self.template["fieldBlocks"].get(name, {})
        fb = dict(base)
        # Direction (derived if missing)
        ft = base.get("fieldType")
        if "direction" not in fb or fb.get("direction") is None:
            fb["direction"] = self.default_dir_for_type(ft)
        # bubbleDimensions fallback
        if "bubbleDimensions" not in fb or not isinstance(fb.get("bubbleDimensions"), list):
            fb["bubbleDimensions"] = list(self.template.get("bubbleDimensions", [20, 20]))
        # bubbleValues derived for render if absent
        if not fb.get("bubbleValues") and ft in FIELD_TYPES:
            fb["bubbleValues"] = list(FIELD_TYPES[ft].get("bubbleValues", []))
        return fb

    def get_block_base(self, name: str) -> Dict[str, Any]:
        return self.template["fieldBlocks"].setdefault(name, {})

    def push_state(self, reason: str = ""):
        # snapshot current state
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

    def next_block_name(self) -> str:
        base = "FieldBlock"
        n = 1
        existing = set(self.template["fieldBlocks"].keys())
        while f"{base}_{n}" in existing:
            n += 1
        return f"{base}_{n}"

    def add_block(self, name: str, rect: QtCore.QRectF) -> None:
        self.push_state("add_block")
        x, y = int(rect.left()), int(rect.top())
        w, h = max(30, int(rect.width())), max(30, int(rect.height()))
        # Minimal persisted data; renderer derives direction/bubbleValues
        self.template["fieldBlocks"][name] = {
            "origin": [x, y],
            "bubblesGap": w,
            "labelsGap": h,
            "fieldLabels": ["q1..1"],
            # leave fieldType unset by default
        }
        self.changed.emit()

    def remove_block(self, name: str) -> None:
        if name in self.template["fieldBlocks"]:
            self.push_state("remove_block")
            del self.template["fieldBlocks"][name]
            self.changed.emit()

    def save_as_edited(self) -> Path:
        cleaned = copy.deepcopy(self.template)
        for name, fb in cleaned.get("fieldBlocks", {}).items():
            ft = fb.get("fieldType")
            # Drop direction if it matches type-default
            if "direction" in fb:
                if fb["direction"] == self.default_dir_for_type(ft):
                    fb.pop("direction", None)
            # Drop bubbleValues if it matches type default
            if "bubbleValues" in fb and ft in FIELD_TYPES:
                if fb["bubbleValues"] == FIELD_TYPES[ft].get("bubbleValues", []):
                    fb.pop("bubbleValues", None)
        out = self.template_path.with_name(self.template_path.stem + ".edited.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2)
        return out

# New: run template preprocessors EXACTLY like main.py (apply_preprocessors)
def run_preprocessors_for_editor(template_path: Path, image_path: Path) -> Optional[np.ndarray]:
    """
    Run preprocessing EXACTLY like the main pipeline:
      - Load config.json (if present) merged over CONFIG_DEFAULTS.
      - Build Template(template.json, config).
      - Call Template.image_instance_ops.apply_preprocessors(file_path, image, template),
        which internally:
          * resizes to processing_width/height,
          * applies preProcessors in order (CropPage, CropOnMarkers, blurs, levels, etc.) without overrides,
          * returns the processed image or None on failure.
      - For editor overlay consistency, resize the processed image to template.pageDimensions.
    """
    try:
        # Ensure pipeline imports are available
        if Template is None or CONFIG_DEFAULTS is None:
            raise RuntimeError(
                "Pipeline imports unavailable in Qt editor. "
                f"{'(Details: ' + PIPELINE_IMPORT_ERROR + ')' if 'PIPELINE_IMPORT_ERROR' in globals() and PIPELINE_IMPORT_ERROR else ''}"
            )
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image for preprocessing: {image_path}")

        # Load tuning config exactly like main (no forced overrides)
        template_dir = template_path.parent
        cfg = CONFIG_DEFAULTS
        cfg_path = template_dir / "config.json"
        if open_config_with_defaults is not None and cfg_path.exists():
            cfg = open_config_with_defaults(cfg_path)

        # Build Template and run its preprocessors as-is
        tmpl = Template(template_path, cfg)
        processed = tmpl.image_instance_ops.apply_preprocessors(str(image_path), img, tmpl)
        if processed is None:
            raise RuntimeError(
                "Preprocessors returned None (cropping/preprocessing failed). "
                "Ensure your template.json includes CropPage or CropOnMarkers and that inputs are valid."
            )

        # Align preview to template coordinates (same as draw_template_layout does)
        try:
            pw, ph = tmpl.page_dimensions
            if int(pw) > 0 and int(ph) > 0:
                processed = ImageUtils.resize_util(processed, int(pw), int(ph))
        except Exception:
            # keep processed as-is if pageDimensions invalid
            pass
        return processed
    except Exception as e:
        LOG.error(f"Qt editor preprocessing error: {e}")
        # Bubble the error to caller so UI can show a dialog
        raise

def np_gray_to_qpixmap(img: np.ndarray) -> QtGui.QPixmap:
    h, w = img.shape[:2]
    bytes_per_line = w
    qimg = QtGui.QImage(img.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_Grayscale8)
    return QtGui.QPixmap.fromImage(qimg.copy())

class BlockGraphicsItem(QtWidgets.QGraphicsItem):
    def __init__(self, name: str, model: 'TemplateModel'):
        super().__init__()
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges, True)  # NEW
        self.name = name
        self.model = model
        # Keep rect local; use item position for origin
        self._rect = QtCore.QRectF(0, 0, 120, 60)  # CHANGED
        # New: resize handles
        self._handles: Dict[str, QtWidgets.QGraphicsRectItem] = {}
        self._handle_size = 14.0  # larger grab area
        self.sync_from_model()
        self._create_handles()
        self._update_handles_positions()
        self._set_handles_visible(False)

    def boundingRect(self) -> QtCore.QRectF:
        # Slight padding for the border and handles
        r = self._rect
        pad = max(4.0, self._handle_size / 2.0)  # CHANGED
        return QtCore.QRectF(r.left() - pad, r.top() - pad, r.width() + 2 * pad, r.height() + 2 * pad)

    def paint(self, painter: QtGui.QPainter, option, widget=None):
        fb = self.model.get_block(self.name)
        # Base rect
        pen = QtGui.QPen(QtGui.QColor(200, 60, 60) if self.isSelected() else QtGui.QColor(40, 160, 40), 2)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawRect(self._rect)

        # Title
        painter.setPen(QtGui.QPen(QtGui.QColor(240, 240, 240)))
        painter.setFont(QtGui.QFont("", 9))
        painter.drawText(QtCore.QRectF(self._rect.left()+4, self._rect.top()-16, self._rect.width(), 16),
                         QtCore.Qt.AlignmentFlag.AlignLeft, f"{self.name}")  # CHANGED

        # Bubble preview (grid using bubbleDimensions, gaps, direction, and labels)
        vals: List[str] = fb.get("bubbleValues") or []
        labels: List[str] = fb.get("fieldLabels") or []
        if not isinstance(vals, list):
            vals = []
        if not isinstance(labels, list):
            labels = []
        bw, bh = fb.get("bubbleDimensions", self.model.template.get("bubbleDimensions", [20, 20]))
        try:
            bw = float(bw)
            bh = float(bh)
        except Exception:
            bw, bh = 20.0, 20.0
        direction = fb.get("direction", "horizontal")
        bubbles_gap = int(fb.get("bubblesGap", 12))
        labels_gap = int(fb.get("labelsGap", 12))
        base_x = self._rect.left()
        base_y = self._rect.top()
        painter.setPen(QtGui.QPen(QtGui.QColor(240, 200, 60), 1))
        if direction == "vertical":
            for li, _lab in enumerate(labels):
                start_x = base_x + li * labels_gap
                start_y = base_y
                for vi, _val in enumerate(vals):
                    x = start_x
                    y = start_y + vi * bubbles_gap
                    rx = x + bw * 0.10
                    ry = y + bh * 0.10
                    rw = max(2.0, bw - 2 * bw * 0.10)
                    rh = max(2.0, bh - 2 * bh * 0.10)
                    painter.drawRect(QtCore.QRectF(rx, ry, rw, rh))
        else:
            for li, _lab in enumerate(labels):
                start_x = base_x
                start_y = base_y + li * labels_gap
                for vi, _val in enumerate(vals):
                    x = start_x + vi * bubbles_gap
                    y = start_y
                    rx = x + bw * 0.10
                    ry = y + bh * 0.10
                    rw = max(2.0, bw - 2 * bw * 0.10)
                    rh = max(2.0, bh - 2 * bh * 0.10)
                    painter.drawRect(QtCore.QRectF(rx, ry, rw, rh))

    def sync_from_model(self):
        fb = self.model.get_block(self.name)
        ox, oy = fb.get("origin", [0, 0])
        direction = fb.get("direction", "horizontal")
        vals: List[str] = fb.get("bubbleValues", []) or []
        labels: List[str] = fb.get("fieldLabels", []) or []
        bubbles_gap = int(fb.get("bubblesGap", 12))
        labels_gap = int(fb.get("labelsGap", 12))
        bw, bh = fb.get("bubbleDimensions", self.model.template.get("bubbleDimensions", [20, 20]))
        try:
            bw = int(bw)
            bh = int(bh)
        except Exception:
            bw, bh = 20, 20
        n_vals = max(1, len(vals))
        n_fields = max(1, len(labels))
        if direction == "vertical":
            values_dimension = int(bubbles_gap * (n_vals - 1) + bh)
            fields_dimension = int(labels_gap * (n_fields - 1) + bw)
            width, height = fields_dimension, values_dimension
        else:
            values_dimension = int(bubbles_gap * (n_vals - 1) + bw)
            fields_dimension = int(labels_gap * (n_fields - 1) + bh)
            width, height = values_dimension, fields_dimension
        self.prepareGeometryChange()
        self._rect = QtCore.QRectF(0, 0, max(30, width), max(30, height))
        self.setPos(float(ox), float(oy))

    def update_model_from_item(self):
        """Update origin and derive gaps from current rect dimensions (inverse of calculate_block_dimensions)."""
        fb = self.model.get_block_base(self.name)
        r = self._rect
        fb["origin"] = [int(self.pos().x()), int(self.pos().y())]
        # Use derived default direction when not explicitly set
        base_dir = fb.get("direction", None)
        direction = base_dir if base_dir is not None else self.model.default_dir_for_type(fb.get("fieldType"))
        # use render for counts, base for persistence
        fb_render = self.model.get_block(self.name)
        vals: List[str] = fb_render.get("bubbleValues", []) or []
        labels: List[str] = fb_render.get("fieldLabels", []) or fb.get("fieldLabels", [])
        n_vals = max(1, len(vals))
        n_fields = max(1, len(labels))
        bw, bh = fb_render.get("bubbleDimensions", self.model.template.get("bubbleDimensions", [20, 20]))
        try:
            bw = float(bw)
            bh = float(bh)
        except Exception:
            bw, bh = 20.0, 20.0
        width = float(r.width())
        height = float(r.height())
        if direction == "vertical":
            fields_dimension = width
            values_dimension = height
            fb["labelsGap"] = int(round((fields_dimension - bw) / (n_fields - 1))) if n_fields > 1 else int(bw)
            fb["bubblesGap"] = int(round((values_dimension - bh) / (n_vals - 1))) if n_vals > 1 else int(bh)
        else:
            values_dimension = width
            fields_dimension = height
            fb["bubblesGap"] = int(round((values_dimension - bw) / (n_vals - 1))) if n_vals > 1 else int(bw)
            fb["labelsGap"] = int(round((fields_dimension - bh) / (n_fields - 1))) if n_fields > 1 else int(bh)

    # New: handle creation and positioning
    def _create_handles(self):
        roles = ("tl", "tr", "bl", "br")
        for r in roles:
            item = _ResizeHandle(self, r, self._handle_size)
            self._handles[r] = item

    def _update_handles_positions(self):
        r = self._rect
        s = self._handle_size
        offs = s / 2.0
        positions = {
            "tl": QtCore.QPointF(r.left() - offs, r.top() - offs),
            "tr": QtCore.QPointF(r.right() - offs, r.top() - offs),
            "bl": QtCore.QPointF(r.left() - offs, r.bottom() - offs),
            "br": QtCore.QPointF(r.right() - offs, r.bottom() - offs),
        }
        for role, item in self._handles.items():
            # Suppress itemChange while programmatically repositioning to avoid recursion
            if hasattr(item, "setPosSilently"):
                item.setPosSilently(positions[role])  # type: ignore[attr-defined]
            else:
                item.setPos(positions[role])

    def _set_handles_visible(self, vis: bool):
        for item in self._handles.values():
            item.setVisible(vis)

    def setSelected(self, selected: bool):
        super().setSelected(selected)
        self._set_handles_visible(selected)

    # Single safe override; do not call super().itemChange (avoid enum type error)
    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            # Accept movement; keep rect anchored at (0,0) and update model/handles
            QtCore.QTimer.singleShot(0, self._update_handles_positions)
            QtCore.QTimer.singleShot(0, self.update_model_from_item)
            return value
        return value

    # Public API used by handle items
    def resize_from_handle(self, role: str, handle_local_pos: QtCore.QPointF):
        r = self._rect
        MIN_W, MIN_H = 30, 30
        offs = self._handle_size / 2.0
        # Handles are positioned with top-left at (corner - offs); use center as actual corner
        corner = handle_local_pos + QtCore.QPointF(offs, offs)  # CHANGED
        new = QtCore.QRectF(r)
        if role == "tl":
            new.setTopLeft(corner)
        elif role == "tr":
            new.setTopRight(corner)
        elif role == "bl":
            new.setBottomLeft(corner)
        elif role == "br":
            new.setBottomRight(corner)
        new = new.normalized()
        if new.width() < MIN_W:
            new.setWidth(MIN_W)
        if new.height() < MIN_H:
            new.setHeight(MIN_H)
        self.set_rect(new)
        self._update_handles_positions()

    def set_rect(self, rect: QtCore.QRectF):
        self.prepareGeometryChange()
        self._rect = rect.normalized()
        self.update_model_from_item()
        self.update()
        # snapshot after resize
        self.model.push_state("resize_block")

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        super().mouseReleaseEvent(event)
        # snapshot after move
        self.model.push_state("move_block")

# New: subclass handle item to forward drag to parent
class _ResizeHandle(QtWidgets.QGraphicsRectItem):
    def __init__(self, parent: BlockGraphicsItem, role: str, size: float):
        super().__init__(0, 0, size, size, parent)
        self._parent = parent
        self.role = role  # type: ignore[attr-defined]
        self.setBrush(QtGui.QBrush(QtGui.QColor(255, 200, 0, 220)))
        self.setPen(QtGui.QPen(QtGui.QColor(30, 30, 30), 1))
        self.setZValue(1000)
        # Let parent control actual geometry change; prevent scene panning of the block while resizing
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges, True)
        # Suppress recursion when parent repositions this handle
        self._suppress_item_change = False
        self._dragging = False

    def setPosSilently(self, pos: QtCore.QPointF):
        self._suppress_item_change = True
        try:
            super().setPos(pos)
        finally:
            self._suppress_item_change = False

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        self._dragging = True
        # Disable parent movement while dragging a handle to avoid panning
        self._parent.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        self._dragging = False
        self._parent.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        super().mouseReleaseEvent(event)

    def itemChange(self, change, value):
        # Avoid calling super().itemChange to prevent enum type issues
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            if self._suppress_item_change:
                return value
            # value is the new pos in parent's coordinates; pass directly
            self._parent.resize_from_handle(self.role, value)  # type: ignore[arg-type]
            return value
        return value

class GraphicsView(QtWidgets.QGraphicsView):
    newRectDrawn = QtCore.pyqtSignal(QtCore.QRectF)

    def __init__(self, scene: QtWidgets.QGraphicsScene):
        super().__init__(scene)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._adding = False
        self._rubber_item: Optional[QtWidgets.QGraphicsRectItem] = None
        self._start_scene_pt: Optional[QtCore.QPointF] = None

    def enter_add_mode(self):
        self._adding = True
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)

    def wheelEvent(self, event: QtGui.QWheelEvent):
        # Zoom under mouse
        zoom_in = event.angleDelta().y() > 0
        factor = 1.15 if zoom_in else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if self._adding and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._start_scene_pt = self.mapToScene(event.position().toPoint())
            self._rubber_item = QtWidgets.QGraphicsRectItem()
            pen = QtGui.QPen(QtGui.QColor(255, 200, 0), 1, QtCore.Qt.PenStyle.DashLine)
            self._rubber_item.setPen(pen)
            self.scene().addItem(self._rubber_item)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self._adding and self._start_scene_pt is not None and self._rubber_item:
            cur = self.mapToScene(event.position().toPoint())
            rect = QtCore.QRectF(self._start_scene_pt, cur).normalized()
            self._rubber_item.setRect(rect)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if self._adding and self._rubber_item and event.button() == QtCore.Qt.MouseButton.LeftButton:
            rect = self._rubber_item.rect()
            self.scene().removeItem(self._rubber_item)
            self._rubber_item = None
            self._start_scene_pt = None
            self._adding = False
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            if rect.width() >= 10 and rect.height() >= 10:
                self.newRectDrawn.emit(rect)
            event.accept()
            return
        super().mouseReleaseEvent(event)

class BlockPanel(QtWidgets.QGroupBox):
    changed = QtCore.pyqtSignal(str)  # emits block name

    def __init__(self, name: str, model: 'TemplateModel'):
        super().__init__(name)
        # Collapsible panel using checkable header to toggle body visibility
        self.setCheckable(True)  # CHANGED
        self.setChecked(False)   # CHANGED
        self.name = name
        self.model = model

        # Container for body to collapse
        self._body = QtWidgets.QWidget()  # NEW
        form = QtWidgets.QFormLayout(self._body)  # NEW
        outer = QtWidgets.QVBoxLayout(self)  # NEW
        outer.setContentsMargins(6, 6, 6, 6)  # NEW
        outer.addWidget(self._body)  # NEW

        fb = self.model.get_block(name)

        # FieldName
        self.field_name = QtWidgets.QLineEdit(name)
        form.addRow("FieldName", self.field_name)

        # BubbleValues (CSV)
        vals = fb.get("bubbleValues", [])
        self.bubble_values = QtWidgets.QLineEdit(to_csv(vals) if isinstance(vals, list) else "")
        form.addRow("BubbleValues", self.bubble_values)

        # Direction
        self.direction = QtWidgets.QComboBox()
        self.direction.addItems(["horizontal", "vertical"])
        self.direction.setCurrentText(fb.get("direction", "horizontal"))
        form.addRow("Direction", self.direction)

        # FieldLabels (CSV or range)
        self.field_labels = QtWidgets.QLineEdit(to_csv(fb.get("fieldLabels", [])))
        form.addRow("FieldLabels", self.field_labels)

        # LabelsGap (height)
        self.labels_gap = QtWidgets.QSpinBox()
        self.labels_gap.setRange(0, 10000)
        self.labels_gap.setValue(int(fb.get("labelsGap", 60)))
        form.addRow("LabelsGap", self.labels_gap)

        # BubblesGap (width)
        self.bubbles_gap = QtWidgets.QSpinBox()
        self.bubbles_gap.setRange(0, 10000)
        self.bubbles_gap.setValue(int(fb.get("bubblesGap", 120)))
        form.addRow("BubblesGap", self.bubbles_gap)

        # origin
        ox, oy = fb.get("origin", [0, 0])
        self.origin_x = QtWidgets.QSpinBox()
        self.origin_x.setRange(0, 10000)
        self.origin_x.setValue(int(ox))
        self.origin_y = QtWidgets.QSpinBox()
        self.origin_y.setRange(0, 10000)
        self.origin_y.setValue(int(oy))
        origin_layout = QtWidgets.QHBoxLayout()
        origin_layout.addWidget(QtWidgets.QLabel("x"))
        origin_layout.addWidget(self.origin_x)
        origin_layout.addWidget(QtWidgets.QLabel("y"))
        origin_layout.addWidget(self.origin_y)
        form.addRow("Origin", origin_layout)

        # Fieldtype (optional)
        self.field_type = QtWidgets.QComboBox()
        self.field_type.addItem("")  # none
        self.field_type.addItems(list(FIELD_TYPES.keys()))
        self.field_type.setCurrentText(fb.get("fieldType", ""))
        form.addRow("Fieldtype", self.field_type)
        # Delete button
        self.delete_btn = QtWidgets.QPushButton("Delete")
        self.delete_btn.setStyleSheet("QPushButton { color: #ff6666; }")
        form.addRow(self.delete_btn)

        # Connections
        self.field_name.editingFinished.connect(self._apply)
        self.bubble_values.editingFinished.connect(self._apply)
        self.direction.currentTextChanged.connect(self._apply)
        self.field_labels.editingFinished.connect(self._apply)
        self.labels_gap.valueChanged.connect(self._apply)
        self.bubbles_gap.valueChanged.connect(self._apply)
        self.origin_x.valueChanged.connect(self._apply)
        self.origin_y.valueChanged.connect(self._apply)
        self.field_type.currentTextChanged.connect(self._apply_fieldtype)
        self.toggled.connect(self._toggle_body)  # NEW
        self.delete_btn.clicked.connect(self._delete_self)

    def _toggle_body(self, on: bool):  # NEW
        self._body.setVisible(on)

    def _apply_fieldtype(self, text: str):
        base = self.model.get_block_base(self.name)
        self.model.push_state("change_fieldtype")
        if text:
            base["fieldType"] = text
        else:
            base.pop("fieldType", None)
        # Do not auto-write bubbleValues or direction; renderer derives them.
        self.changed.emit(self.name)

    def _apply(self):
        new_name = self.field_name.text().strip()
        base = self.model.get_block_base(self.name)
        self.model.push_state("panel_apply")
        # If renamed
        if new_name and new_name != self.name:
            # Move the dict key
            self.model.template["fieldBlocks"][new_name] = base
            del self.model.template["fieldBlocks"][self.name]
            self.name = new_name
            self.setTitle(new_name)

        # Persist bubbleValues only if user provided explicit text
        bv_text = self.bubble_values.text().strip()
        if bv_text != "":
            base["bubbleValues"] = parse_csv_or_range(bv_text)
        else:
            base.pop("bubbleValues", None)
        # Direction: omit if equals type-default
        sel_dir = self.direction.currentText()
        ft = base.get("fieldType")
        if sel_dir == self.model.default_dir_for_type(ft):
            base.pop("direction", None)
        else:
            base["direction"] = sel_dir
        base["fieldLabels"] = parse_csv_or_range(self.field_labels.text())
        base["labelsGap"] = int(self.labels_gap.value())
        base["bubblesGap"] = int(self.bubbles_gap.value())
        base["origin"] = [int(self.origin_x.value()), int(self.origin_y.value())]
        self.changed.emit(self.name)

    def sync_from_model(self):
        fb_render = self.model.get_block(self.name)
        fb_base = self.model.get_block_base(self.name)
        # Show only explicit bubbleValues (leave blank if derived)
        self.bubble_values.setText(to_csv(fb_base.get("bubbleValues", [])))
        # Direction UI shows explicit if set, else default-for-type
        self.direction.setCurrentText(fb_base.get("direction", self.model.default_dir_for_type(fb_base.get("fieldType"))))
        self.field_labels.setText(to_csv(fb_base.get("fieldLabels", [])))
        self.labels_gap.setValue(int(fb_base.get("labelsGap", 60)))
        self.bubbles_gap.setValue(int(fb_base.get("bubblesGap", 120)))
        ox, oy = fb_base.get("origin", [0, 0])
        self.origin_x.setValue(int(ox))
        self.origin_y.setValue(int(oy))
        self.field_type.setCurrentText(fb_base.get("fieldType", ""))

    def _delete_self(self):
        self.model.remove_block(self.name)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, template_path: Path, image_path: Optional[Path]):
        super().__init__()
        self.setWindowTitle("OMR Template Editor (Qt6)")
        self.resize(1280, 800)
        # Wider sidebar and smoother docks
        self.setDockOptions(self.DockOption.AllowTabbedDocks | self.DockOption.AnimatedDocks)  # NEW
        self._pending_select_name: Optional[str] = None  # select after refresh

        # Model
        if image_path is None:
            image_path = find_first_image_under(PROJECT_ROOT / "inputs")
        if image_path is None:
            raise SystemExit("No image provided and none found under ./inputs")
        self.model = TemplateModel(template_path, image_path)

        # Scene/View
        self.scene = QtWidgets.QGraphicsScene(self)
        self.view = GraphicsView(self.scene)
        # Reduce paint trails when moving items
        self.view.setViewportUpdateMode(QtWidgets.QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)  # NEW
        self.setCentralWidget(self.view)

        # Load and preprocess image (EXACT main pipeline). Show explicit errors if it fails.
        preprocessing_error: Optional[str] = None
        processed: Optional[np.ndarray] = None
        try:
            processed = run_preprocessors_for_editor(template_path, image_path)
        except Exception as e:
            preprocessing_error = str(e)

        if processed is None:
            # Build a helpful message
            msg_lines = []
            if 'PIPELINE_IMPORT_ERROR' in globals() and PIPELINE_IMPORT_ERROR:
                msg_lines.append("Failed to import pipeline modules required for cropping.")
                msg_lines.append(f"Reason: {PIPELINE_IMPORT_ERROR}")
                msg_lines.append("Tip: run from repo root and ensure dependencies are installed (e.g., pip install -r requirements.txt).")
            if preprocessing_error:
                msg_lines.append(preprocessing_error)
            if not msg_lines:
                msg_lines.append("Cropping/preprocessing failed for an unknown reason.")
            full_msg = "\n".join(msg_lines)
            LOG.error(full_msg)
            QtWidgets.QMessageBox.critical(
                self,
                "Preprocessing Error",
                full_msg,
            )
            # Fallback to original image so the editor still opens
            pixmap = load_image_as_pixmap(image_path)
        else:
            pixmap = np_gray_to_qpixmap(processed)

        self.image_item = self.scene.addPixmap(pixmap)
        self.image_item.setZValue(-1000)
        self.scene.setSceneRect(self.image_item.boundingRect())  # NEW

        # Sidebar
        self.sidebar = QtWidgets.QDockWidget("FieldBlocks", self)
        self.sidebar.setAllowedAreas(QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
        self.sidebar.setMinimumWidth(420)  # NEW
        self.sidebar_widget = QtWidgets.QWidget()
        self.sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar_widget)
        self.sidebar_layout.setContentsMargins(6, 6, 6, 6)
        self.sidebar_layout.setSpacing(6)
        self.sidebar_scroll = QtWidgets.QScrollArea()
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_inner = QtWidgets.QWidget()
        self.sidebar_inner_layout = QtWidgets.QVBoxLayout(self.sidebar_inner)
        self.sidebar_inner_layout.setContentsMargins(0, 0, 0, 0)
        self.sidebar_inner_layout.setSpacing(6)
        self.sidebar_scroll.setWidget(self.sidebar_inner)
        self.sidebar_layout.addWidget(self.sidebar_scroll)
        self.sidebar_layout.addStretch(0)
        self.sidebar.setWidget(self.sidebar_widget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.sidebar)

        # Toolbar
        tb = self.addToolBar("Tools")
        add_btn = QtGui.QAction("Add FieldBlock", self)
        add_btn.triggered.connect(self.on_add_block)
        save_btn = QtGui.QAction("Save", self)
        save_btn.triggered.connect(self.on_save)
        tb.addAction(add_btn)
        tb.addAction(save_btn)

        # Build existing blocks
        self.block_items: Dict[str, BlockGraphicsItem] = {}
        self.block_panels: Dict[str, BlockPanel] = {}
        for name, _ in self.model.field_blocks():
            self._create_block_item_and_panel(name)

        # Connect rubberband result
        self.view.newRectDrawn.connect(self._add_block_from_rect)

        # React to model changes (e.g., external updates)
        self.model.changed.connect(self.refresh_items)
        # Jump to/expand panel for selected item
        self.scene.selectionChanged.connect(self._on_scene_selection_changed)  # NEW

    def on_add_block(self):
        self.view.enter_add_mode()

    def _add_block_from_rect(self, rect: QtCore.QRectF):
        # Avoid duplicate creation: only update model; refresh_items will build UI
        name = self.model.next_block_name()
        self._pending_select_name = name
        self.model.add_block(name, rect)
        # Select new item and unselect others; focus its panel
        for n, it in self.block_items.items():
            it.setSelected(n == name)
        panel = self.block_panels.get(name)
        if panel:
            for n, pnl in self.block_panels.items():
                pnl.setChecked(pnl is panel)
            self.sidebar_scroll.ensureWidgetVisible(panel)

    def _create_block_item_and_panel(self, name: str):
        # Graphics item
        item = BlockGraphicsItem(name, self.model)
        self.scene.addItem(item)
        self.block_items[name] = item

        # Sidebar panel
        panel = BlockPanel(name, self.model)
        panel.changed.connect(self._on_panel_changed)
        self.block_panels[name] = panel
        self.sidebar_inner_layout.addWidget(panel)
        panel._toggle_body(False)  # start collapsed (NEW)

    def _on_panel_changed(self, name: str):
        # Simply refresh; handles rename/delete consistently
        self.refresh_items()

    def refresh_items(self):
        # Sync items and panels from model state
        names_now = set(n for n, _ in self.model.field_blocks())
        # Preserve collapse/expanded states
        expanded_states: Dict[str, bool] = {n: w.isChecked() for n, w in self.block_panels.items()}  # NEW

        # Remove deleted
        for old in list(self.block_items.keys()):
            if old not in names_now:
                self.scene.removeItem(self.block_items[old])
                del self.block_items[old]
        for old in list(self.block_panels.keys()):
            if old not in names_now:
                w = self.block_panels[old]
                self.sidebar_inner_layout.removeWidget(w)
                w.deleteLater()
                del self.block_panels[old]

        # Update or create
        for name in names_now:
            if name not in self.block_items:
                it = BlockGraphicsItem(name, self.model)
                self.scene.addItem(it)
                self.block_items[name] = it
            else:
                self.block_items[name].sync_from_model()
                self.block_items[name].update()
            if name not in self.block_panels:
                pnl = BlockPanel(name, self.model)
                pnl.changed.connect(self._on_panel_changed)
                self.block_panels[name] = pnl
                self.sidebar_inner_layout.addWidget(pnl)
            else:
                self.block_panels[name].sync_from_model()

        # Restore expanded state
        for name in names_now:
            if name in expanded_states:
                self.block_panels[name].setChecked(expanded_states[name])  # NEW
        # Select newly added block and focus its panel
        if self._pending_select_name and self._pending_select_name in self.block_items:
            for n, it in self.block_items.items():
                it.setSelected(n == self._pending_select_name)
            panel = self.block_panels.get(self._pending_select_name)
            if panel:
                for n, pnl in self.block_panels.items():
                    pnl.setChecked(pnl is panel)
                self.sidebar_scroll.ensureWidgetVisible(panel)
            self._pending_select_name = None
        # Keep layout tidy
        self.sidebar_inner_layout.addStretch(0)

    def _on_scene_selection_changed(self):  # NEW
        # Jump to and expand the panel for the first selected block
        selected_names = [n for n, it in self.block_items.items() if it.isSelected()]
        if not selected_names:
            return
        name = selected_names[0]
        panel = self.block_panels.get(name)
        if panel is None:
            return
        # Uncheck others, check this one
        for n, pnl in self.block_panels.items():
            pnl.setChecked(pnl is panel)
        self.sidebar_scroll.ensureWidgetVisible(panel)

    def on_save(self):
        out = self.model.save_as_edited()
        QtWidgets.QMessageBox.information(self, "Saved", f"Saved:\n{out}")

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        # Delete selected block
        if event.key() in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
            selected_names = [n for n, it in self.block_items.items() if it.isSelected()]
            if selected_names:
                self.model.remove_block(selected_names[0])
                return
        # Undo/Redo
        if event.matches(QtGui.QKeySequence.StandardKey.Undo):
            if self.model.undo():
                return
        if event.matches(QtGui.QKeySequence.StandardKey.Redo):
            if self.model.redo():
                return
        super().keyPressEvent(event)

def parse_args(argv: List[str]) -> Tuple[Optional[Path], Optional[Path]]:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=False, help="Path to template.json. If omitted, auto-detect under ./inputs")
    ap.add_argument("--image", required=False, help="Path to an image. If omitted, auto-detect (preferring same folder as template)")
    args = ap.parse_args(argv)
    t = Path(args.template).resolve() if args.template else None
    img = Path(args.image).resolve() if args.image else None
    return t, img

def main():
    template_path, image_path = parse_args(sys.argv[1:])
    # Auto-detect template and image like main.py workflow if not provided
    if template_path is None:
        template_path = find_first_template_under(PROJECT_ROOT / "inputs")
        if template_path is None:
            raise SystemExit("No template.json provided and none found under ./inputs")
    # Prefer image under the template's directory; fallback to first under ./inputs
    if image_path is None:
        local_img = find_first_image_under(template_path.parent)
        image_path = local_img if local_img is not None else find_first_image_under(PROJECT_ROOT / "inputs")
        if image_path is None:
            raise SystemExit("No image provided and none found under ./inputs")

    app = QtWidgets.QApplication(sys.argv)
    # Dark-ish palette for contrast
    palette = app.palette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(30, 30, 30))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(37, 37, 37))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(230, 230, 230))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(230, 230, 230))
    app.setPalette(palette)
    w = MainWindow(template_path, image_path)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()