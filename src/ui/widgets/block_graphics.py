"""BlockGraphicsItem, _ResizeHandle, GraphicsView — canvas widgets for block editing."""
from typing import TYPE_CHECKING, Dict, List, Optional

from PyQt6 import QtCore, QtGui, QtWidgets

if TYPE_CHECKING:
    from ..models.template_model import TemplateModel


class BlockGraphicsItem(QtWidgets.QGraphicsItem):
    def __init__(self, name: str, model: "TemplateModel"):
        super().__init__()
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges,
            True,
        )
        self.name = name
        self.model = model
        self._rect = QtCore.QRectF(0, 0, 120, 60)
        self._handles: Dict[str, "_ResizeHandle"] = {}
        self._handle_size = 14.0
        self.sync_from_model()
        self._create_handles()
        self._update_handles_positions()
        self._set_handles_visible(False)

    # ------------------------------------------------------------------
    def boundingRect(self) -> QtCore.QRectF:
        r = self._rect
        pad = max(4.0, self._handle_size / 2.0)
        return QtCore.QRectF(
            r.left() - pad,
            r.top() - pad,
            r.width() + 2 * pad,
            r.height() + 2 * pad,
        )

    def paint(self, painter: QtGui.QPainter, option, widget=None):
        fb = self.model.get_block(self.name)
        pen = QtGui.QPen(
            QtGui.QColor(200, 60, 60)
            if self.isSelected()
            else QtGui.QColor(40, 160, 40),
            2,
        )
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawRect(self._rect)

        painter.setPen(QtGui.QPen(QtGui.QColor(240, 240, 240)))
        painter.setFont(QtGui.QFont("", 9))
        painter.drawText(
            QtCore.QRectF(
                self._rect.left() + 4,
                self._rect.top() - 16,
                self._rect.width(),
                16,
            ),
            QtCore.Qt.AlignmentFlag.AlignLeft,
            f"{self.name}",
        )

        # Bubble preview
        vals: List[str] = fb.get("bubbleValues") or []
        labels: List[str] = fb.get("fieldLabels") or []
        if not isinstance(vals, list):
            vals = []
        if not isinstance(labels, list):
            labels = []
        bw, bh = fb.get(
            "bubbleDimensions", self.model.template.get("bubbleDimensions", [20, 20])
        )
        try:
            bw, bh = float(bw), float(bh)
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
                    painter.drawRect(
                        QtCore.QRectF(
                            x + bw * 0.10,
                            y + bh * 0.10,
                            max(2.0, bw * 0.80),
                            max(2.0, bh * 0.80),
                        )
                    )
        else:
            for li, _lab in enumerate(labels):
                start_x = base_x
                start_y = base_y + li * labels_gap
                for vi, _val in enumerate(vals):
                    x = start_x + vi * bubbles_gap
                    y = start_y
                    painter.drawRect(
                        QtCore.QRectF(
                            x + bw * 0.10,
                            y + bh * 0.10,
                            max(2.0, bw * 0.80),
                            max(2.0, bh * 0.80),
                        )
                    )

    # ------------------------------------------------------------------
    def sync_from_model(self):
        fb = self.model.get_block(self.name)
        ox, oy = fb.get("origin", [0, 0])
        direction = fb.get("direction", "horizontal")
        vals: List[str] = fb.get("bubbleValues", []) or []
        labels: List[str] = fb.get("fieldLabels", []) or []
        bubbles_gap = int(fb.get("bubblesGap", 12))
        labels_gap = int(fb.get("labelsGap", 12))
        bw, bh = fb.get(
            "bubbleDimensions", self.model.template.get("bubbleDimensions", [20, 20])
        )
        try:
            bw, bh = int(bw), int(bh)
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
        fb = self.model.get_block_base(self.name)
        r = self._rect
        fb["origin"] = [int(self.pos().x()), int(self.pos().y())]
        base_dir = fb.get("direction", None)
        direction = (
            base_dir
            if base_dir is not None
            else self.model.default_dir_for_type(fb.get("fieldType"))
        )
        fb_render = self.model.get_block(self.name)
        vals: List[str] = fb_render.get("bubbleValues", []) or []
        labels: List[str] = fb_render.get("fieldLabels", []) or fb.get(
            "fieldLabels", []
        )
        n_vals = max(1, len(vals))
        n_fields = max(1, len(labels))
        bw, bh = fb_render.get(
            "bubbleDimensions", self.model.template.get("bubbleDimensions", [20, 20])
        )
        try:
            bw, bh = float(bw), float(bh)
        except Exception:
            bw, bh = 20.0, 20.0
        width = float(r.width())
        height = float(r.height())
        if direction == "vertical":
            fields_dimension = width
            values_dimension = height
            fb["labelsGap"] = (
                int(round((fields_dimension - bw) / (n_fields - 1)))
                if n_fields > 1
                else int(bw)
            )
            fb["bubblesGap"] = (
                int(round((values_dimension - bh) / (n_vals - 1)))
                if n_vals > 1
                else int(bh)
            )
        else:
            values_dimension = width
            fields_dimension = height
            fb["bubblesGap"] = (
                int(round((values_dimension - bw) / (n_vals - 1)))
                if n_vals > 1
                else int(bw)
            )
            fb["labelsGap"] = (
                int(round((fields_dimension - bh) / (n_fields - 1)))
                if n_fields > 1
                else int(bh)
            )

    # ------------------------------------------------------------------
    # Handle management
    def _create_handles(self):
        for role in ("tl", "tr", "bl", "br"):
            self._handles[role] = _ResizeHandle(self, role, self._handle_size)

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
            item.setPosSilently(positions[role])

    def _set_handles_visible(self, vis: bool):
        for item in self._handles.values():
            item.setVisible(vis)

    def setSelected(self, selected: bool):
        super().setSelected(selected)
        self._set_handles_visible(selected)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            QtCore.QTimer.singleShot(0, self._update_handles_positions)
            QtCore.QTimer.singleShot(0, self.update_model_from_item)
            return value
        return value

    def resize_from_handle(self, role: str, handle_local_pos: QtCore.QPointF):
        r = self._rect
        MIN_W, MIN_H = 30, 30
        offs = self._handle_size / 2.0
        corner = handle_local_pos + QtCore.QPointF(offs, offs)
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
        self.model.push_state("resize_block")

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        super().mouseReleaseEvent(event)
        self.model.push_state("move_block")


# ---------------------------------------------------------------------------


class _ResizeHandle(QtWidgets.QGraphicsRectItem):
    def __init__(self, parent: BlockGraphicsItem, role: str, size: float):
        super().__init__(0, 0, size, size, parent)
        self._parent = parent
        self.role = role
        self.setBrush(QtGui.QBrush(QtGui.QColor(255, 200, 0, 220)))
        self.setPen(QtGui.QPen(QtGui.QColor(30, 30, 30), 1))
        self.setZValue(1000)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges,
            True,
        )
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
        self._parent.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False
        )
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        self._dragging = False
        self._parent.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True
        )
        super().mouseReleaseEvent(event)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            if self._suppress_item_change:
                return value
            self._parent.resize_from_handle(self.role, value)
            return value
        return value


# ---------------------------------------------------------------------------


class GraphicsView(QtWidgets.QGraphicsView):
    """Zoomable, pannable view with rubber-band block drawing mode."""

    newRectDrawn = QtCore.pyqtSignal(QtCore.QRectF)
    # Emitted when user picks a point in pick-origin mode
    originPicked = QtCore.pyqtSignal(QtCore.QPointF)

    def __init__(self, scene: QtWidgets.QGraphicsScene):
        super().__init__(scene)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(
            QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setViewportUpdateMode(
            QtWidgets.QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate
        )
        self._adding = False
        self._picking = False
        self._rubber_item: Optional[QtWidgets.QGraphicsRectItem] = None
        self._start_scene_pt: Optional[QtCore.QPointF] = None

    def enter_add_mode(self):
        self._adding = True
        self._picking = False
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)

    def enter_pick_mode(self):
        """Single-click mode: emits originPicked(scene_pos) on the next left click."""
        self._picking = True
        self._adding = False
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def exit_add_mode(self):
        self._adding = False
        self._picking = False
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

    def fit_image(self):
        self.fitInView(
            self.scene().sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio
        )

    def wheelEvent(self, event: QtGui.QWheelEvent):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if self._picking and event.button() == QtCore.Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            self.exit_add_mode()
            self.originPicked.emit(scene_pos)
            event.accept()
            return
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
            self._rubber_item.setRect(
                QtCore.QRectF(self._start_scene_pt, cur).normalized()
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if (
            self._adding
            and self._rubber_item
            and event.button() == QtCore.Qt.MouseButton.LeftButton
        ):
            rect = self._rubber_item.rect()
            self.scene().removeItem(self._rubber_item)
            self._rubber_item = None
            self._start_scene_pt = None
            self.exit_add_mode()
            if rect.width() >= 10 and rect.height() >= 10:
                self.newRectDrawn.emit(rect)
            event.accept()
            return
        super().mouseReleaseEvent(event)
