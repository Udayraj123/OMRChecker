"""BlockPanel — collapsible sidebar panel for editing a single FieldBlock."""
from typing import TYPE_CHECKING

from PyQt6 import QtCore, QtWidgets

from ..common import FIELD_TYPES, parse_csv_or_range, to_csv

if TYPE_CHECKING:
    from ..models.template_model import TemplateModel


class BlockPanel(QtWidgets.QGroupBox):
    """Collapsible QGroupBox for editing one field block's properties."""

    changed = QtCore.pyqtSignal(str)  # block name
    # Emitted when user wants to pick origin from canvas
    pickOriginRequested = QtCore.pyqtSignal(str)

    def __init__(self, name: str, model: "TemplateModel"):
        super().__init__(name)
        self.setCheckable(True)
        self.setChecked(False)
        self.name = name
        self.model = model
        self._updating = False  # guard against re-entrant signals

        # --- Body widget (collapses when header unchecked) ---
        self._body = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self._body)
        form.setContentsMargins(4, 4, 4, 4)
        form.setSpacing(4)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.addWidget(self._body)

        fb_base = self.model.get_block_base(name)

        # FieldName
        self.field_name = QtWidgets.QLineEdit(name)
        form.addRow("Name", self.field_name)

        # FieldType (primary — drives defaults for direction/bubbleValues)
        self.field_type = QtWidgets.QComboBox()
        self.field_type.addItem("(custom)")
        self.field_type.addItems(list(FIELD_TYPES.keys()))
        self.field_type.setCurrentText(
            fb_base.get("fieldType", "(custom)") or "(custom)"
        )
        form.addRow("FieldType", self.field_type)

        # Direction
        self.direction = QtWidgets.QComboBox()
        self.direction.addItems(["horizontal", "vertical"])
        self.direction.setCurrentText(
            fb_base.get(
                "direction", self.model.default_dir_for_type(fb_base.get("fieldType"))
            )
        )
        form.addRow("Direction", self.direction)

        # FieldLabels
        self.field_labels = QtWidgets.QLineEdit(to_csv(fb_base.get("fieldLabels", [])))
        form.addRow("FieldLabels", self.field_labels)

        # BubbleValues (CSV — shown; blank = auto from fieldType)
        vals = fb_base.get("bubbleValues", [])
        self.bubble_values = QtWidgets.QLineEdit(
            to_csv(vals) if isinstance(vals, list) else ""
        )
        self.bubble_values.setPlaceholderText("auto from FieldType")
        form.addRow("BubbleValues", self.bubble_values)

        # BubblesGap
        self.bubbles_gap = QtWidgets.QSpinBox()
        self.bubbles_gap.setRange(0, 10000)
        self.bubbles_gap.setValue(int(fb_base.get("bubblesGap", 60)))
        form.addRow("BubblesGap", self.bubbles_gap)

        # LabelsGap
        self.labels_gap = QtWidgets.QSpinBox()
        self.labels_gap.setRange(0, 10000)
        self.labels_gap.setValue(int(fb_base.get("labelsGap", 60)))
        form.addRow("LabelsGap", self.labels_gap)

        # BubbleDimensions (per-block override)
        bdims = fb_base.get("bubbleDimensions", [])
        self.bubble_dim_w = QtWidgets.QSpinBox()
        self.bubble_dim_w.setRange(0, 500)
        self.bubble_dim_w.setSpecialValueText("global")
        self.bubble_dim_w.setValue(
            int(bdims[0]) if isinstance(bdims, list) and bdims else 0
        )
        self.bubble_dim_h = QtWidgets.QSpinBox()
        self.bubble_dim_h.setRange(0, 500)
        self.bubble_dim_h.setSpecialValueText("global")
        self.bubble_dim_h.setValue(
            int(bdims[1]) if isinstance(bdims, list) and len(bdims) > 1 else 0
        )
        bdim_row = QtWidgets.QHBoxLayout()
        bdim_row.addWidget(QtWidgets.QLabel("W"))
        bdim_row.addWidget(self.bubble_dim_w)
        bdim_row.addWidget(QtWidgets.QLabel("H"))
        bdim_row.addWidget(self.bubble_dim_h)
        form.addRow("BubbleDims", bdim_row)

        # Origin
        ox, oy = fb_base.get("origin", [0, 0])
        self.origin_x = QtWidgets.QSpinBox()
        self.origin_x.setRange(0, 10000)
        self.origin_x.setValue(int(ox))
        self.origin_y = QtWidgets.QSpinBox()
        self.origin_y.setRange(0, 10000)
        self.origin_y.setValue(int(oy))
        pick_btn = QtWidgets.QPushButton("Pick…")
        pick_btn.setMaximumWidth(50)
        pick_btn.setToolTip("Click a point on the canvas to set origin")
        origin_row = QtWidgets.QHBoxLayout()
        origin_row.addWidget(QtWidgets.QLabel("X"))
        origin_row.addWidget(self.origin_x)
        origin_row.addWidget(QtWidgets.QLabel("Y"))
        origin_row.addWidget(self.origin_y)
        origin_row.addWidget(pick_btn)
        form.addRow("Origin", origin_row)

        # Delete button
        self.delete_btn = QtWidgets.QPushButton("Delete Block")
        self.delete_btn.setStyleSheet("QPushButton { color: #ff6666; }")
        form.addRow(self.delete_btn)

        # --- Wire signals ---
        self.field_name.editingFinished.connect(self._apply)
        self.field_type.currentTextChanged.connect(self._on_fieldtype_changed)
        self.direction.currentTextChanged.connect(self._apply)
        self.field_labels.editingFinished.connect(self._apply)
        self.bubble_values.editingFinished.connect(self._apply)
        self.bubbles_gap.valueChanged.connect(self._apply)
        self.labels_gap.valueChanged.connect(self._apply)
        self.bubble_dim_w.valueChanged.connect(self._apply)
        self.bubble_dim_h.valueChanged.connect(self._apply)
        self.origin_x.valueChanged.connect(self._apply)
        self.origin_y.valueChanged.connect(self._apply)
        self.toggled.connect(self._toggle_body)
        self.delete_btn.clicked.connect(self._delete_self)
        pick_btn.clicked.connect(lambda: self.pickOriginRequested.emit(self.name))

    # ------------------------------------------------------------------
    def _toggle_body(self, on: bool):
        self._body.setVisible(on)

    def _on_fieldtype_changed(self, text: str):
        if self._updating:
            return
        base = self.model.get_block_base(self.name)
        self.model.push_state("change_fieldtype")
        ft = text if text and text != "(custom)" else None
        if ft:
            base["fieldType"] = ft
        else:
            base.pop("fieldType", None)
        # Update direction hint (but don't force-persist it)
        self._updating = True
        self.direction.setCurrentText(self.model.default_dir_for_type(ft))
        self._updating = False
        self.changed.emit(self.name)

    def _apply(self):
        if self._updating:
            return
        new_name = self.field_name.text().strip()
        base = self.model.get_block_base(self.name)
        self.model.push_state("panel_apply")

        # Rename
        if new_name and new_name != self.name:
            self.model.template["fieldBlocks"][new_name] = base
            del self.model.template["fieldBlocks"][self.name]
            self.name = new_name
            self.setTitle(new_name)

        # BubbleValues
        bv_text = self.bubble_values.text().strip()
        if bv_text:
            base["bubbleValues"] = parse_csv_or_range(bv_text)
        else:
            base.pop("bubbleValues", None)

        # Direction
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

        # Per-block bubbleDimensions (0 = use global)
        bdw = int(self.bubble_dim_w.value())
        bdh = int(self.bubble_dim_h.value())
        if bdw > 0 and bdh > 0:
            base["bubbleDimensions"] = [bdw, bdh]
        else:
            base.pop("bubbleDimensions", None)

        self.changed.emit(self.name)

    def sync_from_model(self):
        self._updating = True
        try:
            fb_base = self.model.get_block_base(self.name)
            self.bubble_values.setText(to_csv(fb_base.get("bubbleValues", [])))
            ft = fb_base.get("fieldType", "")
            self.field_type.setCurrentText(ft if ft else "(custom)")
            self.direction.setCurrentText(
                fb_base.get("direction", self.model.default_dir_for_type(ft))
            )
            self.field_labels.setText(to_csv(fb_base.get("fieldLabels", [])))
            self.labels_gap.setValue(int(fb_base.get("labelsGap", 60)))
            self.bubbles_gap.setValue(int(fb_base.get("bubblesGap", 120)))
            ox, oy = fb_base.get("origin", [0, 0])
            self.origin_x.setValue(int(ox))
            self.origin_y.setValue(int(oy))
            bdims = fb_base.get("bubbleDimensions", [])
            self.bubble_dim_w.setValue(
                int(bdims[0]) if isinstance(bdims, list) and bdims else 0
            )
            self.bubble_dim_h.setValue(
                int(bdims[1]) if isinstance(bdims, list) and len(bdims) > 1 else 0
            )
        finally:
            self._updating = False

    def set_origin(self, x: int, y: int) -> None:
        """Called from canvas pick-origin mode."""
        self._updating = True
        self.origin_x.setValue(x)
        self.origin_y.setValue(y)
        self._updating = False
        self._apply()

    def _delete_self(self):
        self.model.remove_block(self.name)
