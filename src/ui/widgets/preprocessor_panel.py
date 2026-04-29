"""PreprocessorPanel — edit and reorder the preProcessors pipeline."""
import copy
from typing import TYPE_CHECKING, Any, Dict, Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from ..common import PROCESSOR_DESCRIPTIONS, PROCESSOR_TYPES

if TYPE_CHECKING:
    from ..models.template_model import TemplateModel


# ---------------------------------------------------------------------------
# Per-processor property forms
# ---------------------------------------------------------------------------


class _CropPageForm(QtWidgets.QWidget):
    name = "CropPage"

    def __init__(self, options: Dict[str, Any], parent=None):
        super().__init__(parent)
        form = QtWidgets.QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        mk = options.get("morphKernel", [10, 10])
        self.mk_w = QtWidgets.QSpinBox()
        self.mk_w.setRange(1, 200)
        self.mk_w.setValue(int(mk[0]))
        self.mk_h = QtWidgets.QSpinBox()
        self.mk_h.setRange(1, 200)
        self.mk_h.setValue(int(mk[1]) if len(mk) > 1 else int(mk[0]))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("W"))
        row.addWidget(self.mk_w)
        row.addWidget(QtWidgets.QLabel("H"))
        row.addWidget(self.mk_h)
        form.addRow("morphKernel", row)

    def get_options(self) -> Dict[str, Any]:
        return {"morphKernel": [self.mk_w.value(), self.mk_h.value()]}


class _CropOnMarkersForm(QtWidgets.QWidget):
    name = "CropOnMarkers"

    def __init__(self, options: Dict[str, Any], template_dir, parent=None):
        super().__init__(parent)
        self._template_dir = template_dir
        form = QtWidgets.QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)

        # relativePath
        rp_row = QtWidgets.QHBoxLayout()
        self.relative_path = QtWidgets.QLineEdit(
            options.get("relativePath", "omr_marker.jpg")
        )
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.setMaximumWidth(70)
        browse_btn.clicked.connect(self._browse_marker)
        rp_row.addWidget(self.relative_path)
        rp_row.addWidget(browse_btn)
        form.addRow("Marker file", rp_row)

        # sheetToMarkerWidthRatio
        self.sheet_ratio = QtWidgets.QDoubleSpinBox()
        self.sheet_ratio.setRange(0.01, 1.0)
        self.sheet_ratio.setSingleStep(0.01)
        self.sheet_ratio.setDecimals(3)
        self.sheet_ratio.setValue(options.get("sheetToMarkerWidthRatio", 0.05))
        form.addRow("sheetToMarkerWidthRatio", self.sheet_ratio)

        # min_matching_threshold
        self.min_thresh = QtWidgets.QDoubleSpinBox()
        self.min_thresh.setRange(0.0, 1.0)
        self.min_thresh.setSingleStep(0.01)
        self.min_thresh.setDecimals(3)
        self.min_thresh.setValue(options.get("min_matching_threshold", 0.3))
        form.addRow("min_matching_threshold", self.min_thresh)

        # max_matching_variation
        self.max_var = QtWidgets.QDoubleSpinBox()
        self.max_var.setRange(0.0, 1.0)
        self.max_var.setSingleStep(0.01)
        self.max_var.setDecimals(3)
        self.max_var.setValue(options.get("max_matching_variation", 0.41))
        form.addRow("max_matching_variation", self.max_var)

        # marker_rescale_range
        rescale = options.get("marker_rescale_range", [35, 100])
        self.rescale_min = QtWidgets.QSpinBox()
        self.rescale_min.setRange(1, 200)
        self.rescale_min.setValue(int(rescale[0]))
        self.rescale_max = QtWidgets.QSpinBox()
        self.rescale_max.setRange(1, 200)
        self.rescale_max.setValue(int(rescale[1]) if len(rescale) > 1 else 100)
        rscl_row = QtWidgets.QHBoxLayout()
        rscl_row.addWidget(QtWidgets.QLabel("min%"))
        rscl_row.addWidget(self.rescale_min)
        rscl_row.addWidget(QtWidgets.QLabel("max%"))
        rscl_row.addWidget(self.rescale_max)
        form.addRow("marker_rescale_range", rscl_row)

        # marker_rescale_steps
        self.rescale_steps = QtWidgets.QSpinBox()
        self.rescale_steps.setRange(1, 100)
        self.rescale_steps.setValue(int(options.get("marker_rescale_steps", 10)))
        form.addRow("rescale_steps", self.rescale_steps)

        # apply_erode_subtract
        self.erode_sub = QtWidgets.QCheckBox()
        self.erode_sub.setChecked(bool(options.get("apply_erode_subtract", True)))
        form.addRow("apply_erode_subtract", self.erode_sub)

    def _browse_marker(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Marker Image",
            str(self._template_dir),
            "Images (*.jpg *.jpeg *.png *.bmp)",
            options=QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            from pathlib import Path

            try:
                rel = Path(path).relative_to(self._template_dir)
                self.relative_path.setText(str(rel))
            except ValueError:
                self.relative_path.setText(path)

    def get_options(self) -> Dict[str, Any]:
        opts = {
            "relativePath": self.relative_path.text().strip(),
            "min_matching_threshold": self.min_thresh.value(),
            "max_matching_variation": self.max_var.value(),
            "marker_rescale_range": [
                self.rescale_min.value(),
                self.rescale_max.value(),
            ],
            "marker_rescale_steps": self.rescale_steps.value(),
            "apply_erode_subtract": self.erode_sub.isChecked(),
        }
        ratio = self.sheet_ratio.value()
        if ratio != 0.05:
            opts["sheetToMarkerWidthRatio"] = ratio
        return opts


class _FeatureAlignmentForm(QtWidgets.QWidget):
    name = "FeatureBasedAlignment"

    def __init__(self, options: Dict[str, Any], template_dir, parent=None):
        super().__init__(parent)
        self._template_dir = template_dir
        form = QtWidgets.QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)

        # reference path
        ref_row = QtWidgets.QHBoxLayout()
        self.reference = QtWidgets.QLineEdit(options.get("reference", ""))
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.setMaximumWidth(70)
        browse_btn.clicked.connect(self._browse_reference)
        ref_row.addWidget(self.reference)
        ref_row.addWidget(browse_btn)
        form.addRow("reference", ref_row)

        # maxFeatures
        self.max_features = QtWidgets.QSpinBox()
        self.max_features.setRange(10, 10000)
        self.max_features.setValue(int(options.get("maxFeatures", 500)))
        form.addRow("maxFeatures", self.max_features)

        # goodMatchPercent
        self.good_match = QtWidgets.QDoubleSpinBox()
        self.good_match.setRange(0.01, 1.0)
        self.good_match.setSingleStep(0.01)
        self.good_match.setDecimals(3)
        self.good_match.setValue(options.get("goodMatchPercent", 0.15))
        form.addRow("goodMatchPercent", self.good_match)

        # 2d (affine vs homography)
        self.use_2d = QtWidgets.QCheckBox("Use 2D affine (vs full homography)")
        self.use_2d.setChecked(bool(options.get("2d", False)))
        form.addRow("", self.use_2d)

    def _browse_reference(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Reference Image",
            str(self._template_dir),
            "Images (*.jpg *.jpeg *.png *.bmp)",
            options=QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            from pathlib import Path

            try:
                rel = Path(path).relative_to(self._template_dir)
                self.reference.setText(str(rel))
            except ValueError:
                self.reference.setText(path)

    def get_options(self) -> Dict[str, Any]:
        opts = {
            "reference": self.reference.text().strip(),
            "maxFeatures": self.max_features.value(),
            "goodMatchPercent": self.good_match.value(),
        }
        if self.use_2d.isChecked():
            opts["2d"] = True
        return opts


class _GaussianBlurForm(QtWidgets.QWidget):
    name = "GaussianBlur"

    def __init__(self, options: Dict[str, Any], parent=None):
        super().__init__(parent)
        form = QtWidgets.QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        ks = options.get("kSize", [3, 3])
        self.ks_w = QtWidgets.QSpinBox()
        self.ks_w.setRange(1, 99)
        self.ks_w.setSingleStep(2)  # must be odd
        self.ks_w.setValue(int(ks[0]))
        self.ks_h = QtWidgets.QSpinBox()
        self.ks_h.setRange(1, 99)
        self.ks_h.setSingleStep(2)
        self.ks_h.setValue(int(ks[1]) if len(ks) > 1 else int(ks[0]))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("W"))
        row.addWidget(self.ks_w)
        row.addWidget(QtWidgets.QLabel("H"))
        row.addWidget(self.ks_h)
        form.addRow("kSize", row)
        self.sigma_x = QtWidgets.QDoubleSpinBox()
        self.sigma_x.setRange(0.0, 50.0)
        self.sigma_x.setValue(options.get("sigmaX", 0.0))
        form.addRow("sigmaX", self.sigma_x)

    def get_options(self) -> Dict[str, Any]:
        return {
            "kSize": [self.ks_w.value(), self.ks_h.value()],
            "sigmaX": self.sigma_x.value(),
        }


class _MedianBlurForm(QtWidgets.QWidget):
    name = "MedianBlur"

    def __init__(self, options: Dict[str, Any], parent=None):
        super().__init__(parent)
        form = QtWidgets.QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        self.ks = QtWidgets.QSpinBox()
        self.ks.setRange(1, 99)
        self.ks.setSingleStep(2)
        self.ks.setValue(int(options.get("kSize", 3)))
        form.addRow("kSize (odd)", self.ks)

    def get_options(self) -> Dict[str, Any]:
        v = self.ks.value()
        if v % 2 == 0:
            v += 1
        return {"kSize": v}


class _LevelsForm(QtWidgets.QWidget):
    name = "Levels"

    def __init__(self, options: Dict[str, Any], parent=None):
        super().__init__(parent)
        form = QtWidgets.QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        self.low = QtWidgets.QDoubleSpinBox()
        self.low.setRange(0.0, 1.0)
        self.low.setSingleStep(0.01)
        self.low.setValue(options.get("low", 0.0))
        self.high = QtWidgets.QDoubleSpinBox()
        self.high.setRange(0.0, 1.0)
        self.high.setSingleStep(0.01)
        self.high.setValue(options.get("high", 1.0))
        self.gamma = QtWidgets.QDoubleSpinBox()
        self.gamma.setRange(0.01, 5.0)
        self.gamma.setSingleStep(0.05)
        self.gamma.setValue(options.get("gamma", 1.0))
        form.addRow("low (0–1)", self.low)
        form.addRow("high (0–1)", self.high)
        form.addRow("gamma (>1 = brighter)", self.gamma)

    def get_options(self) -> Dict[str, Any]:
        return {
            "low": self.low.value(),
            "high": self.high.value(),
            "gamma": self.gamma.value(),
        }


_FORM_CLASSES = {
    "CropPage": lambda opts, td: _CropPageForm(opts),
    "CropOnMarkers": lambda opts, td: _CropOnMarkersForm(opts, td),
    "FeatureBasedAlignment": lambda opts, td: _FeatureAlignmentForm(opts, td),
    "GaussianBlur": lambda opts, td: _GaussianBlurForm(opts),
    "MedianBlur": lambda opts, td: _MedianBlurForm(opts),
    "Levels": lambda opts, td: _LevelsForm(opts),
}


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------


class PreprocessorPanel(QtWidgets.QWidget):
    """
    Sidebar panel for editing the preProcessors list.
    Shows a drag-reorderable list and a per-processor property form.
    Emits 'changed' when the model is updated.
    """

    changed = QtCore.pyqtSignal()

    def __init__(self, model: "TemplateModel", parent=None):
        super().__init__(parent)
        self.model = model
        self._form_widget: Optional[QtWidgets.QWidget] = None

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        hint = QtWidgets.QLabel(
            "Add preprocessors that transform images before bubble detection.\n"
            "Drag to reorder. Select to edit options."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #aaa; font-size: 11px;")
        root.addWidget(hint)

        # List + buttons
        list_row = QtWidgets.QHBoxLayout()
        self._list = QtWidgets.QListWidget()
        self._list.setDragDropMode(
            QtWidgets.QAbstractItemView.DragDropMode.InternalMove
        )
        self._list.setMinimumHeight(120)
        list_row.addWidget(self._list, 3)

        btn_col = QtWidgets.QVBoxLayout()
        add_btn = QtWidgets.QPushButton("+ Add")
        add_btn.clicked.connect(self._add_processor)
        del_btn = QtWidgets.QPushButton("− Remove")
        del_btn.clicked.connect(self._remove_processor)
        up_btn = QtWidgets.QPushButton("▲")
        up_btn.clicked.connect(self._move_up)
        dn_btn = QtWidgets.QPushButton("▼")
        dn_btn.clicked.connect(self._move_down)
        btn_col.addWidget(add_btn)
        btn_col.addWidget(del_btn)
        btn_col.addStretch()
        btn_col.addWidget(up_btn)
        btn_col.addWidget(dn_btn)
        list_row.addLayout(btn_col)
        root.addLayout(list_row)

        # Description label
        self._desc_label = QtWidgets.QLabel()
        self._desc_label.setWordWrap(True)
        self._desc_label.setStyleSheet(
            "color: #8ecfff; font-size: 11px; padding: 4px; "
            "background: #2a2a3a; border-radius: 4px;"
        )
        self._desc_label.setVisible(False)
        root.addWidget(self._desc_label)

        # Options group (dynamic per-processor form)
        self._options_group = QtWidgets.QGroupBox("Options")
        self._options_group.setVisible(False)
        options_vbox = QtWidgets.QVBoxLayout(self._options_group)
        options_vbox.setContentsMargins(4, 4, 4, 4)
        self._options_container = QtWidgets.QWidget()
        options_vbox.addWidget(self._options_container)
        # Apply options button
        apply_opts_btn = QtWidgets.QPushButton("Apply Options")
        apply_opts_btn.clicked.connect(self._apply_current_options)
        options_vbox.addWidget(apply_opts_btn)
        root.addWidget(self._options_group)

        root.addStretch()

        # Wire selection change
        self._list.currentRowChanged.connect(self._on_selection_changed)
        self._list.model().rowsMoved.connect(self._on_rows_moved)

        # Initial load
        self._load_from_model()
        model.changed.connect(self._on_model_changed)

    # ------------------------------------------------------------------
    # List management
    # ------------------------------------------------------------------

    def _load_from_model(self):
        self._list.blockSignals(True)
        self._list.clear()
        for proc in self.model.get_preprocessors():
            self._list.addItem(proc.get("name", "?"))
        self._list.blockSignals(False)
        self._options_group.setVisible(False)
        self._desc_label.setVisible(False)
        self._form_widget = None

    def _on_model_changed(self):
        # Avoid recursive reload — only reload list names, preserve selection
        row = self._list.currentRow()
        procs = self.model.get_preprocessors()
        if self._list.count() == len(procs):
            for i, proc in enumerate(procs):
                item = self._list.item(i)
                if item:
                    item.setText(proc.get("name", "?"))
        else:
            self._load_from_model()
        if row >= 0 and row < self._list.count():
            self._list.setCurrentRow(row)

    def _on_rows_moved(self, *_):
        """List was reordered by drag — sync to model."""
        names = [self._list.item(i).text() for i in range(self._list.count())]
        procs = {p.get("name"): p for p in self.model.get_preprocessors()}
        new_order = [procs[n] for n in names if n in procs]
        self.model.set_preprocessors(new_order)
        self.changed.emit()

    def _add_processor(self):
        menu = QtWidgets.QMenu(self)
        for ptype in PROCESSOR_TYPES:
            action = menu.addAction(ptype)
            action.setToolTip(PROCESSOR_DESCRIPTIONS.get(ptype, ""))
        chosen = menu.exec(QtGui.QCursor.pos())
        if chosen is None:
            return
        ptype = chosen.text()
        procs = copy.deepcopy(self.model.get_preprocessors())
        # Build minimal default options
        defaults: Dict[str, Any] = {}
        if ptype == "CropOnMarkers":
            defaults = {"relativePath": "omr_marker.jpg"}
        elif ptype == "FeatureBasedAlignment":
            defaults = {"reference": "reference.jpg"}
        elif ptype == "GaussianBlur":
            defaults = {"kSize": [3, 3], "sigmaX": 0}
        elif ptype == "MedianBlur":
            defaults = {"kSize": 3}
        elif ptype == "Levels":
            defaults = {"low": 0, "high": 1, "gamma": 1}
        elif ptype == "CropPage":
            defaults = {"morphKernel": [10, 10]}
        procs.append({"name": ptype, "options": defaults})
        self.model.set_preprocessors(procs)
        self._load_from_model()
        self._list.setCurrentRow(self._list.count() - 1)
        self.changed.emit()

    def _remove_processor(self):
        row = self._list.currentRow()
        if row < 0:
            return
        procs = copy.deepcopy(self.model.get_preprocessors())
        if row < len(procs):
            del procs[row]
            self.model.set_preprocessors(procs)
            self._load_from_model()
            self.changed.emit()

    def _move_up(self):
        row = self._list.currentRow()
        if row <= 0:
            return
        procs = copy.deepcopy(self.model.get_preprocessors())
        procs[row - 1], procs[row] = procs[row], procs[row - 1]
        self.model.set_preprocessors(procs)
        self._load_from_model()
        self._list.setCurrentRow(row - 1)
        self.changed.emit()

    def _move_down(self):
        row = self._list.currentRow()
        procs = copy.deepcopy(self.model.get_preprocessors())
        if row < 0 or row >= len(procs) - 1:
            return
        procs[row], procs[row + 1] = procs[row + 1], procs[row]
        self.model.set_preprocessors(procs)
        self._load_from_model()
        self._list.setCurrentRow(row + 1)
        self.changed.emit()

    # ------------------------------------------------------------------
    # Options form
    # ------------------------------------------------------------------

    def _on_selection_changed(self, row: int):
        # Clear old form
        if self._form_widget:
            layout = self._options_container.layout()
            if layout:
                while layout.count():
                    w = layout.takeAt(0).widget()
                    if w:
                        w.deleteLater()
            self._form_widget = None

        procs = self.model.get_preprocessors()
        if row < 0 or row >= len(procs):
            self._options_group.setVisible(False)
            self._desc_label.setVisible(False)
            return

        proc = procs[row]
        pname = proc.get("name", "")
        options = proc.get("options", {})
        template_dir = self.model.template_path.parent

        # Description
        desc = PROCESSOR_DESCRIPTIONS.get(pname, "")
        if desc:
            self._desc_label.setText(desc)
            self._desc_label.setVisible(True)
        else:
            self._desc_label.setVisible(False)

        builder = _FORM_CLASSES.get(pname)
        if builder:
            form_widget = builder(options, template_dir)
            self._form_widget = form_widget
            layout = self._options_container.layout()
            if layout is None:
                layout = QtWidgets.QVBoxLayout(self._options_container)
                layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(form_widget)
            self._options_group.setTitle(f"{pname} options")
            self._options_group.setVisible(True)
        else:
            self._options_group.setVisible(False)

    def _apply_current_options(self):
        if self._form_widget is None:
            return
        row = self._list.currentRow()
        procs = copy.deepcopy(self.model.get_preprocessors())
        if row < 0 or row >= len(procs):
            return
        procs[row]["options"] = self._form_widget.get_options()
        self.model.set_preprocessors(procs)
        self.changed.emit()
