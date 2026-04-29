"""TemplateSettingsPanel — edit pageDimensions, bubbleDimensions, customLabels, outputColumns."""
from typing import TYPE_CHECKING

from PyQt6 import QtWidgets

if TYPE_CHECKING:
    from ..models.template_model import TemplateModel


class TemplateSettingsPanel(QtWidgets.QWidget):
    """
    Sidebar panel for global template properties:
      • pageDimensions (W × H) with auto-detect
      • bubbleDimensions (W × H)
      • customLabels table (label name → comma-separated field names)
      • outputColumns list with drag-to-reorder
    """

    def __init__(self, model: "TemplateModel", parent=None):
        super().__init__(parent)
        self.model = model
        self._updating = False

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        # ---- pageDimensions ----
        pg_box = QtWidgets.QGroupBox("Page Dimensions (pixels)")
        pg_form = QtWidgets.QFormLayout(pg_box)
        pg_form.setSpacing(4)
        pd = model.get_page_dimensions()
        self.page_w = QtWidgets.QSpinBox()
        self.page_w.setRange(0, 20000)
        self.page_w.setValue(int(pd[0]) if pd else 0)
        self.page_h = QtWidgets.QSpinBox()
        self.page_h.setRange(0, 20000)
        self.page_h.setValue(int(pd[1]) if len(pd) > 1 else 0)
        pw_row = QtWidgets.QHBoxLayout()
        pw_row.addWidget(QtWidgets.QLabel("W"))
        pw_row.addWidget(self.page_w)
        pw_row.addWidget(QtWidgets.QLabel("H"))
        pw_row.addWidget(self.page_h)
        pg_form.addRow("Size", pw_row)
        auto_btn = QtWidgets.QPushButton("Auto-detect from image")
        auto_btn.clicked.connect(self._auto_detect_page_dims)
        pg_form.addRow(auto_btn)
        root.addWidget(pg_box)

        # ---- bubbleDimensions ----
        bd_box = QtWidgets.QGroupBox("Default Bubble Dimensions (pixels)")
        bd_form = QtWidgets.QFormLayout(bd_box)
        bd_form.setSpacing(4)
        bud = model.get_bubble_dimensions()
        self.bub_w = QtWidgets.QSpinBox()
        self.bub_w.setRange(1, 500)
        self.bub_w.setValue(int(bud[0]) if bud else 32)
        self.bub_h = QtWidgets.QSpinBox()
        self.bub_h.setRange(1, 500)
        self.bub_h.setValue(int(bud[1]) if len(bud) > 1 else 32)
        bw_row = QtWidgets.QHBoxLayout()
        bw_row.addWidget(QtWidgets.QLabel("W"))
        bw_row.addWidget(self.bub_w)
        bw_row.addWidget(QtWidgets.QLabel("H"))
        bw_row.addWidget(self.bub_h)
        bd_form.addRow("Size", bw_row)
        root.addWidget(bd_box)

        # ---- customLabels ----
        cl_box = QtWidgets.QGroupBox("Custom Labels")
        cl_vbox = QtWidgets.QVBoxLayout(cl_box)
        cl_hint = QtWidgets.QLabel(
            "Map a label name → comma-separated field names for grouped scoring."
        )
        cl_hint.setWordWrap(True)
        cl_hint.setStyleSheet("color: #aaa; font-size: 11px;")
        cl_vbox.addWidget(cl_hint)
        self.custom_labels_table = QtWidgets.QTableWidget(0, 2)
        self.custom_labels_table.setHorizontalHeaderLabels(
            ["Label Name", "Fields (CSV)"]
        )
        self.custom_labels_table.horizontalHeader().setStretchLastSection(True)
        self.custom_labels_table.setMinimumHeight(100)
        cl_vbox.addWidget(self.custom_labels_table)
        cl_btn_row = QtWidgets.QHBoxLayout()
        cl_add = QtWidgets.QPushButton("+ Add")
        cl_add.clicked.connect(self._add_custom_label_row)
        cl_del = QtWidgets.QPushButton("− Remove")
        cl_del.clicked.connect(self._del_custom_label_row)
        cl_btn_row.addWidget(cl_add)
        cl_btn_row.addWidget(cl_del)
        cl_btn_row.addStretch()
        cl_vbox.addLayout(cl_btn_row)
        root.addWidget(cl_box)
        self._load_custom_labels()

        # ---- outputColumns ----
        oc_box = QtWidgets.QGroupBox("Output Columns (CSV column order)")
        oc_vbox = QtWidgets.QVBoxLayout(oc_box)
        oc_hint = QtWidgets.QLabel(
            "Drag rows to reorder. These become the CSV header column order."
        )
        oc_hint.setWordWrap(True)
        oc_hint.setStyleSheet("color: #aaa; font-size: 11px;")
        oc_vbox.addWidget(oc_hint)
        self.output_cols_list = QtWidgets.QListWidget()
        self.output_cols_list.setDragDropMode(
            QtWidgets.QAbstractItemView.DragDropMode.InternalMove
        )
        self.output_cols_list.setMinimumHeight(80)
        oc_vbox.addWidget(self.output_cols_list)
        oc_btn_row = QtWidgets.QHBoxLayout()
        oc_add = QtWidgets.QPushButton("+ Add")
        oc_add.clicked.connect(self._add_output_col)
        oc_del = QtWidgets.QPushButton("− Remove")
        oc_del.clicked.connect(self._del_output_col)
        oc_btn_row.addWidget(oc_add)
        oc_btn_row.addWidget(oc_del)
        oc_btn_row.addStretch()
        oc_vbox.addLayout(oc_btn_row)
        root.addWidget(oc_box)
        self._load_output_columns()

        # Apply button
        apply_btn = QtWidgets.QPushButton("Apply Changes")
        apply_btn.setStyleSheet(
            "QPushButton { background-color: #2a82da; color: white; font-weight: bold; padding: 4px 12px; }"
        )
        apply_btn.clicked.connect(self._apply)
        root.addWidget(apply_btn)
        root.addStretch()

        # Sync on model change
        model.changed.connect(self._sync_from_model)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_custom_labels(self):
        self.custom_labels_table.setRowCount(0)
        for k, v in self.model.get_custom_labels().items():
            row = self.custom_labels_table.rowCount()
            self.custom_labels_table.insertRow(row)
            self.custom_labels_table.setItem(row, 0, QtWidgets.QTableWidgetItem(k))
            val_str = ",".join(v) if isinstance(v, list) else str(v)
            self.custom_labels_table.setItem(
                row, 1, QtWidgets.QTableWidgetItem(val_str)
            )

    def _load_output_columns(self):
        self.output_cols_list.clear()
        for col in self.model.get_output_columns():
            self.output_cols_list.addItem(col)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _add_custom_label_row(self):
        row = self.custom_labels_table.rowCount()
        self.custom_labels_table.insertRow(row)
        self.custom_labels_table.setItem(row, 0, QtWidgets.QTableWidgetItem("label"))
        self.custom_labels_table.setItem(row, 1, QtWidgets.QTableWidgetItem(""))

    def _del_custom_label_row(self):
        row = self.custom_labels_table.currentRow()
        if row >= 0:
            self.custom_labels_table.removeRow(row)

    def _add_output_col(self):
        text, ok = QtWidgets.QInputDialog.getText(self, "Add Column", "Column name:")
        if ok and text.strip():
            self.output_cols_list.addItem(text.strip())

    def _del_output_col(self):
        row = self.output_cols_list.currentRow()
        if row >= 0:
            self.output_cols_list.takeItem(row)

    def _auto_detect_page_dims(self):
        import cv2

        from ..common import find_first_image_under

        folder = self.model.template_path.parent
        img_path = find_first_image_under(folder)
        if img_path is None and self.model.image_path:
            img_path = self.model.image_path
        if img_path is None:
            QtWidgets.QMessageBox.warning(
                self, "No Image", "No image found to auto-detect dimensions."
            )
            return
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError("cv2.imread returned None")
            h, w = img.shape[:2]
            self._updating = True
            self.page_w.setValue(int(w))
            self.page_h.setValue(int(h))
            self._updating = False
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Could not read image: {e}")

    # ------------------------------------------------------------------
    # Apply / sync
    # ------------------------------------------------------------------

    def _apply(self):
        # pageDimensions
        self.model.set_page_dimensions(self.page_w.value(), self.page_h.value())
        # bubbleDimensions
        self.model.set_bubble_dimensions(self.bub_w.value(), self.bub_h.value())
        # customLabels
        labels = {}
        for row in range(self.custom_labels_table.rowCount()):
            k_item = self.custom_labels_table.item(row, 0)
            v_item = self.custom_labels_table.item(row, 1)
            if k_item and k_item.text().strip():
                v_str = v_item.text().strip() if v_item else ""
                labels[k_item.text().strip()] = [
                    x.strip() for x in v_str.split(",") if x.strip()
                ]
        self.model.set_custom_labels(labels)
        # outputColumns
        cols = [
            self.output_cols_list.item(i).text()
            for i in range(self.output_cols_list.count())
            if self.output_cols_list.item(i).text().strip()
        ]
        self.model.set_output_columns(cols)

    def _sync_from_model(self):
        if self._updating:
            return
        self._updating = True
        try:
            pd = self.model.get_page_dimensions()
            self.page_w.setValue(int(pd[0]) if pd else 0)
            self.page_h.setValue(int(pd[1]) if len(pd) > 1 else 0)
            bud = self.model.get_bubble_dimensions()
            self.bub_w.setValue(int(bud[0]) if bud else 32)
            self.bub_h.setValue(int(bud[1]) if len(bud) > 1 else 32)
            self._load_custom_labels()
            self._load_output_columns()
        finally:
            self._updating = False
