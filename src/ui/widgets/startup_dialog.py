"""StartupDialog — select or browse for an input folder on launch."""
from pathlib import Path
from typing import Optional, Tuple

from PyQt6 import QtCore, QtWidgets

from ..common import (
    PROJECT_ROOT,
    add_recent_folder,
    find_all_images_under,
    find_first_template_under,
    get_recent_folders,
)


class StartupDialog(QtWidgets.QDialog):
    """
    Dialog shown at startup (when no CLI arg is given).
    User selects an input folder; the dialog scans it for template.json
    and images, then returns the template path + a representative image path.
    Has Recent Folders list and a "New Template…" option.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OMR Template Editor — Open")
        self.setMinimumSize(600, 460)
        self.resize(700, 520)

        self._template_path: Optional[Path] = None
        self._image_path: Optional[Path] = None
        self._new_template = False  # user selected "New Template"

        # --- Layout ---
        root = QtWidgets.QVBoxLayout(self)

        # Folder row
        folder_row = QtWidgets.QHBoxLayout()
        self._folder_edit = QtWidgets.QLineEdit()
        self._folder_edit.setPlaceholderText(
            "Select a folder containing template.json …"
        )
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        folder_row.addWidget(self._folder_edit)
        folder_row.addWidget(browse_btn)
        root.addLayout(folder_row)

        # Recent folders
        recent_label = QtWidgets.QLabel("Recent folders:")
        root.addWidget(recent_label)
        self._recent_list = QtWidgets.QListWidget()
        self._recent_list.setMaximumHeight(140)
        self._recent_list.itemDoubleClicked.connect(self._on_recent_double_click)
        self._recent_list.itemClicked.connect(self._on_recent_click)
        root.addWidget(self._recent_list)
        self._populate_recent()

        # Scan result info box
        self._info = QtWidgets.QTextEdit()
        self._info.setReadOnly(True)
        self._info.setMaximumHeight(120)
        self._info.setPlaceholderText("Folder scan results will appear here…")
        root.addWidget(self._info)

        # Buttons row
        btn_row = QtWidgets.QHBoxLayout()
        self._open_btn = QtWidgets.QPushButton("Open")
        self._open_btn.setDefault(True)
        self._open_btn.setEnabled(False)
        self._open_btn.clicked.connect(self._on_open)
        new_btn = QtWidgets.QPushButton("New Template…")
        new_btn.clicked.connect(self._on_new_template)
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(new_btn)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(self._open_btn)
        root.addLayout(btn_row)

        # Wire folder edit
        self._folder_edit.textChanged.connect(self._on_folder_text_changed)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def template_path(self) -> Optional[Path]:
        return self._template_path

    @property
    def image_path(self) -> Optional[Path]:
        return self._image_path

    @property
    def want_new_template(self) -> bool:
        return self._new_template

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _populate_recent(self):
        self._recent_list.clear()
        for p in get_recent_folders():
            item = QtWidgets.QListWidgetItem(str(p))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, p)
            self._recent_list.addItem(item)

    def _browse(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Input Folder",
            str(PROJECT_ROOT / "inputs"),
            QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if folder:
            self._folder_edit.setText(folder)

    def _on_folder_text_changed(self, text: str):
        self._scan_folder(Path(text.strip()) if text.strip() else None)

    def _on_recent_click(self, item: QtWidgets.QListWidgetItem):
        p: Path = item.data(QtCore.Qt.ItemDataRole.UserRole)
        self._folder_edit.setText(str(p))

    def _on_recent_double_click(self, item: QtWidgets.QListWidgetItem):
        p: Path = item.data(QtCore.Qt.ItemDataRole.UserRole)
        self._folder_edit.setText(str(p))
        if self._template_path:
            self._on_open()

    def _scan_folder(self, folder: Optional[Path]):
        self._template_path = None
        self._image_path = None
        self._open_btn.setEnabled(False)
        if folder is None or not folder.exists():
            self._info.setPlainText("")
            return

        lines = [f"📁 {folder}"]

        tmpl = find_first_template_under(folder)
        if tmpl:
            lines.append(f"✅ template.json: {tmpl.relative_to(folder)}")
            self._template_path = tmpl
        else:
            lines.append("⚠️  No template.json found — you can create a new one.")

        images = find_all_images_under(folder)
        if images:
            lines.append(f"🖼  {len(images)} image(s) found: ")
            for img in images[:5]:
                lines.append(f"      {img.relative_to(folder)}")
            if len(images) > 5:
                lines.append(f"      … and {len(images) - 5} more")
            self._image_path = images[0]
        else:
            lines.append("⚠️  No images found in this folder.")

        self._info.setPlainText("\n".join(lines))
        # Allow opening even without template (user will create one)
        self._open_btn.setEnabled(True)

    def _on_open(self):
        folder_text = self._folder_edit.text().strip()
        if not folder_text:
            return
        folder = Path(folder_text)
        add_recent_folder(folder)
        self.accept()

    def _on_new_template(self):
        """User wants a brand-new template in a chosen folder."""
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Choose folder for new template",
            str(PROJECT_ROOT / "inputs"),
            QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if not folder:
            return
        p = Path(folder)
        self._template_path = p / "template.json"
        images = find_all_images_under(p)
        self._image_path = images[0] if images else None
        self._new_template = True
        add_recent_folder(p)
        self.accept()

    # ------------------------------------------------------------------
    # Static factory
    # ------------------------------------------------------------------

    @staticmethod
    def run(parent=None) -> Tuple[Optional[Path], Optional[Path], bool]:
        """
        Show the dialog and return (template_path, image_path, want_new).
        Returns (None, None, False) if user cancelled.
        """
        dlg = StartupDialog(parent)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            return dlg.template_path, dlg.image_path, dlg.want_new_template
        return None, None, False
