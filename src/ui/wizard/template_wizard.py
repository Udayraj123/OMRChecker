"""TemplateWizard — QWizard to create a template.json from scratch."""
import json
from pathlib import Path
from typing import Optional

import cv2
from PyQt6 import QtCore, QtGui, QtWidgets

from ..common import find_all_images_under, find_first_image_under, np_to_qpixmap

# ---------------------------------------------------------------------------
# Page 0 — Welcome
# ---------------------------------------------------------------------------


class _WelcomePage(QtWidgets.QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Welcome to the OMR Template Wizard")
        self.setSubTitle(
            "This wizard helps you create a template.json from scratch, "
            "step by step."
        )
        vbox = QtWidgets.QVBoxLayout(self)
        vbox.addWidget(
            QtWidgets.QLabel(
                "<b>A template.json describes:</b><br>"
                "• the expected page dimensions (W × H)<br>"
                "• the bubble size used on the sheet<br>"
                "• which preprocessing step aligns / crops the image<br>"
                "• where each field block of bubbles is located<br><br>"
                "Click <b>Next</b> to begin."
            )
        )


# ---------------------------------------------------------------------------
# Page 1 — Input folder
# ---------------------------------------------------------------------------


class _FolderPage(QtWidgets.QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Input Folder")
        self.setSubTitle("Select the folder that contains the scanned images.")
        vbox = QtWidgets.QVBoxLayout(self)

        row = QtWidgets.QHBoxLayout()
        self._folder_edit = QtWidgets.QLineEdit()
        self._folder_edit.setPlaceholderText("Path to folder with images…")
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        row.addWidget(self._folder_edit)
        row.addWidget(browse_btn)
        vbox.addLayout(row)

        self._info = QtWidgets.QLabel()
        self._info.setWordWrap(True)
        self._info.setStyleSheet("color: #aaa;")
        vbox.addWidget(self._info)

        self.registerField("inputFolder*", self._folder_edit)
        self._folder_edit.textChanged.connect(self._scan)

    def _browse(self):
        from ..common import PROJECT_ROOT

        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Folder",
            str(PROJECT_ROOT / "inputs"),
            QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if folder:
            self._folder_edit.setText(folder)

    def _scan(self, text: str):
        p = Path(text.strip() or ".")
        imgs = find_all_images_under(p)
        if imgs:
            self._info.setText(f"✅  {len(imgs)} image(s) found.")
        else:
            self._info.setText("⚠️  No images found.")
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        p = Path(self._folder_edit.text().strip() or ".")
        return bool(find_all_images_under(p))


# ---------------------------------------------------------------------------
# Page 2 — Page dimensions
# ---------------------------------------------------------------------------


class _PageDimsPage(QtWidgets.QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Page Dimensions")
        self.setSubTitle(
            "The dimensions (width × height in pixels) that all images will be "
            "resized to before processing. Usually auto-detected."
        )
        form = QtWidgets.QFormLayout(self)
        self._w = QtWidgets.QSpinBox()
        self._w.setRange(0, 20000)
        self._h = QtWidgets.QSpinBox()
        self._h.setRange(0, 20000)
        wh_row = QtWidgets.QHBoxLayout()
        wh_row.addWidget(QtWidgets.QLabel("W"))
        wh_row.addWidget(self._w)
        wh_row.addWidget(QtWidgets.QLabel("H"))
        wh_row.addWidget(self._h)
        form.addRow("Page size (px)", wh_row)
        auto_btn = QtWidgets.QPushButton("Auto-detect from first image")
        auto_btn.clicked.connect(self._auto_detect)
        form.addRow(auto_btn)
        self._preview_label = QtWidgets.QLabel()
        form.addRow(self._preview_label)
        self.registerField("pageW", self._w, "value")
        self.registerField("pageH", self._h, "value")

    def initializePage(self):
        folder = Path(self.field("inputFolder") or ".")
        imgs = find_all_images_under(folder)
        if imgs:
            self._auto_detect_from(imgs[0])

    def _auto_detect(self):
        folder = Path(self.field("inputFolder") or ".")
        img_path = find_first_image_under(folder)
        if img_path:
            self._auto_detect_from(img_path)

    def _auto_detect_from(self, img_path: Path):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            h, w = img.shape[:2]
            self._w.setValue(int(w))
            self._h.setValue(int(h))
            pix = np_to_qpixmap(img).scaled(
                300, 200, QtCore.Qt.AspectRatioMode.KeepAspectRatio
            )
            self._preview_label.setPixmap(pix)


# ---------------------------------------------------------------------------
# Page 3 — Preprocessing approach
# ---------------------------------------------------------------------------


class _PreprocessingApproachPage(QtWidgets.QWizardPage):
    _APPROACH_OPTIONS = [
        ("None", "No preprocessing — use raw image dimensions directly."),
        (
            "CropPage",
            "Edge detection + perspective crop. Best for flat, clean scans with a visible page border.",
        ),
        (
            "CropOnMarkers",
            "Corner marker detection & warp. Requires omr_marker.jpg printed at all four corners.",
        ),
        (
            "FeatureBasedAlignment",
            "ORB keypoint matching to a reference image. Best for mobile camera photos.",
        ),
    ]

    def __init__(self):
        super().__init__()
        self.setTitle("Preprocessing Approach")
        self.setSubTitle(
            "Choose how images will be aligned and cropped before bubble detection."
        )
        vbox = QtWidgets.QVBoxLayout(self)
        self._btn_group = QtWidgets.QButtonGroup(self)
        for i, (name, desc) in enumerate(self._APPROACH_OPTIONS):
            radio = QtWidgets.QRadioButton(name)
            radio.setToolTip(desc)
            if i == 0:
                radio.setChecked(True)
            self._btn_group.addButton(radio, i)
            lbl = QtWidgets.QLabel(f"  <i>{desc}</i>")  # noqa: E221
            lbl.setWordWrap(True)
            lbl.setStyleSheet("color: #aaa; font-size: 11px;")
            vbox.addWidget(radio)
            vbox.addWidget(lbl)
        self.registerField("preprocessApproach", self, "approach_value")

    @QtCore.pyqtProperty(str)
    def approach_value(self) -> str:
        checked = self._btn_group.checkedButton()
        return checked.text() if checked else "None"


# ---------------------------------------------------------------------------
# Page 4 — Configure preprocessor
# ---------------------------------------------------------------------------


class _ConfigurePreprocessorPage(QtWidgets.QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Configure Preprocessor")
        self.setSubTitle("Set options for the chosen preprocessing step.")
        self._vbox = QtWidgets.QVBoxLayout(self)
        self._form_widget = None
        self._approach = "None"

    def initializePage(self):
        approach = self.field("preprocessApproach")
        if approach == self._approach and self._form_widget:
            return  # unchanged

        # Remove old form
        if self._form_widget:
            self._vbox.removeWidget(self._form_widget)
            self._form_widget.deleteLater()
            self._form_widget = None
        self._approach = approach

        if approach == "None":
            lbl = QtWidgets.QLabel("No configuration needed.")
            lbl.setStyleSheet("color: #aaa;")
            self._vbox.addWidget(lbl)
            self._form_widget = lbl
        elif approach == "CropPage":
            from ..widgets.preprocessor_panel import _CropPageForm

            self._form_widget = _CropPageForm({})
            self._vbox.addWidget(self._form_widget)
        elif approach == "CropOnMarkers":
            folder = Path(self.field("inputFolder") or ".")
            from ..widgets.preprocessor_panel import _CropOnMarkersForm

            self._form_widget = _CropOnMarkersForm({}, folder)
            self._vbox.addWidget(self._form_widget)
        elif approach == "FeatureBasedAlignment":
            folder = Path(self.field("inputFolder") or ".")
            from ..widgets.preprocessor_panel import _FeatureAlignmentForm

            self._form_widget = _FeatureAlignmentForm({}, folder)
            self._vbox.addWidget(self._form_widget)
        # Preview hint
        preview_hint = QtWidgets.QLabel(
            "💡 After creating the template, use the <b>Preview</b> tab to "
            "verify alignment on your images."
        )
        preview_hint.setWordWrap(True)
        preview_hint.setStyleSheet("color: #8ecfff; font-size: 11px; margin-top: 8px;")
        self._vbox.addWidget(preview_hint)

    def get_preprocessors(self) -> list:
        approach = self._approach
        if approach == "None" or self._form_widget is None:
            return []
        if hasattr(self._form_widget, "get_options"):
            return [{"name": approach, "options": self._form_widget.get_options()}]
        return []


# ---------------------------------------------------------------------------
# Page 5 — Bubble dimensions
# ---------------------------------------------------------------------------


class _BubbleDimsPage(QtWidgets.QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Bubble Dimensions")
        self.setSubTitle(
            "Set the size (in pixels at page resolution) of a single bubble."
        )
        vbox = QtWidgets.QVBoxLayout(self)
        vbox.addWidget(
            QtWidgets.QLabel(
                "Measure the width and height of one bubble in the scanned image.\n"
                "If bubbles are circular, set W = H = bubble diameter.\n"
                "Zoom into the canvas to measure precisely."
            )
        )
        form = QtWidgets.QFormLayout()
        self._bw = QtWidgets.QSpinBox()
        self._bw.setRange(1, 500)
        self._bw.setValue(32)
        self._bh = QtWidgets.QSpinBox()
        self._bh.setRange(1, 500)
        self._bh.setValue(32)
        brow = QtWidgets.QHBoxLayout()
        brow.addWidget(QtWidgets.QLabel("W"))
        brow.addWidget(self._bw)
        brow.addWidget(QtWidgets.QLabel("H"))
        brow.addWidget(self._bh)
        form.addRow("Bubble size (px)", brow)
        vbox.addLayout(form)
        self.registerField("bubbleW", self._bw, "value")
        self.registerField("bubbleH", self._bh, "value")


# ---------------------------------------------------------------------------
# Page 6 — Review & Save
# ---------------------------------------------------------------------------


class _ReviewPage(QtWidgets.QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Review & Save")
        self.setSubTitle(
            "Review the generated template.json. "
            "Click Save to write it to the input folder."
        )
        self._vbox = QtWidgets.QVBoxLayout(self)
        self._preview = QtWidgets.QTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setFont(QtGui.QFont("Monospace", 10))
        self._vbox.addWidget(self._preview)
        save_btn = QtWidgets.QPushButton("💾  Save template.json")
        save_btn.clicked.connect(self._save)
        self._vbox.addWidget(save_btn)
        self._save_status = QtWidgets.QLabel()
        self._vbox.addWidget(self._save_status)

    def initializePage(self):
        self._preview.setPlainText(json.dumps(self._build_template(), indent=2))

    def _build_template(self) -> dict:
        wz = self.wizard()
        procs_page: _ConfigurePreprocessorPage = wz.page(4)
        pre_processors = procs_page.get_preprocessors()
        return {
            "pageDimensions": [wz.field("pageW"), wz.field("pageH")],
            "bubbleDimensions": [wz.field("bubbleW"), wz.field("bubbleH")],
            "preProcessors": pre_processors,
            "fieldBlocks": {},
        }

    def _save(self):
        folder = Path(self.wizard().field("inputFolder") or ".")
        out_path = folder / "template.json"
        try:
            out_path.write_text(
                json.dumps(self._build_template(), indent=2), encoding="utf-8"
            )
            self._save_status.setText(f"✅  Saved: {out_path}")
            self._save_status.setStyleSheet("color: #80ff80;")
            # Store on wizard so caller can access it
            self.wizard().setProperty("savedTemplatePath", str(out_path))
        except Exception as e:
            self._save_status.setText(f"❌  Error: {e}")
            self._save_status.setStyleSheet("color: #ff8080;")


# ---------------------------------------------------------------------------
# Wizard
# ---------------------------------------------------------------------------


class TemplateWizard(QtWidgets.QWizard):
    """8-step QWizard that creates a template.json from scratch."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Template Wizard")
        self.setWizardStyle(QtWidgets.QWizard.WizardStyle.ModernStyle)
        self.resize(700, 600)

        self.addPage(_WelcomePage())
        self.addPage(_FolderPage())
        self.addPage(_PageDimsPage())
        self.addPage(_PreprocessingApproachPage())
        self._config_page = _ConfigurePreprocessorPage()
        self.addPage(self._config_page)
        self.addPage(_BubbleDimsPage())
        self.addPage(_ReviewPage())

    def get_saved_template_path(self) -> Optional[Path]:
        v = self.property("savedTemplatePath")
        return Path(v) if v else None

    @staticmethod
    def run_wizard(parent=None) -> Optional[Path]:
        """Show wizard and return saved template path (or None)."""
        wz = TemplateWizard(parent)
        if wz.exec() == QtWidgets.QWizard.DialogCode.Accepted:
            return wz.get_saved_template_path()
        return None
