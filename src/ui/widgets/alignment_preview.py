"""AlignmentPreviewWidget — run preprocessors in a thread and show step-by-step results."""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from ..common import find_all_images_under, np_to_qpixmap

# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------


class _PreviewWorker(QtCore.QThread):
    """Runs preprocessing in a background thread. Emits stepsDone or errorOccurred."""

    stepsDone = QtCore.pyqtSignal(list)  # List[Tuple[str, np.ndarray]]
    errorOccurred = QtCore.pyqtSignal(str)

    def __init__(self, template_path: Path, image_path: Path, parent=None):
        super().__init__(parent)
        self._template_path = template_path
        self._image_path = image_path

    def run(self):
        try:
            from ..processing import run_preprocessors_stepwise

            steps = run_preprocessors_stepwise(self._template_path, self._image_path)
            self.stepsDone.emit(steps)
        except Exception as e:
            self.errorOccurred.emit(str(e))


class _FeatureMatchWorker(QtCore.QThread):
    """Runs ORB feature matching between reference and input; emits the annotated image."""

    matchDone = QtCore.pyqtSignal(
        object, object, object
    )  # ref_img, input_img, match_img
    errorOccurred = QtCore.pyqtSignal(str)

    def __init__(
        self,
        reference_path: Path,
        input_path: Path,
        max_features: int = 500,
        good_match_pct: float = 0.15,
        parent=None,
    ):
        super().__init__(parent)
        self._ref = reference_path
        self._inp = input_path
        self._max_features = max_features
        self._good_match_pct = good_match_pct

    def run(self):
        try:
            ref = cv2.imread(str(self._ref), cv2.IMREAD_GRAYSCALE)
            inp = cv2.imread(str(self._inp), cv2.IMREAD_GRAYSCALE)
            if ref is None or inp is None:
                raise RuntimeError("Could not read one or both images.")

            orb = cv2.ORB_create(self._max_features)
            kp1, d1 = orb.detectAndCompute(ref, None)
            kp2, d2 = orb.detectAndCompute(inp, None)
            if d1 is None or d2 is None:
                raise RuntimeError("No keypoints detected.")

            matcher = cv2.DescriptorMatcher_create(
                cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
            )
            matches = matcher.match(d1, d2, None)
            matches = sorted(matches, key=lambda x: x.distance)
            n_good = max(1, int(len(matches) * self._good_match_pct))
            good = matches[:n_good]

            # Draw matches side by side
            match_img = cv2.drawMatches(
                ref,
                kp1,
                inp,
                kp2,
                good,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            self.matchDone.emit(ref, inp, match_img)
        except Exception as e:
            self.errorOccurred.emit(str(e))


# ---------------------------------------------------------------------------
# Step thumbnail strip
# ---------------------------------------------------------------------------


class _StepThumbnail(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal(object)  # np.ndarray

    def __init__(self, title: str, img: np.ndarray, parent=None):
        super().__init__(parent)
        self._img = img
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setToolTip(title)
        pix = np_to_qpixmap(img).scaled(
            160,
            120,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pix)
        self.setFixedSize(164, 140)
        caption = QtWidgets.QLabel(title, self)
        caption.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        caption.setStyleSheet("color: #ccc; font-size: 10px;")
        caption.setGeometry(0, 122, 164, 18)
        self.setStyleSheet(
            "QLabel { border: 1px solid #555; background: #1e1e1e; border-radius: 3px; }"
        )
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        self.clicked.emit(self._img)
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# Main preview widget
# ---------------------------------------------------------------------------


class AlignmentPreviewWidget(QtWidgets.QWidget):
    """
    Tab/panel that allows:
      1. Selecting an image from the input folder
      2. Running the preprocessor pipeline (step-by-step thumbnails)
      3. For FeatureBasedAlignment: side-by-side / overlay of reference vs. result
    """

    # Emitted with the final processed image (to reload the canvas background)
    processedImageReady = QtCore.pyqtSignal(object)

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self._model = model
        self._worker: Optional[_PreviewWorker] = None
        self._feature_worker: Optional[_FeatureMatchWorker] = None
        self._last_processed: Optional[np.ndarray] = None

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # --- Image selector row ---
        sel_row = QtWidgets.QHBoxLayout()
        sel_row.addWidget(QtWidgets.QLabel("Image:"))
        self._image_combo = QtWidgets.QComboBox()
        self._image_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        sel_row.addWidget(self._image_combo, 3)
        run_btn = QtWidgets.QPushButton("▶  Run Preprocessors")
        run_btn.clicked.connect(self._run_preview)
        sel_row.addWidget(run_btn)
        root.addLayout(sel_row)

        # Spinner / status
        self._status_label = QtWidgets.QLabel("Select an image and click Run.")
        self._status_label.setStyleSheet("color: #aaa; font-size: 11px;")
        root.addWidget(self._status_label)

        # Error box
        self._error_box = QtWidgets.QLabel()
        self._error_box.setWordWrap(True)
        self._error_box.setStyleSheet(
            "color: #ff8080; background: #2a1a1a; "
            "border: 1px solid #ff4444; border-radius: 4px; padding: 4px;"
        )
        self._error_box.setVisible(False)
        root.addWidget(self._error_box)

        # Step thumbnail strip
        strip_label = QtWidgets.QLabel("Pipeline steps (click to enlarge):")
        strip_label.setStyleSheet("color: #888; font-size: 11px;")
        root.addWidget(strip_label)
        self._strip_scroll = QtWidgets.QScrollArea()
        self._strip_scroll.setWidgetResizable(True)
        self._strip_scroll.setFixedHeight(160)
        self._strip_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )
        self._strip_inner = QtWidgets.QWidget()
        self._strip_layout = QtWidgets.QHBoxLayout(self._strip_inner)
        self._strip_layout.setContentsMargins(4, 4, 4, 4)
        self._strip_layout.setSpacing(8)
        self._strip_layout.addStretch()
        self._strip_scroll.setWidget(self._strip_inner)
        root.addWidget(self._strip_scroll)

        # Large preview image with tabs (result / reference / side-by-side / overlay)
        self._preview_tabs = QtWidgets.QTabWidget()
        self._result_label = _ZoomLabel()
        self._preview_tabs.addTab(self._result_label, "Result")
        self._ref_label = _ZoomLabel()
        self._preview_tabs.addTab(self._ref_label, "Reference")
        self._sidebyside_label = _ZoomLabel()
        self._preview_tabs.addTab(self._sidebyside_label, "Side-by-Side (Features)")
        overlay_widget = QtWidgets.QWidget()
        ov_vbox = QtWidgets.QVBoxLayout(overlay_widget)
        self._alpha_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._alpha_slider.setRange(0, 100)
        self._alpha_slider.setValue(50)
        self._alpha_slider.valueChanged.connect(self._update_overlay)
        ov_vbox.addWidget(QtWidgets.QLabel("Blend (0=reference, 100=result):"))
        ov_vbox.addWidget(self._alpha_slider)
        self._overlay_label = _ZoomLabel()
        ov_vbox.addWidget(self._overlay_label, 1)
        self._preview_tabs.addTab(overlay_widget, "Overlay")
        root.addWidget(self._preview_tabs, 1)

        # Use processed button
        use_btn = QtWidgets.QPushButton("↑ Use this result as canvas background")
        use_btn.clicked.connect(self._use_as_canvas)
        root.addWidget(use_btn)

        # Populate image list
        model.changed.connect(self._reload_image_list)
        self._reload_image_list()

    # ------------------------------------------------------------------

    def _reload_image_list(self):
        folder = self._model.template_path.parent
        images = find_all_images_under(folder)
        self._image_combo.clear()
        for img in images:
            try:
                rel = str(img.relative_to(folder))
            except ValueError:
                rel = str(img)
            self._image_combo.addItem(rel, img)

    def _current_image_path(self) -> Optional[Path]:
        idx = self._image_combo.currentIndex()
        if idx < 0:
            return None
        return self._image_combo.itemData(idx)

    # ------------------------------------------------------------------
    # Preview runner
    # ------------------------------------------------------------------

    def _run_preview(self):
        img_path = self._current_image_path()
        if img_path is None:
            self._show_error("No image selected.")
            return
        template_path = self._model.template_path
        if not template_path.exists():
            self._show_error(
                "template.json has not been saved yet. "
                "Save the template first (File > Save), then run Preview."
            )
            return

        # Clear strip
        while self._strip_layout.count() > 1:
            item = self._strip_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._status_label.setText("⏳  Running preprocessors…")
        self._error_box.setVisible(False)

        self._worker = _PreviewWorker(template_path, img_path)
        self._worker.stepsDone.connect(self._on_steps_done)
        self._worker.errorOccurred.connect(self._show_error)
        self._worker.start()

    def _on_steps_done(self, steps: list):
        self._status_label.setText(f"✅  Done — {len(steps)} step(s)")

        # Populate thumbnail strip
        while self._strip_layout.count() > 1:
            item = self._strip_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for title, img in steps:
            thumb = _StepThumbnail(title, img)
            thumb.clicked.connect(self._show_enlarged)
            self._strip_layout.insertWidget(self._strip_layout.count() - 1, thumb)

        # Show final result
        if steps:
            _title_last, last_img = steps[-1]
            self._last_processed = last_img
            self._result_label.setImage(last_img)

        # Try to load reference image for feature alignment tab
        self._load_reference_image()

        # Run feature match if FeatureBasedAlignment in pipeline
        procs = self._model.get_preprocessors()
        for proc in procs:
            if proc.get("name") == "FeatureBasedAlignment":
                ref_rel = proc.get("options", {}).get("reference", "")
                ref_path = self._model.template_path.parent / ref_rel
                if ref_path.exists():
                    img_path = self._current_image_path()
                    mf = proc.get("options", {}).get("maxFeatures", 500)
                    gmp = proc.get("options", {}).get("goodMatchPercent", 0.15)
                    self._run_feature_match(ref_path, img_path, mf, gmp)
                break

    def _load_reference_image(self):
        procs = self._model.get_preprocessors()
        for proc in procs:
            if proc.get("name") == "FeatureBasedAlignment":
                ref_rel = proc.get("options", {}).get("reference", "")
                ref_path = self._model.template_path.parent / ref_rel
                if ref_path.exists():
                    ref_img = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
                    if ref_img is not None:
                        self._ref_img = ref_img
                        self._ref_label.setImage(ref_img)
                        return

    def _run_feature_match(self, ref_path: Path, img_path: Path, mf: int, gmp: float):
        self._feature_worker = _FeatureMatchWorker(ref_path, img_path, mf, gmp)
        self._feature_worker.matchDone.connect(self._on_feature_match_done)
        self._feature_worker.errorOccurred.connect(
            lambda e: self._status_label.setText(f"⚠️ Feature match: {e}")
        )
        self._feature_worker.start()

    def _on_feature_match_done(self, ref_img, inp_img, match_img):
        self._ref_img = ref_img
        self._inp_img = inp_img
        self._match_img = match_img
        self._sidebyside_label.setImage(match_img, is_bgr=True)
        self._update_overlay()

    def _update_overlay(self):
        alpha = self._alpha_slider.value() / 100.0
        ref_img = getattr(self, "_ref_img", None)
        processed = self._last_processed
        if ref_img is None or processed is None:
            return
        try:
            # Resize both to same size
            h = max(ref_img.shape[0], processed.shape[0])
            w = max(ref_img.shape[1], processed.shape[1])
            r = cv2.resize(ref_img, (w, h))
            p = cv2.resize(processed, (w, h))
            blended = cv2.addWeighted(r, 1 - alpha, p, alpha, 0)
            self._overlay_label.setImage(blended)
        except Exception:
            pass

    def _show_enlarged(self, img: np.ndarray):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Preview")
        dlg.resize(900, 700)
        vbox = QtWidgets.QVBoxLayout(dlg)
        lbl = _ZoomLabel()
        lbl.setImage(img)
        vbox.addWidget(lbl)
        dlg.exec()

    def _show_error(self, msg: str):
        self._status_label.setText("❌  Preprocessing failed")
        self._error_box.setText(msg)
        self._error_box.setVisible(True)

    def _use_as_canvas(self):
        if self._last_processed is not None:
            self.processedImageReady.emit(self._last_processed)


# ---------------------------------------------------------------------------
# Zoom-able image label
# ---------------------------------------------------------------------------


class _ZoomLabel(QtWidgets.QScrollArea):
    """A scroll area holding an image label with zoom support."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(False)
        self._label = QtWidgets.QLabel()
        self._label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft
        )
        self.setWidget(self._label)
        self._scale = 1.0
        self._pix: Optional[QtGui.QPixmap] = None

    def setImage(self, img: np.ndarray, is_bgr: bool = False):
        if is_bgr:
            pix = np_to_qpixmap(img)
        else:
            pix = np_to_qpixmap(img)
        self._pix = pix
        self._render()

    def _render(self):
        if self._pix is None:
            return
        w = int(self._pix.width() * self._scale)
        h = int(self._pix.height() * self._scale)
        scaled = self._pix.scaled(
            w,
            h,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)
        self._label.resize(scaled.size())

    def wheelEvent(self, event: QtGui.QWheelEvent):
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self._scale = max(0.1, min(10.0, self._scale * factor))
            self._render()
            event.accept()
        else:
            super().wheelEvent(event)
