"""FeatureGuideDialog — in-app help with sample images and descriptions."""
from pathlib import Path
from typing import Optional

from PyQt6 import QtCore, QtWidgets

from ..common import PROJECT_ROOT, find_all_images_under, np_to_qpixmap

# ---------------------------------------------------------------------------
# Guide content
# ---------------------------------------------------------------------------

GUIDE_CHAPTERS = [
    {
        "title": "Overview",
        "content": """
<h2>OMR Template Editor Guide</h2>
<p>This editor helps you create and edit <b>template.json</b> files that describe
where bubble fields are located on an OMR (Optical Mark Recognition) sheet.</p>

<h3>Workflow</h3>
<ol>
<li><b>Open</b> an input folder (File → Open, or use the startup dialog).</li>
<li><b>Configure Preprocessors</b> in the Preprocessors panel (right sidebar) — this aligns and crops the image.</li>
<li><b>Preview</b> preprocessing on actual images using the Preview tab.</li>
<li><b>Draw Field Blocks</b> on the canvas by clicking "Add FieldBlock" then drag-drawing rectangles.</li>
<li><b>Configure each block</b> in its sidebar panel (fieldType, labels, gaps, origin).</li>
<li><b>Set global Template Settings</b> (page dimensions, bubble dimensions, output columns).</li>
<li><b>Save</b> the template (File → Save or Ctrl+S).</li>
</ol>

<h3>Keyboard shortcuts</h3>
<ul>
<li><b>Ctrl+Z</b> — Undo</li>
<li><b>Ctrl+Y / Ctrl+Shift+Z</b> — Redo</li>
<li><b>Del / Backspace</b> — Delete selected block</li>
<li><b>Ctrl+S</b> — Save</li>
<li><b>Mouse wheel</b> — Zoom in/out on canvas</li>
</ul>
""",
        "sample": None,
        "images": [],
    },
    {
        "title": "CropPage",
        "content": """
<h2>CropPage Preprocessor</h2>
<p>Detects the paper boundary using edge detection and perspective-crops the sheet
so only the form remains.</p>
<h3>When to use</h3>
<ul>
<li>Flat, cleanly scanned sheets with a visible page border</li>
<li>Scanner output where the paper edge is visible but you want to normalize the crop</li>
</ul>
<h3>Options</h3>
<ul>
<li><b>morphKernel [W, H]</b> — Size of morphological kernel used for edge detection.
Larger = finds coarser boundaries. Default [10, 10] works for most scans.</li>
</ul>
<h3>Samples that use CropPage</h3>
<p>sample2, sample3, sample4</p>
""",
        "sample": "sample2",
        "images": [],
    },
    {
        "title": "CropOnMarkers",
        "content": """
<h2>CropOnMarkers Preprocessor</h2>
<p>Finds four corner markers (fiducial square images) printed on the OMR sheet,
then warps the image so the markers align exactly to the template layout.</p>
<h3>When to use</h3>
<ul>
<li>Sheets printed with <b>omr_marker.jpg</b> at all four corners</li>
<li>Mobile camera photos where perspective distortion is significant</li>
<li>Batch processing where sheets may be rotated or skewed</li>
</ul>
<h3>Key options</h3>
<ul>
<li><b>relativePath</b> — Path to the marker image (relative to template.json).</li>
<li><b>sheetToMarkerWidthRatio</b> — Approximate fraction of sheet width occupied by marker. Default 0.05.</li>
<li><b>min_matching_threshold</b> — Minimum template-match confidence (0–1). Lower = finds markers in worse quality. Default 0.3.</li>
<li><b>max_matching_variation</b> — Maximum allowed variation in match confidence between the four corners. Default 0.41.</li>
<li><b>marker_rescale_range [min%, max%]</b> — Search for marker at scales from min% to max% of sheet width. Default [35, 100].</li>
</ul>
<h3>Samples that use CropOnMarkers</h3>
<p>sample1, sample5</p>
""",
        "sample": "sample5",
        "images": [],
    },
    {
        "title": "FeatureBasedAlignment",
        "content": """
<h2>FeatureBasedAlignment Preprocessor</h2>
<p>Uses ORB keypoint detection and descriptor matching to align a photo of the form
to a clean reference image, then applies full homography to correct perspective.</p>
<h3>When to use</h3>
<ul>
<li>Mobile camera photos where no corner markers are printed</li>
<li>Complex form designs where edge detection fails</li>
<li>Scanned documents that may be slightly shifted or rotated</li>
</ul>
<h3>Options</h3>
<ul>
<li><b>reference</b> — Path to reference image (relative to template.json).
This should be a high-quality scan or digital rendering of the blank form.</li>
<li><b>maxFeatures</b> — Number of ORB keypoints to detect. Default 500.</li>
<li><b>goodMatchPercent</b> — Fraction of matches to keep (best N%). Default 0.15.</li>
<li><b>2d</b> — If true, use 2D affine transform (less correction power) instead of full homography.</li>
</ul>
<h3>Preview tip</h3>
<p>Use the <b>Side-by-Side (Features)</b> tab in the Preview panel to visualize
which keypoints are matched between the reference and input image.</p>
<h3>Samples that use FeatureBasedAlignment</h3>
<p>sample6 (template_fb_align.json), TestMio</p>
""",
        "sample": "sample6",
        "images": [],
    },
    {
        "title": "Field Types",
        "content": """
<h2>Field Types (fieldType)</h2>
<p>The <b>fieldType</b> property of a FieldBlock determines the bubble values and default
direction for that block. If absent, you must supply <b>bubbleValues</b> and
<b>direction</b> manually.</p>

<table border="1" cellspacing="4" cellpadding="4">
<tr><th>Type</th><th>Bubble Values</th><th>Direction</th><th>Use for</th></tr>
<tr><td>QTYPE_INT</td><td>0–9</td><td>vertical</td><td>Integer answers (roll number, numeric ID)</td></tr>
<tr><td>QTYPE_INT_FROM_1</td><td>1–9, 0</td><td>vertical</td><td>Integer starting at 1 (some exam conventions)</td></tr>
<tr><td>QTYPE_MCQ4</td><td>A, B, C, D</td><td>horizontal</td><td>4-option MCQ</td></tr>
<tr><td>QTYPE_MCQ5</td><td>A, B, C, D, E</td><td>horizontal</td><td>5-option MCQ</td></tr>
</table>

<h3>Custom types</h3>
<p>Leave fieldType blank and set <b>bubbleValues</b> manually (e.g. "T,F" for True/False).</p>

<h3>FieldLabels</h3>
<p>Use range notation <code>q1..10</code> to expand to q1, q2, … q10.
Or list them manually: <code>roll1,roll2,roll3</code></p>

<h3>BubblesGap / LabelsGap</h3>
<ul>
<li>Horizontal direction: <b>bubblesGap</b> = horizontal distance between adjacent bubble centers;
<b>labelsGap</b> = vertical distance between question rows.</li>
<li>Vertical direction: reversed.</li>
</ul>
""",
        "sample": None,
        "images": [],
    },
    {
        "title": "Blurs & Levels",
        "content": """
<h2>Image Enhancement Preprocessors</h2>
<p>These are typically applied <i>before</i> cropping/alignment to improve image quality.</p>

<h3>GaussianBlur</h3>
<ul>
<li>Smooths fine noise. Use when images have grain or scanner artifacts.</li>
<li><b>kSize</b> [W, H]: Kernel size (must be odd). Larger = stronger blur.</li>
<li><b>sigmaX</b>: Gaussian sigma. 0 = auto-compute from kSize.</li>
</ul>

<h3>MedianBlur</h3>
<ul>
<li>Removes salt-and-pepper noise (isolated dark/white pixels).</li>
<li><b>kSize</b>: Must be odd. Typical value: 3 or 5.</li>
</ul>

<h3>Levels</h3>
<ul>
<li>Adjusts brightness and contrast range.</li>
<li><b>low</b> (0–1): Input black point — values below this become black.</li>
<li><b>high</b> (0–1): Input white point — values above become white.</li>
<li><b>gamma</b>: > 1 brightens midtones; < 1 darkens.</li>
<li>Sample6 uses <code>{"low": 0, "high": 0.6, "gamma": 1}</code> to boost contrast.</li>
</ul>
""",
        "sample": "sample4",
        "images": [],
    },
    {
        "title": "customLabels & outputColumns",
        "content": """
<h2>customLabels</h2>
<p>Groups field block labels into named aggregates for evaluation scoring:</p>
<pre>
"customLabels": {
  "Maths": ["q1", "q2", "q3"],
  "Science": ["q4", "q5", "q6"]
}
</pre>
<p>The evaluation.json can reference these group names for section-wise scoring.</p>

<h2>outputColumns</h2>
<p>Controls the order of columns in the output CSV:</p>
<pre>
"outputColumns": ["roll_no", "q1", "q2", ..., "score"]
</pre>
<p>If absent, columns follow the order of fieldBlocks in the template.</p>
""",
        "sample": None,
        "images": [],
    },
    {
        "title": "FAQ",
        "content": """
<h2>Frequently Asked Questions</h2>

<h3>Why does the preview show preprocessing failed?</h3>
<p>You must <b>Save the template first</b> (Ctrl+S) before running preview — the
pipeline reads template.json from disk. Also ensure dependencies are installed:
<code>pip install -r requirements.txt</code></p>

<h3>Why are my bubble positions slightly off?</h3>
<p>The <b>origin</b> of each FieldBlock is its top-left corner in page-dimension coordinates.
After cropping, the page is resized to <b>pageDimensions</b>. Make sure pageDimensions
matches the output resolution of your preprocessing step.</p>

<h3>How do I add a second page?</h3>
<p>Run the editor again for the second page with a different template.json.
Each template describes one page. Use OMRChecker's multi-directory support to process both.</p>

<h3>How do I test my template without running full evaluation?</h3>
<p>Use the Preview tab to see the preprocessed image, then check that your field block
overlays land on the correct bubbles.</p>

<h3>What is .edited.json?</h3>
<p>Legacy save format from older versions of the editor. The current editor saves
to <b>template.json</b> directly (or a user-chosen path via Save As).</p>
""",
        "sample": None,
        "images": [],
    },
]


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


class FeatureGuideDialog(QtWidgets.QDialog):
    """In-app feature guide with tree navigation and HTML content."""

    # Emitted when user wants to open a sample in the editor
    openSampleRequested = QtCore.pyqtSignal(Path)  # template path

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OMR Template Editor — Feature Guide")
        self.setMinimumSize(900, 650)
        self.resize(1050, 720)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # Left: chapter tree
        self._tree = QtWidgets.QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setMaximumWidth(200)
        for ch in GUIDE_CHAPTERS:
            item = QtWidgets.QTreeWidgetItem([ch["title"]])
            self._tree.addTopLevelItem(item)
        self._tree.currentItemChanged.connect(self._on_chapter_changed)
        splitter.addWidget(self._tree)

        # Right: content + images
        right = QtWidgets.QWidget()
        right_vbox = QtWidgets.QVBoxLayout(right)
        right_vbox.setContentsMargins(4, 4, 4, 4)

        self._browser = QtWidgets.QTextBrowser()
        self._browser.setOpenExternalLinks(True)
        right_vbox.addWidget(self._browser, 1)

        # Sample images strip
        self._img_strip_widget = QtWidgets.QWidget()
        self._img_strip_layout = QtWidgets.QHBoxLayout(self._img_strip_widget)
        self._img_strip_layout.setContentsMargins(0, 0, 0, 0)
        img_scroll = QtWidgets.QScrollArea()
        img_scroll.setWidgetResizable(True)
        img_scroll.setFixedHeight(140)
        img_scroll.setWidget(self._img_strip_widget)
        right_vbox.addWidget(img_scroll)

        # Open sample button
        self._open_sample_btn = QtWidgets.QPushButton("Open this sample in editor")
        self._open_sample_btn.setVisible(False)
        self._open_sample_btn.clicked.connect(self._on_open_sample)
        right_vbox.addWidget(self._open_sample_btn)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 4)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(splitter)

        # Select first chapter
        self._tree.setCurrentItem(self._tree.topLevelItem(0))
        self._current_sample: Optional[str] = None

    # ------------------------------------------------------------------

    def _on_chapter_changed(self, current: QtWidgets.QTreeWidgetItem, previous):
        if current is None:
            return
        idx = self._tree.indexOfTopLevelItem(current)
        ch = GUIDE_CHAPTERS[idx]
        self._browser.setHtml(ch["content"])
        self._current_sample = ch.get("sample")

        # Load sample images
        while self._img_strip_layout.count():
            item = self._img_strip_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        sample = ch.get("sample")
        if sample:
            sample_dir = PROJECT_ROOT / "samples" / sample
            images = find_all_images_under(sample_dir)[:4]
            for img_path in images:
                try:
                    import cv2

                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    pix = np_to_qpixmap(img).scaled(
                        180,
                        110,
                        QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                        QtCore.Qt.TransformationMode.SmoothTransformation,
                    )
                    lbl = QtWidgets.QLabel()
                    lbl.setPixmap(pix)
                    lbl.setToolTip(str(img_path.name))
                    lbl.setStyleSheet(
                        "border: 1px solid #555; background: #1e1e1e; "
                        "border-radius: 3px; padding: 2px;"
                    )
                    self._img_strip_layout.addWidget(lbl)
                except Exception:
                    pass
        self._img_strip_layout.addStretch()
        self._open_sample_btn.setVisible(bool(sample))

    def _on_open_sample(self):
        sample = self._current_sample
        if not sample:
            return
        sample_dir = PROJECT_ROOT / "samples" / sample
        tmpl = None
        for subdir in sorted(sample_dir.rglob("template.json")):
            tmpl = subdir
            break
        if tmpl and tmpl.exists():
            self.openSampleRequested.emit(tmpl)
            self.accept()
        else:
            QtWidgets.QMessageBox.warning(
                self, "Not found", f"No template.json found in {sample_dir}"
            )

    @staticmethod
    def show_guide(parent=None, on_open_sample=None) -> None:
        dlg = FeatureGuideDialog(parent)
        if on_open_sample:
            dlg.openSampleRequested.connect(on_open_sample)
        dlg.exec()
