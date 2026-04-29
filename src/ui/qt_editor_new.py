"""
OMR Template Editor — Qt6
Entry point and MainWindow.

New module layout:
  src/ui/common.py              — shared constants + helpers
  src/ui/processing.py          — pipeline runner
  src/ui/models/                — TemplateModel
  src/ui/widgets/               — all sidebar + canvas widgets
  src/ui/wizard/                — TemplateWizard
  qt_editor.py                  — MainWindow + main()
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from .common import (
    LOG,
    PROJECT_ROOT,
    apply_dark_palette,
    find_first_image_under,
    find_first_template_under,
    load_image_as_pixmap,
    np_to_qpixmap,
)
from .models.template_model import TemplateModel
from .processing import run_preprocessors_for_editor
from .widgets.alignment_preview import AlignmentPreviewWidget
from .widgets.block_graphics import BlockGraphicsItem, GraphicsView
from .widgets.block_panel import BlockPanel
from .widgets.feature_guide import FeatureGuideDialog
from .widgets.preprocessor_panel import PreprocessorPanel
from .widgets.startup_dialog import StartupDialog
from .widgets.template_settings import TemplateSettingsPanel
from .wizard.template_wizard import TemplateWizard

# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------


class MainWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        template_path: Optional[Path],
        image_path: Optional[Path],
        new_template: bool = False,
    ):
        super().__init__()
        self.setWindowTitle("OMR Template Editor")
        self.resize(1440, 860)
        self.setDockOptions(
            self.DockOption.AllowTabbedDocks | self.DockOption.AnimatedDocks
        )

        self._pending_select_name: Optional[str] = None
        self._pick_origin_block: Optional[str] = None  # block waiting for pick

        # ------------------------------------------------------------------
        # Model
        # ------------------------------------------------------------------
        if template_path is None:
            template_path = PROJECT_ROOT / "inputs" / "template.json"
        if image_path is None:
            image_path = find_first_image_under(
                template_path.parent
            ) or find_first_image_under(PROJECT_ROOT / "inputs")

        self.model = TemplateModel(template_path, image_path)

        if new_template:
            self.model.new_template(image_path)

        # ------------------------------------------------------------------
        # Scene / View
        # ------------------------------------------------------------------
        self.scene = QtWidgets.QGraphicsScene(self)
        self.view = GraphicsView(self.scene)
        self.setCentralWidget(self.view)

        # Load canvas background image
        self._reload_canvas_background(image_path)

        # ------------------------------------------------------------------
        # Build UI structure
        # ------------------------------------------------------------------
        self._build_sidebar()
        self._build_toolbar()
        self._build_menus()

        # ------------------------------------------------------------------
        # Field blocks
        # ------------------------------------------------------------------
        self.block_items: Dict[str, BlockGraphicsItem] = {}
        self.block_panels: Dict[str, BlockPanel] = {}
        for name, _ in self.model.field_blocks():
            self._create_block_item_and_panel(name)

        # ------------------------------------------------------------------
        # Signals
        # ------------------------------------------------------------------
        self.view.newRectDrawn.connect(self._add_block_from_rect)
        self.view.originPicked.connect(self._on_origin_picked)
        self.model.changed.connect(self.refresh_items)
        self.scene.selectionChanged.connect(self._on_scene_selection_changed)

        self._update_title()

    # ------------------------------------------------------------------
    # Canvas background
    # ------------------------------------------------------------------

    def _reload_canvas_background(self, image_path: Optional[Path] = None):
        """Try to run preprocessors; fall back to raw image."""
        if image_path is None:
            image_path = self.model.image_path
        if image_path is None or not image_path.exists():
            return

        # Remove old background
        if hasattr(self, "image_item") and self.image_item:
            self.scene.removeItem(self.image_item)

        processed: Optional[np.ndarray] = None
        if self.model.template_path.exists():
            try:
                processed = run_preprocessors_for_editor(
                    self.model.template_path, image_path
                )
            except Exception as e:
                LOG.warning(f"Preprocessing for canvas background failed: {e}")

        if processed is not None:
            pixmap = np_to_qpixmap(processed)
        else:
            try:
                pixmap = load_image_as_pixmap(image_path)
            except Exception:
                return

        self.image_item = self.scene.addPixmap(pixmap)
        self.image_item.setZValue(-1000)
        self.scene.setSceneRect(self.image_item.boundingRect())
        self.model.image_path = image_path

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_sidebar(self):
        # ---- Right dock: tab widget with all sidebar panels ----
        self.sidebar_dock = QtWidgets.QDockWidget("Template", self)
        self.sidebar_dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
        self.sidebar_dock.setMinimumWidth(440)

        tabs = QtWidgets.QTabWidget()

        # Tab 0: FieldBlocks
        blocks_scroll = QtWidgets.QScrollArea()
        blocks_scroll.setWidgetResizable(True)
        self.sidebar_inner = QtWidgets.QWidget()
        self.sidebar_inner_layout = QtWidgets.QVBoxLayout(self.sidebar_inner)
        self.sidebar_inner_layout.setContentsMargins(0, 0, 0, 0)
        self.sidebar_inner_layout.setSpacing(6)
        blocks_scroll.setWidget(self.sidebar_inner)
        self.sidebar_scroll = blocks_scroll
        tabs.addTab(blocks_scroll, "FieldBlocks")

        # Tab 1: Template Settings
        settings_scroll = QtWidgets.QScrollArea()
        settings_scroll.setWidgetResizable(True)
        self.template_settings_panel = TemplateSettingsPanel(self.model)
        settings_scroll.setWidget(self.template_settings_panel)
        tabs.addTab(settings_scroll, "Settings")

        # Tab 2: Preprocessors
        procs_scroll = QtWidgets.QScrollArea()
        procs_scroll.setWidgetResizable(True)
        self.preprocessor_panel = PreprocessorPanel(self.model)
        self.preprocessor_panel.changed.connect(self._on_preprocessors_changed)
        procs_scroll.setWidget(self.preprocessor_panel)
        tabs.addTab(procs_scroll, "Preprocessors")

        # Tab 3: Preview
        self.preview_widget = AlignmentPreviewWidget(self.model)
        self.preview_widget.processedImageReady.connect(
            self._on_preview_result_to_canvas
        )
        tabs.addTab(self.preview_widget, "Preview")

        self.sidebar_dock.setWidget(tabs)
        self.addDockWidget(
            QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.sidebar_dock
        )

    def _build_toolbar(self):
        tb = self.addToolBar("Tools")
        tb.setMovable(False)

        add_act = QtGui.QAction("➕  Add FieldBlock", self)
        add_act.setToolTip("Drag-draw a new field block on the canvas")
        add_act.triggered.connect(self._enter_add_mode)
        tb.addAction(add_act)

        tb.addSeparator()

        refresh_act = QtGui.QAction("🔄  Refresh Preview", self)
        refresh_act.setToolTip("Reload canvas background from preprocessors")
        refresh_act.triggered.connect(lambda: self._reload_canvas_background())
        tb.addAction(refresh_act)

        fit_act = QtGui.QAction("⊞  Fit View", self)
        fit_act.triggered.connect(self.view.fit_image)
        tb.addAction(fit_act)

        tb.addSeparator()

        save_act = QtGui.QAction("💾  Save", self)
        save_act.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        save_act.triggered.connect(self._on_save)
        tb.addAction(save_act)

    def _build_menus(self):
        menubar = self.menuBar()

        # File
        file_menu = menubar.addMenu("File")

        new_act = QtGui.QAction("New Template (Wizard)…", self)
        new_act.setShortcut("Ctrl+N")
        new_act.triggered.connect(self._on_new_wizard)
        file_menu.addAction(new_act)

        open_act = QtGui.QAction("Open…", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._on_open)
        file_menu.addAction(open_act)

        open_folder_act = QtGui.QAction("Open Input Folder…", self)
        open_folder_act.triggered.connect(self._on_open_folder)
        file_menu.addAction(open_folder_act)

        file_menu.addSeparator()

        save_act = QtGui.QAction("Save", self)
        save_act.setShortcut("Ctrl+S")
        save_act.triggered.connect(self._on_save)
        file_menu.addAction(save_act)

        save_as_act = QtGui.QAction("Save As…", self)
        save_as_act.setShortcut("Ctrl+Shift+S")
        save_as_act.triggered.connect(self._on_save_as)
        file_menu.addAction(save_as_act)

        file_menu.addSeparator()

        exit_act = QtGui.QAction("Exit", self)
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        # Edit
        edit_menu = menubar.addMenu("Edit")

        undo_act = QtGui.QAction("Undo", self)
        undo_act.setShortcut(QtGui.QKeySequence.StandardKey.Undo)
        undo_act.triggered.connect(lambda: self.model.undo())
        edit_menu.addAction(undo_act)

        redo_act = QtGui.QAction("Redo", self)
        redo_act.setShortcut(QtGui.QKeySequence.StandardKey.Redo)
        redo_act.triggered.connect(lambda: self.model.redo())
        edit_menu.addAction(redo_act)

        edit_menu.addSeparator()

        delete_act = QtGui.QAction("Delete Selected Block", self)
        delete_act.setShortcut(QtGui.QKeySequence.StandardKey.Delete)
        delete_act.triggered.connect(self._delete_selected)
        edit_menu.addAction(delete_act)

        # View
        view_menu = menubar.addMenu("View")

        fit_act = QtGui.QAction("Fit Image", self)
        fit_act.triggered.connect(self.view.fit_image)
        view_menu.addAction(fit_act)

        reload_bg_act = QtGui.QAction("Reload Background Image", self)
        reload_bg_act.triggered.connect(lambda: self._reload_canvas_background())
        view_menu.addAction(reload_bg_act)

        # Help
        help_menu = menubar.addMenu("Help")

        guide_act = QtGui.QAction("Feature Guide…", self)
        guide_act.triggered.connect(self._on_open_guide)
        help_menu.addAction(guide_act)

        about_act = QtGui.QAction("About", self)
        about_act.triggered.connect(self._on_about)
        help_menu.addAction(about_act)

    # ------------------------------------------------------------------
    # Toolbar / menu handlers
    # ------------------------------------------------------------------

    def _enter_add_mode(self):
        self.view.enter_add_mode()
        self.statusBar().showMessage(
            "Drag to draw a new field block. Esc to cancel.", 5000
        )

    def _delete_selected(self):
        names = [n for n, it in self.block_items.items() if it.isSelected()]
        if names:
            self.model.remove_block(names[0])

    def _on_save(self):
        out = self.model.save(self.model.template_path)
        self.statusBar().showMessage(f"Saved: {out}", 4000)
        self._update_title()

    def _on_save_as(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Template As",
            str(self.model.template_path.parent / "template.json"),
            "JSON Files (*.json)",
            options=QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            out = self.model.save(Path(path))
            self.statusBar().showMessage(f"Saved: {out}", 4000)
            self._update_title()

    def _on_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Template",
            str(PROJECT_ROOT / "inputs"),
            "JSON Files (template.json *.json)",
            options=QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            self._load_template(Path(path))

    def _on_open_folder(self):
        tmpl, img, want_new = StartupDialog.run(self)
        if tmpl or want_new:
            self._load_template(tmpl, img, want_new)

    def _on_new_wizard(self):
        saved = TemplateWizard.run_wizard(self)
        if saved and saved.exists():
            self._load_template(saved)

    def _on_preprocessors_changed(self):
        # Reload canvas background after pipeline changes
        self._reload_canvas_background()

    def _on_preview_result_to_canvas(self, img: np.ndarray):
        """User clicked 'Use as canvas background' in the Preview tab."""
        # Resize to template page dimensions so block coordinates stay aligned
        try:
            pw, ph = self.model.get_page_dimensions()
            if pw > 0 and ph > 0:
                img = cv2.resize(img, (int(pw), int(ph)))
        except Exception:
            pass
        if hasattr(self, "image_item") and self.image_item:
            self.scene.removeItem(self.image_item)
        pix = np_to_qpixmap(img)
        self.image_item = self.scene.addPixmap(pix)
        self.image_item.setZValue(-1000)
        self.scene.setSceneRect(self.image_item.boundingRect())
        self.statusBar().showMessage("Canvas background updated from preview.", 3000)

    def _on_open_guide(self):
        FeatureGuideDialog.show_guide(self, on_open_sample=self._load_template)

    def _on_about(self):
        QtWidgets.QMessageBox.information(
            self,
            "About",
            "<b>OMR Template Editor</b><br>"
            "A visual editor for OMRChecker template.json files.<br><br>"
            "Built with PyQt6.",
        )

    # ------------------------------------------------------------------
    # Load template
    # ------------------------------------------------------------------

    def _load_template(
        self,
        template_path: Optional[Path],
        image_path: Optional[Path] = None,
        new_template: bool = False,
    ):
        if template_path is None and not new_template:
            return
        if template_path is None:
            template_path = PROJECT_ROOT / "inputs" / "template.json"

        if image_path is None:
            image_path = find_first_image_under(template_path.parent)

        if new_template:
            self.model.template_path = template_path
            self.model.image_path = image_path
            self.model.new_template(image_path)
        else:
            self.model.load_from_path(template_path, image_path)

        # Reload canvas
        self._reload_canvas_background(image_path)

        # Rebuild block items/panels
        self._clear_block_items()
        self.block_items = {}
        self.block_panels = {}
        for name, _ in self.model.field_blocks():
            self._create_block_item_and_panel(name)

        # Update sub-panels
        self.template_settings_panel._sync_from_model()
        self.preprocessor_panel._load_from_model()
        self.preview_widget._reload_image_list()
        self._update_title()

    def _clear_block_items(self):
        for it in self.block_items.values():
            self.scene.removeItem(it)
        for pnl in self.block_panels.values():
            self.sidebar_inner_layout.removeWidget(pnl)
            pnl.deleteLater()

    def _update_title(self):
        dirty = "* " if self.model.is_dirty else ""
        self.setWindowTitle(f"{dirty}OMR Template Editor — {self.model.template_path}")

    # ------------------------------------------------------------------
    # Block creation / management
    # ------------------------------------------------------------------

    def _add_block_from_rect(self, rect: QtCore.QRectF):
        name = self.model.next_block_name()
        self._pending_select_name = name
        self.model.add_block(name, rect)
        for n, it in self.block_items.items():
            it.setSelected(n == name)
        panel = self.block_panels.get(name)
        if panel:
            for n, pnl in self.block_panels.items():
                pnl.setChecked(pnl is panel)
            self.sidebar_scroll.ensureWidgetVisible(panel)

    def _create_block_item_and_panel(self, name: str):
        item = BlockGraphicsItem(name, self.model)
        self.scene.addItem(item)
        self.block_items[name] = item

        panel = BlockPanel(name, self.model)
        panel.changed.connect(self._on_panel_changed)
        panel.pickOriginRequested.connect(self._on_pick_origin_requested)
        self.block_panels[name] = panel
        self.sidebar_inner_layout.addWidget(panel)
        panel._toggle_body(False)

    def _on_panel_changed(self, name: str):
        self.refresh_items()
        self._update_title()

    def refresh_items(self):
        names_now = set(n for n, _ in self.model.field_blocks())
        expanded_states: Dict[str, bool] = {
            n: w.isChecked() for n, w in self.block_panels.items()
        }

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

        # Update / create
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
                pnl.pickOriginRequested.connect(self._on_pick_origin_requested)
                self.block_panels[name] = pnl
                self.sidebar_inner_layout.addWidget(pnl)
            else:
                self.block_panels[name].sync_from_model()

        for name in names_now:
            if name in expanded_states:
                self.block_panels[name].setChecked(expanded_states[name])

        if self._pending_select_name and self._pending_select_name in self.block_items:
            for n, it in self.block_items.items():
                it.setSelected(n == self._pending_select_name)
            panel = self.block_panels.get(self._pending_select_name)
            if panel:
                for n, pnl in self.block_panels.items():
                    pnl.setChecked(pnl is panel)
                self.sidebar_scroll.ensureWidgetVisible(panel)
            self._pending_select_name = None

        self.sidebar_inner_layout.addStretch(0)

    def _on_scene_selection_changed(self):
        selected = [n for n, it in self.block_items.items() if it.isSelected()]
        if not selected:
            return
        name = selected[0]
        panel = self.block_panels.get(name)
        if panel is None:
            return
        for n, pnl in self.block_panels.items():
            pnl.setChecked(pnl is panel)
        self.sidebar_scroll.ensureWidgetVisible(panel)

    # ------------------------------------------------------------------
    # Pick-origin mode
    # ------------------------------------------------------------------

    def _on_pick_origin_requested(self, block_name: str):
        self._pick_origin_block = block_name
        self.view.enter_pick_mode()
        self.statusBar().showMessage(
            f"Click on canvas to set origin for '{block_name}'. Esc to cancel.", 0
        )

    def _on_origin_picked(self, scene_pos: QtCore.QPointF):
        if self._pick_origin_block is None:
            return
        name = self._pick_origin_block
        self._pick_origin_block = None
        self.statusBar().clearMessage()
        panel = self.block_panels.get(name)
        if panel:
            panel.set_origin(int(scene_pos.x()), int(scene_pos.y()))
        self.view.exit_add_mode()

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Escape:
            if self._pick_origin_block:
                self._pick_origin_block = None
                self.view.exit_add_mode()
                self.statusBar().clearMessage()
                return

        if event.key() in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
            self._delete_selected()
            return

        if event.matches(QtGui.QKeySequence.StandardKey.Undo):
            if self.model.undo():
                self._update_title()
            return

        if event.matches(QtGui.QKeySequence.StandardKey.Redo):
            if self.model.redo():
                self._update_title()
            return

        super().keyPressEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.model.is_dirty:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Unsaved changes",
                "Template has unsaved changes. Save before closing?",
                QtWidgets.QMessageBox.StandardButton.Save
                | QtWidgets.QMessageBox.StandardButton.Discard
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Save,
            )
            if reply == QtWidgets.QMessageBox.StandardButton.Save:
                self._on_save()
                event.accept()
            elif reply == QtWidgets.QMessageBox.StandardButton.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: List[str]):
    import argparse

    ap = argparse.ArgumentParser(
        description="OMR Template Editor",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument(
        "folder",
        nargs="?",
        help="Input folder (or template.json path). Auto-detect if omitted.",
    )
    ap.add_argument("--template", help="Explicit path to template.json")
    ap.add_argument("--image", help="Explicit path to a sample image")
    ap.add_argument("--new", action="store_true", help="Start wizard for new template")
    args = ap.parse_args(argv)

    tmpl: Optional[Path] = None
    img: Optional[Path] = None
    new_template = bool(args.new)

    if args.template:
        tmpl = Path(args.template).resolve()
    if args.image:
        img = Path(args.image).resolve()
    if args.folder and tmpl is None:
        p = Path(args.folder).resolve()
        if p.is_file() and p.suffix == ".json":
            tmpl = p
        else:
            tmpl = find_first_template_under(p)
            if img is None:
                img = find_first_image_under(p)

    return tmpl, img, new_template


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    tmpl, img, new_template = parse_args(sys.argv[1:])

    app = QtWidgets.QApplication(sys.argv)
    apply_dark_palette(app)

    # If no template provided from CLI, show startup dialog
    want_new = new_template
    if tmpl is None and not new_template:
        tmpl, img, want_new = StartupDialog.run()
        if tmpl is None and not want_new:
            # User cancelled startup dialog — auto-detect from ./inputs
            tmpl = find_first_template_under(PROJECT_ROOT / "inputs")
            if tmpl is None:
                # Launch wizard for new template
                saved = TemplateWizard.run_wizard()
                if saved and saved.exists():
                    tmpl = saved
                else:
                    sys.exit(0)

    if want_new and tmpl is None:
        saved = TemplateWizard.run_wizard()
        if saved and saved.exists():
            tmpl = saved
            want_new = False
        else:
            sys.exit(0)

    if tmpl is None:
        tmpl = PROJECT_ROOT / "inputs" / "template.json"

    if img is None:
        img = find_first_image_under(tmpl.parent) or find_first_image_under(
            PROJECT_ROOT / "inputs"
        )

    w = MainWindow(tmpl, img, new_template=want_new)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
