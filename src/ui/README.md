# Qt6 OMR Template Editor

Features
- Load template.json and a (cropped, non-blurred) image.
- Zoom with mouse wheel.
- Add FieldBlock: click “Add FieldBlock”, then click-drag on the image to create a rectangle.
- Sidebar adds a collapsible item named “MCQ_Block_Q<N>” with:
  FieldName, BubbleValues, Direction, FieldLabels, LabelsGap, Origin, Fieldtype, BubblesGap.
- Drag blocks to move. Live bubble preview. Save to <template>.edited.json.

Requirements
- Python 3.8+
- Linux (tested), macOS/Windows should work too.

Install in a venv (recommended)
- python3 -m venv .venv
- source .venv/bin/activate
- python3 -m pip install --upgrade pip
- python3 -m pip install PyQt6

Optional (only for image helpers): opencv-python
- python3 -m pip install opencv-python

Run
- From repo root:
  - python3 -m src.ui.qt_editor --template samples/sample1/template.json --image samples/sample1/page1.png

Notes
- If --image is omitted, the editor loads the first image found under ./inputs/.
- “Non blurred”: the editor shows the provided image as-is (no preprocessing).
- Fieldtype list is read from src/constants.py if available; otherwise it falls back to a minimal set.