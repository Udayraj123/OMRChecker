# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for OMRChecker desktop application.

Build commands (run from the project root with the venv active):

    pip install pyinstaller pywebview
    pyinstaller OMRChecker.spec

Output:  dist/OMRChecker/OMRChecker.exe   (one-directory bundle)

One-directory mode is used instead of --onefile because:
  * Faster launch (no self-extraction step on every run).
  * Windows Defender is less likely to flag it (single exe triggers more
    aggressive scanning than a directory of known-good files).
  * Easier to update individual data files without a full rebuild.

To build a single-file exe instead, change ``onefile=False`` to ``True``
and set ``console=False`` if you want to suppress the terminal window.
"""

import sys
from pathlib import Path

ROOT = Path(SPECPATH)  # noqa: F821  (SPECPATH is injected by PyInstaller)

# ---------------------------------------------------------------------------
# Data files (copied into the bundle as-is)
# ---------------------------------------------------------------------------
# Format: list of (source_glob_or_path, destination_folder_in_bundle)

datas = [
    # Web UI templates, static assets, and storage skeleton
    (str(ROOT / "webui" / "templates"),  "webui/templates"),
    (str(ROOT / "webui" / "static"),     "webui/static"),
    # OMR engine presets / schema files
    (str(ROOT / "src" / "constants"),    "src/constants"),
    (str(ROOT / "src" / "defaults"),     "src/defaults"),
    (str(ROOT / "src" / "schemas"),      "src/schemas"),
    # Sample sheets so first-time users can try the app immediately
    (str(ROOT / "samples"),              "samples"),
    # Prefill blank template image (read at runtime by the prefill service)
    (str(ROOT / "prefill_only_package" / "blank_template_reference.png"),
     "prefill_only_package"),
]

# ---------------------------------------------------------------------------
# Hidden imports
# ---------------------------------------------------------------------------
# Libraries that PyInstaller cannot auto-detect because they are imported
# dynamically (e.g. inside __init__ blocks, lazy imports, or C extensions).

hiddenimports = [
    # FastAPI / Starlette internals
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "fastapi",
    "starlette.routing",
    "starlette.middleware.cors",
    "starlette.staticfiles",
    "starlette.templating",
    "jinja2",
    "jinja2.ext",
    "anyio",
    "anyio._backends._asyncio",
    # FastAPI form parsing (Form(...) / UploadFile)
    "multipart",
    "python_multipart",
    "python_multipart.multipart",
    # Pydantic v2 settings loader
    "pydantic_settings",
    # HTTP / WebSocket transport: h11 is the pure-Python fallback that
    # uvicorn falls back to when httptools / websockets are not installed
    # (the default on Windows). Listed here so PyInstaller never strips
    # it during dead-code analysis.
    "h11",
    # Computer vision / image processing
    "cv2",
    "fitz",
    "PIL",
    "PIL.Image",
    "numpy",
    # OMR runtime modules that are looked up by string in some paths
    "webui.app",
    "webui.api",
    "webui.views",
    "webui.log_stream",
    "webui.schemas_settings",
    "webui.services.scan_simulation",
    "prefill_only_package.prefill_answer_sheet_final",
    # OMR engine processor plugins. ``src.processors.manager`` discovers
    # these dynamically via ``pkgutil.walk_packages`` which does NOT see
    # modules stored inside a frozen PYZ archive, so we list them here.
    "src.processors.manager",
    "src.processors.CropPage",
    "src.processors.CropOnMarkers",
    "src.processors.FeatureBasedAlignment",
    "src.processors.builtins",
    "src.processors.interfaces.ImagePreprocessor",
    # pywebview backends (Windows uses EdgeChromium / mshtml)
    "webview",
    "webview.platforms.winforms",
    # Python stdlib modules sometimes missed
    "email.mime.multipart",
    "email.mime.text",
    "multiprocessing.spawn",
    "multiprocessing.forkserver",
]

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

a = Analysis(
    [str(ROOT / "desktop.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude test frameworks and dev tools from the bundle
        "pytest",
        "pytest_asyncio",
        "httpx",
        "mypy",
        "ruff",
        "_pytest",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)  # noqa: F821

# ---------------------------------------------------------------------------
# EXE / bundle
# ---------------------------------------------------------------------------

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,   # one-dir mode: binaries go to COLLECT
    name="OMRChecker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,               # UPX can trigger antivirus false-positives
    console=True,            # keep True so users can see startup errors;
                             # set to False for a silent background launcher
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon="webui/static/favicon.ico",  # uncomment if you add an icon
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="OMRChecker",
)
