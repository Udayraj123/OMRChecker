"""Desktop launcher for OMRChecker.

Starts the FastAPI/Uvicorn server in a background thread, waits for it to
become ready, then opens a ``pywebview`` native window pointing to the local
server.  When the window is closed the server thread is stopped and the
process exits.

Usage (development)::

    python desktop.py

Packaging (after ``pip install pyinstaller pywebview``)::

    pyinstaller OMRChecker.spec

The resulting ``dist/OMRChecker/OMRChecker.exe`` (or ``dist/OMRChecker.exe``
in one-file mode) requires no separate server start command.
"""

from __future__ import annotations

import base64
import logging
import multiprocessing
import os
import shutil
import socket
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve paths correctly whether running from source or from a PyInstaller
# one-dir / one-file bundle.
# ---------------------------------------------------------------------------
if getattr(sys, "frozen", False):
    # Running inside a PyInstaller bundle — _MEIPASS is the extracted temp dir
    BASE_DIR = sys._MEIPASS  # type: ignore[attr-defined]
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure the project root is on sys.path so that ``webui`` and ``src`` are
# importable when launched via ``python desktop.py`` from any cwd.
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def _user_data_root() -> Path:
    """Return a writable per-user directory for batches, cache and logs.

    Inside a PyInstaller bundle ``Path(__file__).parent.parent`` points
    into ``_internal`` (read-only for users who installed under Program
    Files). We must redirect writes to ``%LOCALAPPDATA%\\OMRChecker`` on
    Windows, ``~/Library/Application Support/OMRChecker`` on macOS, and
    ``~/.local/share/OMRChecker`` elsewhere.
    """
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
        return Path(base) / "OMRChecker"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "OMRChecker"
    return Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")) / "OMRChecker"


def _configure_user_data_dirs() -> None:
    """Point ``OMR_WEBUI_*`` settings at the per-user data directory.

    Done before importing ``webui.app`` so the Settings model picks them
    up via its env prefix. Skipped when the caller already set them so
    deployments can pin their own paths.
    """
    if not getattr(sys, "frozen", False):
        # Source / dev runs keep the repo-local storage so test fixtures
        # and tooling continue to find them in the working tree.
        return
    root = _user_data_root()
    storage_default = str(root / "storage" / "batches")
    cache_default = str(root / "cache")
    os.environ.setdefault("OMR_WEBUI_STORAGE_ROOT", storage_default)
    os.environ.setdefault("OMR_WEBUI_CACHE_ROOT", cache_default)
    try:
        Path(storage_default).mkdir(parents=True, exist_ok=True)
        Path(cache_default).mkdir(parents=True, exist_ok=True)
    except OSError:
        # Surface in logs later; don't crash startup over a writable-dir
        # race when the parent already exists.
        pass


_configure_user_data_dirs()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HOST = "127.0.0.1"
PORT = 5050
READY_TIMEOUT = 30  # seconds to wait for the server to accept connections
WINDOW_TITLE = "OMRChecker"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 900
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_free_port(preferred: int) -> int:
    """Return *preferred* if it is free, otherwise a random available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((HOST, preferred))
            return preferred
        except OSError:
            # Port in use — let the OS assign a free one
            s.bind((HOST, 0))
            return s.getsockname()[1]


def _wait_for_server(host: str, port: int, timeout: float) -> bool:
    """Poll until the server accepts TCP connections or *timeout* elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.25)
    return False


def _safe_download_name(filename: str) -> str:
    """Return a filesystem-safe download filename."""
    cleaned = "".join(ch for ch in filename if ch not in '<>:"/\\|?*').strip()
    return cleaned or "download"


def _with_download_extension(target: str, filename: str) -> str:
    """Append the generated file extension when the save dialog omits it."""
    expected_suffix = Path(_safe_download_name(filename)).suffix
    target_path = Path(target)
    if expected_suffix and not target_path.suffix:
        return str(target_path.with_suffix(expected_suffix))
    return str(target_path)


class DesktopApi:
    """Small pywebview bridge for reliable desktop downloads."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def _choose_path(self, filename: str) -> str | None:
        import webview  # type: ignore[import]

        window = webview.windows[0] if webview.windows else None
        if window is None:
            return None
        file_dialog = getattr(getattr(webview, "FileDialog", None), "SAVE", None)
        if file_dialog is None:
            file_dialog = webview.SAVE_DIALOG
        result = window.create_file_dialog(
            file_dialog,
            save_filename=_safe_download_name(filename),
        )
        if not result:
            return None
        if isinstance(result, (list, tuple)):
            return str(result[0]) if result else None
        return str(result)

    def _download_url_to_path(self, absolute_url: str, target: str) -> None:
        """Stream a generated local file to disk in a background thread."""
        try:
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            logger.info("Desktop download save started | target=%s", target)
            with urllib.request.urlopen(absolute_url, timeout=120) as response, open(target, "wb") as out:
                shutil.copyfileobj(response, out)
            size = Path(target).stat().st_size
            logger.info("Desktop download save complete | target=%s | size_mb=%.1f", target, size / (1024 * 1024))
        except Exception as exc:  # noqa: BLE001
            logger.error("Desktop download save failed | target=%s | error=%s: %s", target, type(exc).__name__, exc)

    def save_download_url(self, url: str, filename: str) -> dict[str, str | bool]:
        """Prompt for a path, then start saving a generated local file."""
        target = self._choose_path(filename)
        if not target:
            return {"ok": False, "cancelled": True, "message": "Download cancelled."}
        target = _with_download_extension(target, filename)

        absolute_url = urllib.parse.urljoin(self.base_url, url)
        parsed = urllib.parse.urlparse(absolute_url)
        if parsed.hostname not in {HOST, "localhost"}:
            return {"ok": False, "message": "Refusing to download from a non-local URL."}

        thread = threading.Thread(
            target=self._download_url_to_path,
            args=(absolute_url, target),
            daemon=True,
            name="omr-desktop-download",
        )
        thread.start()
        return {"ok": True, "path": target, "started": True}

    def save_download_base64(self, filename: str, data: str) -> dict[str, str | bool]:
        """Prompt for a path, then save browser-provided base64 data."""
        target = self._choose_path(filename)
        if not target:
            return {"ok": False, "cancelled": True, "message": "Download cancelled."}
        target = _with_download_extension(target, filename)

        try:
            if "," in data:
                data = data.split(",", 1)[1]
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            Path(target).write_bytes(base64.b64decode(data))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"Save failed: {type(exc).__name__}: {exc}"}
        return {"ok": True, "path": target}


# ---------------------------------------------------------------------------
# Server thread
# ---------------------------------------------------------------------------

def _run_server(port: int) -> None:
    """Run Uvicorn in the calling thread (meant to be a daemon thread)."""
    import uvicorn
    from webui.app import create_app  # direct import so frozen PYZ works

    uvicorn.run(
        create_app,
        factory=True,
        host=HOST,
        port=port,
        # Never use --reload in desktop mode — no file watcher needed, and
        # it would spawn child processes that complicate the process tree.
        reload=False,
        log_level="info",
        access_log=False,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        import webview  # type: ignore[import]
    except ImportError:
        print(
            "pywebview is not installed.  Run:\n"
            "    pip install pywebview\n"
            "then retry.",
            file=sys.stderr,
        )
        sys.exit(1)

    port = _find_free_port(PORT)
    url = f"http://{HOST}:{port}"

    # Start server in a daemon thread so it dies when the main thread exits.
    server_thread = threading.Thread(
        target=_run_server,
        args=(port,),
        daemon=True,
        name="omr-uvicorn",
    )
    server_thread.start()

    # Show a simple loading window while waiting for the server.
    print(f"Starting OMRChecker server on {url} …", flush=True)
    if not _wait_for_server(HOST, port, READY_TIMEOUT):
        print(
            f"Server did not start within {READY_TIMEOUT}s. "
            "Check for errors above.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Server ready — opening window.", flush=True)

    window = webview.create_window(
        WINDOW_TITLE,
        url,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        min_size=(800, 600),
        js_api=DesktopApi(url),
        # Allow the page to resize the window (some UIs use this)
        resizable=True,
    )

    # pywebview.start() blocks until the window is closed.
    # debug=False keeps the DevTools console closed by default; set to True
    # during development to inspect the page.
    webview.start(debug=False)

    # When the window closes, the process exits naturally because the server
    # thread is a daemon.
    sys.exit(0)


if __name__ == "__main__":
    # MUST be the first call inside the ``__main__`` guard for PyInstaller
    # on Windows. Without it, every child process spawned by the OMR
    # ProcessPoolExecutor / PDF splitter re-runs ``main()`` and forks a
    # second window — eventually exhausting OS resources.
    multiprocessing.freeze_support()
    main()
