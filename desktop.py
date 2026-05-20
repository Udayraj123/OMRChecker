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

import os
import socket
import sys
import threading
import time

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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HOST = "127.0.0.1"
PORT = 5050
READY_TIMEOUT = 30  # seconds to wait for the server to accept connections
WINDOW_TITLE = "OMRChecker"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 900


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
    main()
