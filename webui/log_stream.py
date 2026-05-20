"""In-process log streaming for the OMRChecker UI.

A custom :class:`logging.Handler` pushes formatted records into:

* A fixed-size ring buffer (replayed to new SSE clients so they see recent
  history immediately on connect).
* A sequence-numbered poll buffer used by the ``/api/v1/logs/poll`` endpoint
  so browser clients (especially WebView2) can retrieve logs via plain HTTP
  GET instead of SSE, which has known buffering issues in some WebView2 builds.
* Per-client :class:`asyncio.Queue` instances that are drained by the SSE
  endpoint (kept for non-WebView2 clients and future use).

Usage
-----
Call :func:`attach` once at server startup with the running event loop.
Call :func:`detach` at shutdown.  The generator returned by :func:`stream`
is consumed by the ``/api/v1/logs/stream`` SSE endpoint.  :func:`poll` is
called by the ``/api/v1/logs/poll`` endpoint.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import threading
from typing import AsyncGenerator

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_MAX_HISTORY = 500
_ring: collections.deque[str] = collections.deque(maxlen=_MAX_HISTORY)

# Sequence-numbered ring for polling clients.
_poll_ring: collections.deque[tuple[int, str]] = collections.deque(maxlen=_MAX_HISTORY)
_poll_next_seq: int = 0

_clients: set[asyncio.Queue[str | None]] = set()
_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None

# ---------------------------------------------------------------------------
# Logging handler
# ---------------------------------------------------------------------------

_FMT = logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)


class _StreamHandler(logging.Handler):
    """Append records to the ring buffer and notify live SSE clients."""

    def emit(self, record: logging.LogRecord) -> None:
        global _poll_next_seq
        try:
            msg = self.format(record)
        except Exception:  # noqa: BLE001
            return

        with _lock:
            _ring.append(msg)
            seq = _poll_next_seq
            _poll_next_seq += 1
            _poll_ring.append((seq, msg))
            clients = list(_clients)

        if _loop is None or _loop.is_closed():
            return

        for q in clients:
            try:
                _loop.call_soon_threadsafe(q.put_nowait, msg)
            except Exception:  # noqa: BLE001
                pass


_handler = _StreamHandler()
_handler.setFormatter(_FMT)
_handler.setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def attach(loop: asyncio.AbstractEventLoop) -> None:
    """Register the streaming handler on the root logger.

    Must be called once the asyncio event loop is running (e.g. inside the
    FastAPI ``lifespan`` context manager).
    """
    global _loop
    _loop = loop
    root = logging.getLogger()
    # Guard against frozen-bundle cases where src/logger.py's basicConfig
    # was not imported, leaving the root logger at its default WARNING level.
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)
    if _handler not in root.handlers:
        root.addHandler(_handler)


def detach() -> None:
    """Remove the streaming handler.  Called at server shutdown."""
    global _loop
    logging.getLogger().removeHandler(_handler)
    # Signal all waiting clients to close
    if _loop and not _loop.is_closed():
        with _lock:
            clients = list(_clients)
        for q in clients:
            try:
                _loop.call_soon_threadsafe(q.put_nowait, None)
            except Exception:  # noqa: BLE001
                pass
    _loop = None


def poll(since: int = -1) -> dict:
    """Return log entries with sequence number > *since*.

    Used by the ``/api/v1/logs/poll`` endpoint so that browser clients can
    retrieve logs via plain HTTP GET, avoiding SSE buffering issues in WebView2.

    Returns ``{"entries": [{"seq": int, "msg": str}, ...], "latest_seq": int}``.
    """
    with _lock:
        entries = [(s, m) for s, m in _poll_ring if s > since]
        latest = _poll_ring[-1][0] if _poll_ring else max(since, 0)
    return {
        "entries": [{"seq": s, "msg": m} for s, m in entries],
        "latest_seq": latest,
    }


async def stream() -> AsyncGenerator[str, None]:
    """Async generator that yields SSE-formatted log lines.

    * Replays the ring-buffer history first so the client sees recent logs
      immediately.
    * Then streams live records as they arrive.
    * Cleans up the client queue on disconnect / generator close.
    """
    q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=2000)

    # Register and snapshot history atomically under the same lock so there
    # is no window where a log message can arrive after the snapshot but before
    # the client queue is in _clients (which would silently drop it).
    with _lock:
        history = list(_ring)
        _clients.add(q)

    try:
        for line in history:
            yield f"data: {line}\n\n"
        while True:
            msg = await q.get()
            if msg is None:
                break
            yield f"data: {msg}\n\n"
    finally:
        with _lock:
            _clients.discard(q)
