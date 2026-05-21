"""Tests for the persistent ProcessPoolExecutor (Fix #4).

Verifies that _get_pdf_render_pool returns a singleton pool, replaces it when
the worker count changes, survives failing tasks, and registers an atexit hook.
"""

from __future__ import annotations

import atexit
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_pdf_bytes(page_count: int = 2) -> bytes:
    """Return minimal PDF bytes with *page_count* blank pages."""
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for i in range(page_count):
        p = doc.new_page(width=72, height=72)
        p.insert_text((10, 40), str(i))
    data = doc.tobytes()
    doc.close()
    return data


class _WrappedPPE(ProcessPoolExecutor):
    """ProcessPoolExecutor that increments a shared counter on each instantiation."""

    _instance_count: int = 0

    def __init__(self, *args, **kwargs):
        _WrappedPPE._instance_count += 1
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# Pool singleton / reuse
# ---------------------------------------------------------------------------

def test_pool_is_reused_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """_get_pdf_render_pool must return the same object when workers is unchanged."""
    import webui.services.batches as mod

    # Isolate module pool state for this test.
    monkeypatch.setattr(mod, "_PDF_RENDER_POOL", None)
    monkeypatch.setattr(mod, "_PDF_RENDER_POOL_WORKERS", 0)

    _WrappedPPE._instance_count = 0
    monkeypatch.setattr(mod, "ProcessPoolExecutor", _WrappedPPE)

    pool_a = mod._get_pdf_render_pool(2)
    pool_b = mod._get_pdf_render_pool(2)

    assert pool_a is pool_b, "Same worker count must return the same pool instance"
    assert _WrappedPPE._instance_count == 1, (
        f"Pool must be constructed exactly once; got {_WrappedPPE._instance_count}"
    )


def test_pool_is_resized_when_worker_count_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_get_pdf_render_pool must create a new pool when workers differs."""
    import webui.services.batches as mod

    monkeypatch.setattr(mod, "_PDF_RENDER_POOL", None)
    monkeypatch.setattr(mod, "_PDF_RENDER_POOL_WORKERS", 0)

    _WrappedPPE._instance_count = 0
    monkeypatch.setattr(mod, "ProcessPoolExecutor", _WrappedPPE)

    pool_2 = mod._get_pdf_render_pool(2)
    pool_4 = mod._get_pdf_render_pool(4)

    assert pool_2 is not pool_4, "Different worker count must yield a different pool"
    assert _WrappedPPE._instance_count == 2, (
        f"Two pools must be created (one per worker count); got {_WrappedPPE._instance_count}"
    )


# ---------------------------------------------------------------------------
# Pool resilience
# ---------------------------------------------------------------------------

def test_pool_survives_a_failing_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker exception must not break the pool for subsequent submissions.

    We use `json.loads` with invalid JSON as the "failing" task because it is a
    module-level callable (picklable on Windows spawn) and raises a well-known
    exception type.  We then verify the pool still handles a healthy task.
    """
    import webui.services.batches as mod

    # Reset so we get a fresh real pool (not a mock).
    monkeypatch.setattr(mod, "_PDF_RENDER_POOL", None)
    monkeypatch.setattr(mod, "_PDF_RENDER_POOL_WORKERS", 0)

    pool = mod._get_pdf_render_pool(1)

    # Submit a task that raises in the worker process (invalid JSON).
    fut = pool.submit(json.loads, "{ this is not valid JSON !!!")
    with pytest.raises(json.JSONDecodeError):
        fut.result()

    # The pool must still be usable after the worker raised.
    result = pool.submit(json.loads, '{"ok": true}').result()
    assert result == {"ok": True}, (
        f"Pool should still work after a failing task; got {result}"
    )


# ---------------------------------------------------------------------------
# Atexit registration
# ---------------------------------------------------------------------------

def test_atexit_handler_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling _get_pdf_render_pool must register the atexit cleanup handler.

    We verify this by patching ``atexit.register`` to capture registrations —
    more portable than introspecting ``atexit._exithandlers`` (which is a C
    internal unavailable in Python ≥ 3.12 without debug builds).
    """
    import webui.services.batches as mod

    registered: list = []
    original_register = atexit.register

    def _spy_register(func, *args, **kwargs):
        registered.append(func)
        return original_register(func, *args, **kwargs)

    monkeypatch.setattr(atexit, "register", _spy_register)
    monkeypatch.setattr(mod, "_ATEXIT_REGISTERED", False)
    monkeypatch.setattr(mod, "_PDF_RENDER_POOL", None)
    monkeypatch.setattr(mod, "_PDF_RENDER_POOL_WORKERS", 0)

    mod._get_pdf_render_pool(1)

    assert mod._shutdown_pdf_render_pool in registered, (
        "_shutdown_pdf_render_pool must be passed to atexit.register on first call"
    )


def test_atexit_handler_registered_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated _get_pdf_render_pool calls must not register duplicate atexit hooks."""
    import webui.services.batches as mod

    registered: list = []
    original_register = atexit.register

    def _spy_register(func, *args, **kwargs):
        registered.append(func)
        return original_register(func, *args, **kwargs)

    monkeypatch.setattr(atexit, "register", _spy_register)
    monkeypatch.setattr(mod, "_ATEXIT_REGISTERED", False)
    monkeypatch.setattr(mod, "_PDF_RENDER_POOL", None)
    monkeypatch.setattr(mod, "_PDF_RENDER_POOL_WORKERS", 0)

    mod._get_pdf_render_pool(1)
    mod._get_pdf_render_pool(1)  # same count — pool reused, no new atexit
    mod._get_pdf_render_pool(2)  # resize triggers new pool, but NOT new atexit

    shutdown_registrations = [
        f for f in registered if f is mod._shutdown_pdf_render_pool
    ]
    assert len(shutdown_registrations) == 1, (
        f"Expected exactly 1 atexit registration; got {len(shutdown_registrations)}"
    )
