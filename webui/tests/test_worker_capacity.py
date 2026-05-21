"""Tests for shared worker-capacity backpressure."""

from __future__ import annotations

import threading
import time


def test_worker_capacity_grants_partial_capacity_without_oversubscription() -> None:
    from webui.services import capacity

    capacity._reset_for_tests(4)
    try:
        with capacity.reserve_worker_capacity(3, label="test-a") as first:
            assert first == 3
            assert capacity._WORKER_CAPACITY.in_use == 3
            with capacity.reserve_worker_capacity(3, label="test-b") as second:
                assert second == 1
                assert capacity._WORKER_CAPACITY.in_use == 4
        assert capacity._WORKER_CAPACITY.in_use == 0
    finally:
        capacity._reset_for_tests()


def test_worker_capacity_waits_when_no_minimum_capacity_available() -> None:
    from webui.services import capacity

    capacity._reset_for_tests(2)
    acquired: list[int] = []

    def reserve_after_holder() -> None:
        with capacity.reserve_worker_capacity(1, label="waiter") as granted:
            acquired.append(granted)

    try:
        with capacity.reserve_worker_capacity(2, label="holder"):
            thread = threading.Thread(target=reserve_after_holder)
            thread.start()
            time.sleep(0.2)
            assert acquired == []

        thread.join(timeout=2)
        assert acquired == [1]
        assert capacity._WORKER_CAPACITY.in_use == 0
    finally:
        capacity._reset_for_tests()


def test_worker_settings_are_clamped_to_shared_capacity() -> None:
    from webui.services import capacity
    from webui.services.batches import _default_pdf_split_workers
    from webui.services.omr import _coerce_max_workers
    from webui.settings import Settings

    capacity._reset_for_tests(4)
    try:
        assert _default_pdf_split_workers(Settings(pdf_split_workers=32)) == 4
        assert _coerce_max_workers(32) == 4
    finally:
        capacity._reset_for_tests()
