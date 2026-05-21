"""Shared worker-capacity gate for CPU-bound local process pools.

The web UI can run more than one CPU-heavy operation at a time: PDF page
rendering and OMR processing both use ``ProcessPoolExecutor``. Without a shared
budget, an upload during an OMR run can spawn more worker processes than the
machine has cores, which hurts throughput on Windows laptops and makes AV
scanning contention worse.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from collections.abc import Iterator

logger = logging.getLogger(__name__)


def _default_capacity() -> int:
    env_val = os.environ.get("OMR_WEBUI_WORKER_CAPACITY")
    if env_val:
        try:
            return max(1, min(64, int(env_val)))
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid OMR_WEBUI_WORKER_CAPACITY=%r", env_val)
    return max(1, os.cpu_count() or 2)


class WorkerCapacity:
    """A small weighted semaphore for process-pool worker slots."""

    def __init__(self, capacity: int | None = None) -> None:
        self._capacity = max(1, int(capacity or _default_capacity()))
        self._in_use = 0
        self._condition = threading.Condition()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def in_use(self) -> int:
        with self._condition:
            return self._in_use

    @contextlib.contextmanager
    def reserve(
        self,
        requested: int,
        *,
        label: str,
        minimum: int = 1,
    ) -> Iterator[int]:
        """Reserve up to ``requested`` slots and yield the granted amount.

        The call waits only until at least ``minimum`` slots are available, then
        grants as many slots as possible up to ``requested``. This keeps the app
        responsive: a PDF upload during a large OMR run can still make slow
        progress with one worker instead of spawning past the CPU budget.
        """
        requested = max(1, min(int(requested), self._capacity))
        minimum = max(1, min(int(minimum), requested))
        with self._condition:
            while self._capacity - self._in_use < minimum:
                logger.info(
                    "Worker capacity waiting | label=%s | requested=%d | "
                    "minimum=%d | in_use=%d | capacity=%d",
                    label, requested, minimum, self._in_use, self._capacity,
                )
                self._condition.wait(timeout=5.0)

            available = self._capacity - self._in_use
            granted = min(requested, available)
            self._in_use += granted
            logger.info(
                "Worker capacity reserved | label=%s | requested=%d | "
                "granted=%d | in_use=%d | capacity=%d",
                label, requested, granted, self._in_use, self._capacity,
            )

        try:
            yield granted
        finally:
            with self._condition:
                self._in_use -= granted
                self._condition.notify_all()
                logger.info(
                    "Worker capacity released | label=%s | released=%d | "
                    "in_use=%d | capacity=%d",
                    label, granted, self._in_use, self._capacity,
                )


_WORKER_CAPACITY = WorkerCapacity()


def worker_capacity_limit() -> int:
    return _WORKER_CAPACITY.capacity


def reserve_worker_capacity(
    requested: int,
    *,
    label: str,
    minimum: int = 1,
) -> contextlib.AbstractContextManager[int]:
    return _WORKER_CAPACITY.reserve(requested, label=label, minimum=minimum)


def _reset_for_tests(capacity: int | None = None) -> None:
    """Reset the singleton capacity gate for deterministic tests."""
    global _WORKER_CAPACITY
    _WORKER_CAPACITY = WorkerCapacity(capacity)
