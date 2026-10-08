"""Fault-tolerance primitives for tool I/O: hard timeouts, exponential backoff, TTL cache.

CrewAI calls ``BaseTool._run`` synchronously from agent worker threads, so these
helpers are synchronous and thread-safe. LLM-side throttling is handled by
CrewAI itself (``BaseLLM`` wraps calls in jittered 429 retry) plus
``LLM(max_retries=...)`` and the crew/agent ``max_rpm`` limiter.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Generic, TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")

TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    httpx.TransportError,
    httpx.HTTPStatusError,
    TimeoutError,
    ConnectionError,
)

# Shared pool for timeout enforcement; daemon threads never block interpreter exit.
_POOL = ThreadPoolExecutor(max_workers=16, thread_name_prefix="agri-tool-io")


class ToolTimeoutError(TimeoutError):
    """Raised when a tool data call exceeds its hard deadline."""


def run_with_timeout(fn: Callable[[], T], timeout_s: float) -> T:
    """Run ``fn`` with a hard wall-clock deadline.

    The worker thread cannot be killed, but the agent is unblocked immediately
    and the late result is discarded.
    """
    future = _POOL.submit(fn)
    try:
        return future.result(timeout=timeout_s)
    except FutureTimeout as exc:
        future.cancel()
        raise ToolTimeoutError(f"data call exceeded {timeout_s:.1f}s") from exc


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {408, 425, 429, 500, 502, 503, 504}
    return isinstance(exc, TRANSIENT_ERRORS)


@dataclass(frozen=True)
class BackoffPolicy:
    max_attempts: int = 3
    initial_s: float = 0.5
    max_s: float = 8.0
    multiplier: float = 2.0
    jitter: float = 0.25  # +/- fraction

    def delay(self, attempt: int, rng: random.Random | None = None) -> float:
        """Delay before retry number ``attempt`` (1-based), with symmetric jitter."""
        base = min(self.max_s, self.initial_s * self.multiplier ** (attempt - 1))
        r = (rng or random).uniform(-self.jitter, self.jitter)
        return max(0.0, base * (1 + r))


def retry_with_backoff(
    fn: Callable[[], T],
    policy: BackoffPolicy,
    *,
    label: str,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``fn``, retrying transient failures with exponential backoff and jitter."""
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt >= policy.max_attempts or not is_retryable(exc):
                raise
            delay = policy.delay(attempt)
            logger.info(
                "%s transient failure (attempt %d/%d): %r; retrying in %.2fs",
                label,
                attempt,
                policy.max_attempts,
                exc,
                delay,
            )
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


class TTLCache(Generic[T]):
    """Thread-safe in-process TTL cache holding last-known-good data per key."""

    def __init__(self, ttl_s: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._ttl_s = ttl_s
        self._clock = clock
        self._items: dict[str, tuple[float, T]] = {}
        self._lock = threading.Lock()

    def get(self, key: str, *, allow_stale: bool = False) -> T | None:
        with self._lock:
            item = self._items.get(key)
        if item is None:
            return None
        stored_at, value = item
        if not allow_stale and self._clock() - stored_at > self._ttl_s:
            return None
        return value

    def set(self, key: str, value: T) -> None:
        with self._lock:
            self._items[key] = (self._clock(), value)

    def __len__(self) -> int:
        return len(self._items)
