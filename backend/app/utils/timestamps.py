from __future__ import annotations

import time


def now_ms() -> int:
    """Wall-clock milliseconds."""
    return int(time.time() * 1000)


def monotonic_ms() -> int:
    """Monotonic milliseconds for latency measurement."""
    return int(time.perf_counter() * 1000)
