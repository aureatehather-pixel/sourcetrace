"""Platform-specific event loop selection.

psycopg 3's async mode cannot run on Windows' default ProactorEventLoop, which
is what ``asyncio.run`` and uvicorn both pick there. Every entrypoint that opens
an async database connection routes through here so the failure is fixed in one
place rather than rediscovered per script.

On non-Windows platforms this is a no-op.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Coroutine
from typing import Any

# Assigned rather than tested inline so type checkers keep both branches live:
# this module is cross-platform even when analysed on one platform.
IS_WINDOWS: bool = sys.platform == "win32"


def psycopg_loop_factory() -> Callable[[], asyncio.AbstractEventLoop] | None:
    """Return the loop factory psycopg needs, or None to accept the default."""
    if IS_WINDOWS:
        return asyncio.SelectorEventLoop
    return None


def run_async[T](coro: Coroutine[Any, Any, T]) -> T:
    """``asyncio.run`` on a loop psycopg can actually use."""
    with asyncio.Runner(loop_factory=psycopg_loop_factory()) as runner:
        return runner.run(coro)
