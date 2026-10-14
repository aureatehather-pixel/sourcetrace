"""Which event loop the async tests run on, decided once per platform.

psycopg 3's async mode cannot use Windows' default ProactorEventLoop, so on
Windows every async test must run on the loop `psycopg_loop_factory` names.
Everywhere else pytest-asyncio's default loop is already correct.

The decision is not "return None from the hook to mean the default". In the
locked pytest-asyncio, implementing `pytest_asyncio_loop_factories` at all is a
commitment: once an implementation exists, its result must be a non-empty
mapping of names to callables, and `None` is a `UsageError` that fails
collection. That is exactly how macOS and Linux lost the whole suite while
Windows stayed green. So the platform decides whether the hook *exists*, and
when it does, it can only return a mapping.

The decision is a pure function so both answers are testable on one machine.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

LoopFactory = Callable[[], asyncio.AbstractEventLoop]


def required_loop_factories(factory: LoopFactory | None) -> dict[str, LoopFactory] | None:
    """The mapping the hook must return, or None when the hook must not be defined."""
    if factory is None:
        return None
    return {"asyncio": factory}
