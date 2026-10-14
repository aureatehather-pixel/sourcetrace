"""Engine and session lifecycle.

One engine per process, created at startup and disposed at shutdown. Sessions
are per-request and never global: a shared session across concurrent requests
corrupts transaction state.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# A dead database must fail fast rather than hang: an unbounded connect turns
# every probe and request into a timeout the caller cannot distinguish from load.
CONNECT_TIMEOUT_SECONDS = 5


def create_engine(database_url: str, *, echo: bool = False) -> AsyncEngine:
    return create_async_engine(
        database_url,
        echo=echo,
        pool_pre_ping=True,  # a recycled/dead connection should not surface as a 500
        pool_size=5,
        max_overflow=5,
        connect_args={"connect_timeout": CONNECT_TIMEOUT_SECONDS},
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=engine,
        expire_on_commit=False,  # response serialisation must not trigger lazy IO
        autoflush=False,
    )


async def session_scope(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Yield a session, rolling back anything left uncommitted.

    Commits are explicit and owned by the service layer, so an exception - or a
    handler that simply forgot to commit - can never leave a partial write.
    """
    session = factory()
    try:
        yield session
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
