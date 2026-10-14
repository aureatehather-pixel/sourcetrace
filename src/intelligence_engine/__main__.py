"""``python -m intelligence_engine`` - run the API server.

Exists because uvicorn's CLI selects Windows' ProactorEventLoop, which psycopg's
async mode cannot use. This runs the same server on a compatible loop.
"""

from __future__ import annotations

import asyncio
import os

import uvicorn

from intelligence_engine.core.platform import psycopg_loop_factory


def main() -> None:
    config = uvicorn.Config(
        "intelligence_engine.main:create_app",
        factory=True,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        log_config=None,  # our JSON formatter owns stdout
    )
    server = uvicorn.Server(config)
    with asyncio.Runner(loop_factory=psycopg_loop_factory()) as runner:
        runner.run(server.serve())


if __name__ == "__main__":
    main()
