"""``python -m intelligence_engine.cli`` — the same entry point as ``pie``.

Exists so the CLI is reachable before ``uv sync`` has installed the console
script, and so a virtualenv that has the package but not its scripts on PATH can
still run it.
"""

from __future__ import annotations

import sys

from intelligence_engine.cli import main

if __name__ == "__main__":
    sys.exit(main())
