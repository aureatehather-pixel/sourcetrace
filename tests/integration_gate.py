"""The check that stops a Docker-less run from looking green.

The integration suite runs against a real, disposable PostgreSQL 16 container.
When Docker was unavailable the session fixture called `pytest.skip`, so 222
tests silently dropped out and `pytest` still exited 0, printing
"270 passed, 222 skipped". Every deterministic gate the project relies on then
reported success while nothing about JSONB, TIMESTAMPTZ, CHECK constraints,
composite foreign keys or migrations had actually been exercised. A gate that
cannot fire is worse than no gate: it is a gate you trust.

So selection, not availability, decides. If a run *selected* integration tests
and Docker cannot serve them, the run fails loudly with one message rather than
skipping 222 times. The supported way to work without Docker is to say so —
`pytest -m "not integration"` — which deselects them and never reaches here.

The decision is a pure function so both of its answers are testable without
stopping Docker.
"""

from __future__ import annotations

MARKER = "integration"

_MESSAGE = """\
Docker is unavailable, but {count} integration test(s) were selected.

These run against a real PostgreSQL 16 container via Testcontainers. Skipping
them would report a green suite that verified nothing about the database, so
this run fails instead.

Start Docker Desktop and re-run, or ask for the unit suite explicitly:

    uv run pytest -m "not integration"\
"""


def integration_gate_message(*, docker_available: bool, selected_integration_tests: int) -> str:
    """The failure message for this run, or `""` when the run may proceed.

    Returns a message only when integration tests were selected *and* Docker
    cannot run them. Selecting none is a legitimate choice and never fails;
    having Docker is never a problem.
    """
    if docker_available or selected_integration_tests == 0:
        return ""
    return _MESSAGE.format(count=selected_integration_tests)
