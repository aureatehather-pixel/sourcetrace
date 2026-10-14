"""The async suite collects on every platform, not only the one it was written on.

The defect this replaces: the loop-factory hook returned `None` off Windows to
mean "use the default". The locked pytest-asyncio rejects that, so on macOS and
Linux collection failed before a single test ran, while Windows stayed green.

Both platform answers are exercised here on one machine, against the real
plugin, by running a small pytest session in a subprocess for each.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import tests.conftest as conftest
from intelligence_engine.core.platform import psycopg_loop_factory
from tests.loop_factories import required_loop_factories

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_INI = """\
[pytest]
asyncio_mode = auto
asyncio_default_fixture_loop_scope = function
"""

_TEST = """\
import asyncio


async def test_runs_on_the_expected_loop():
    assert isinstance(asyncio.get_running_loop(), {expected})
"""


def _run_session(
    tmp_path: Path, conftest_body: str, expected_loop: str
) -> subprocess.CompletedProcess[str]:
    (tmp_path / "pytest.ini").write_text(_INI, encoding="utf-8")
    (tmp_path / "conftest.py").write_text(textwrap.dedent(conftest_body), encoding="utf-8")
    (tmp_path / "test_loop.py").write_text(_TEST.format(expected=expected_loop), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    return subprocess.run(  # noqa: S603 - fixed argv, shell=False deliberately
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


# The same wiring tests/conftest.py uses, with the platform's answer injected.
_WIRED_CONFTEST = """\
    import asyncio
    from tests.loop_factories import required_loop_factories

    _LOOP_FACTORIES = required_loop_factories({factory})

    if _LOOP_FACTORIES is not None:
        _REQUIRED = _LOOP_FACTORIES

        def pytest_asyncio_loop_factories(config, item):
            return _REQUIRED
    """


def test_no_custom_loop_means_no_hook() -> None:
    assert required_loop_factories(None) is None


def test_a_custom_loop_is_the_only_named_factory() -> None:
    assert required_loop_factories(asyncio.SelectorEventLoop) == {
        "asyncio": asyncio.SelectorEventLoop
    }


def test_the_conftest_defines_the_hook_exactly_where_psycopg_needs_it() -> None:
    assert hasattr(conftest, "pytest_asyncio_loop_factories") is (
        psycopg_loop_factory() is not None
    )


def test_the_default_loop_path_collects_and_runs(tmp_path: Path) -> None:
    """The macOS and Linux path: no hook, the plugin's own loop."""
    result = _run_session(
        tmp_path, _WIRED_CONFTEST.format(factory="None"), "asyncio.AbstractEventLoop"
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout


def test_the_selector_loop_path_collects_and_runs_on_that_loop(tmp_path: Path) -> None:
    """The Windows path: every async test runs on the loop psycopg can use."""
    result = _run_session(
        tmp_path,
        _WIRED_CONFTEST.format(factory="asyncio.SelectorEventLoop"),
        "asyncio.SelectorEventLoop",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout


@pytest.mark.parametrize("returned", ["None", "{}"])
def test_the_plugin_still_refuses_a_hook_that_declines(tmp_path: Path, returned: str) -> None:
    """Pins the contract that made `return None` a collection failure.

    If a future pytest-asyncio starts accepting this again, the design above is
    still correct, but this test should be revisited rather than trusted.
    """
    body = f"""\
        def pytest_asyncio_loop_factories(config, item):
            return {returned}
        """
    result = _run_session(tmp_path, body, "asyncio.AbstractEventLoop")

    assert result.returncode != 0
    assert "pytest_asyncio_loop_factories must return a non-empty mapping" in (
        result.stdout + result.stderr
    )
