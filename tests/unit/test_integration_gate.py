"""Proof that the Docker gate fires, not merely that it exists.

The defect this replaces was a gate that only ever reported success: when Docker
was down the integration suite skipped and the exit status stayed 0. So the
rejection case is the point of this file, and both halves are covered — the pure
decision, and the conftest hook that turns it into a failed run.
"""

from __future__ import annotations

from typing import Any

import pytest

import tests.conftest as conftest
from tests.integration_gate import MARKER, integration_gate_message


def test_docker_up_with_integration_selected_permits_the_run() -> None:
    assert integration_gate_message(docker_available=True, selected_integration_tests=222) == ""


def test_no_integration_selected_permits_the_run_without_docker() -> None:
    """`pytest -m "not integration"` is the supported no-Docker path."""
    assert integration_gate_message(docker_available=False, selected_integration_tests=0) == ""


def test_docker_down_with_integration_selected_is_refused() -> None:
    message = integration_gate_message(docker_available=False, selected_integration_tests=222)

    assert message != ""
    assert "222" in message
    assert 'pytest -m "not integration"' in message


class _FakeItem:
    """The only part of `pytest.Item` the hook touches."""

    def __init__(self, *, integration: bool) -> None:
        self._integration = integration

    def get_closest_marker(self, name: str) -> object | None:
        if name == MARKER and self._integration:
            return object()
        return None


def _run_hook(*, docker_available: bool, items: list[Any]) -> None:
    monkeypatched = pytest.MonkeyPatch()
    try:
        monkeypatched.setattr(conftest, "_docker_available", lambda: docker_available)
        conftest.pytest_collection_modifyitems(pytest.Config, items)  # type: ignore[arg-type]
    finally:
        monkeypatched.undo()


def test_hook_raises_usage_error_when_docker_is_down() -> None:
    """The wiring, not just the decision: a real run would stop here."""
    items = [_FakeItem(integration=True), _FakeItem(integration=False)]

    with pytest.raises(pytest.UsageError) as raised:
        _run_hook(docker_available=False, items=items)

    assert "1 integration test(s) were selected" in str(raised.value)


def test_hook_is_silent_when_only_unit_tests_are_selected() -> None:
    _run_hook(docker_available=False, items=[_FakeItem(integration=False)])


def test_hook_is_silent_when_docker_is_up() -> None:
    _run_hook(docker_available=True, items=[_FakeItem(integration=True)])
