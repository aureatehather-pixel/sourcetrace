"""Where the pipeline stops and asks a human.

The deterministic matcher refuses to guess in exactly two situations: nothing on
file matches a name (`no_candidate`), and more than one thing does
(`ambiguous_candidates`). Both are correct refusals — a false merge fuses two
real people permanently — and both need a judgement the system does not have.

An orchestrator that made those calls itself would be inventing intelligence
semantics to make the happy path look seamless, which is the one thing this
command must not do. So the ambiguity is surfaced rather than resolved: a
`Decider` answers, and there are exactly two implementations — a person at a
terminal, and a non-interactive mode that answers "skip" every time so the run
can finish unattended.

Note what a `Decider` cannot express. There is no "pick the most likely
candidate" answer, and no way to resolve without either naming an existing entity
or explicitly creating one. `Skip` is a first-class outcome, not a deferral:
unresolved is a permanently valid state in this system.
"""

from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass
from typing import Protocol, TextIO

from intelligence_engine.entities.schemas import EntityRead, EntityType


@dataclass(frozen=True)
class Candidate:
    entity_id: uuid.UUID
    canonical_name: str
    disambiguator: str | None

    @classmethod
    def from_entity(cls, entity: EntityRead) -> Candidate:
        return cls(
            entity_id=entity.id,
            canonical_name=entity.canonical_name,
            disambiguator=entity.disambiguator,
        )


@dataclass(frozen=True)
class Question:
    """Everything a human needs to answer well, and nothing they must look up.

    The grounding quote is here because the name alone is not enough to tell two
    same-named people apart, and going to fetch the evidence by hand is exactly
    the friction that stops resolutions from happening.
    """

    name: str
    entity_type: EntityType
    grounding_quote: str
    source_label: str | None
    extraction_item_id: uuid.UUID
    role: str
    candidates: tuple[Candidate, ...] = ()


@dataclass(frozen=True)
class CreateEntity:
    """Assert this identity exists, then link the mention to it."""

    canonical_name: str
    disambiguator: str | None


@dataclass(frozen=True)
class UseEntity:
    """Link the mention to an entity already on file."""

    entity_id: uuid.UUID


@dataclass(frozen=True)
class Skip:
    """Leave it unresolved."""


Answer = CreateEntity | UseEntity | Skip


class Decider(Protocol):
    def decide(self, question: Question) -> Answer: ...


class AlwaysSkip:
    """Non-interactive mode: never resolves, never creates, never blocks.

    Used when `--non-interactive` is given or stdin is not a terminal. The
    pipeline reports every question it could not ask, with the command that
    answers it, so nothing is lost — only deferred to a human who is present.
    """

    def decide(self, question: Question) -> Answer:
        return Skip()


class Prompt:
    """A person at a terminal.

    Writes to stderr on purpose: `--json` keeps stdout machine-readable even
    while a human is being asked questions on the same terminal.
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self._out: TextIO = stream if stream is not None else sys.stderr

    def decide(self, question: Question) -> Answer:
        self._show(question)
        while True:
            choice = self._ask().strip().casefold()
            if choice in {"s", "skip", ""}:
                return Skip()
            if choice in {"c", "create"} and not question.candidates:
                return self._create(question)
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(question.candidates):
                    return UseEntity(question.candidates[index].entity_id)
            self._write("  not one of the options; try again\n")

    def _create(self, question: Question) -> Answer:
        self._write(f"  canonical name [{question.name}]: ")
        name = self._ask().strip() or question.name
        self._write("  disambiguator, e.g. 'backend engineer at DevGuild' (blank for none): ")
        return CreateEntity(canonical_name=name, disambiguator=self._ask().strip() or None)

    def _show(self, question: Question) -> None:
        self._write("\n")
        if question.candidates:
            self._write(
                f'  "{question.name}" ({question.entity_type}) matches '
                f"{len(question.candidates)} entities on file.\n"
            )
        else:
            self._write(f'  No entity on file for "{question.name}" ({question.entity_type}).\n')
        self._write(f"  quote: {question.grounding_quote}\n")
        if question.source_label:
            self._write(f"  from:  {question.source_label}\n")
        for position, candidate in enumerate(question.candidates, start=1):
            suffix = f" - {candidate.disambiguator}" if candidate.disambiguator else ""
            self._write(
                f"    [{position}] {candidate.canonical_name}{suffix}  ({candidate.entity_id})\n"
            )
        if not question.candidates:
            self._write(f"    [c] create a new {question.entity_type} entity\n")
        self._write("    [s] skip, leave unresolved\n")

    def _write(self, text: str) -> None:
        self._out.write(text)
        self._out.flush()

    def _ask(self) -> str:
        self._write("> ")
        try:
            return input()
        except EOFError:
            return "s"
