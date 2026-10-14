"""Extraction application behaviour.

Owns the ordering that makes derived information safe:

    reuse check
      -> record the attempt (committed, so history survives a crash)
      -> call the provider
      -> strict validation
      -> semantic grounding check
      -> ONE transaction: mark succeeded + insert every item

The `running` row is committed before the network call on purpose. If the
process dies mid-call, the attempt is still in the history as `running` rather
than vanishing. That is honest: something did happen, and we do not know how it
ended.

Success is a single transaction, so a run can never be `succeeded` with only
some of its items stored.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import logging
import time
import uuid

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from intelligence_engine.core.errors import EvidenceNotFound, PersistenceError
from intelligence_engine.evidence.fingerprint import content_fingerprint
from intelligence_engine.evidence.models import EvidenceItem
from intelligence_engine.evidence.repository import EvidenceRepository
from intelligence_engine.extraction import versioning
from intelligence_engine.extraction.errors import (
    ExtractionError,
    ExtractionOverflow,
    IrreducibleOverflow,
)
from intelligence_engine.extraction.models import MAX_ERROR_MESSAGE_LENGTH, ExtractionRun
from intelligence_engine.extraction.prompt import SYSTEM_PROMPT, build_user_prompt
from intelligence_engine.extraction.provider import ExtractionProvider, ProviderResult, StopReason
from intelligence_engine.extraction.repository import ExtractionRepository
from intelligence_engine.extraction.schemas import (
    ExtractionOutput,
    ExtractionResponse,
    ExtractionRunList,
    ExtractionRunRead,
    RunStatus,
    wire_schema,
)
from intelligence_engine.extraction.validation import parse_output, verify_grounding
from intelligence_engine.extraction.windowing import (
    MIN_SUBDIVISIBLE_CHARS,
    Window,
    coverage,
    plan_windows,
    subdivide,
)

logger = logging.getLogger(__name__)

MAX_RUNS_LISTED = 100

#: How far one overflowing interval may be pursued before the descent is
#: declared irreducible. Each subdivision at least halves, so this is a bound on
#: the depth of a binary-ish tree rather than an arbitrary retry count: from the
#: 6,000-character default budget, six halvings reach roughly 90 characters,
#: which is already below any plausible unit of dense prose.
#:
#: It bounds cost as well as termination. A single pathological region can cost
#: at most `2**MAX_SUBDIVISION_DEPTH` leaf calls before failing loudly, and
#: real overflow in the corpus resolves in one or two levels, so this is a
#: guard rather than a working parameter.
MAX_SUBDIVISION_DEPTH = 6

#: Window budget used when no caller supplies one. Mirrors
#: `Settings.extraction_window_max_chars`, and is duplicated here for the same
#: reason the version constants are injectable: the service must be constructible
#: in a test without a settings object, and a silent disagreement between the two
#: is caught by `tests/unit/test_extraction_windowing_defaults.py`.
DEFAULT_WINDOW_MAX_CHARS = 6_000


@dataclasses.dataclass(frozen=True)
class _Source:
    """The facts about one evidence item that the extraction loop needs.

    A plain snapshot, taken once, deliberately **not** the ORM object. Recording
    a failed attempt rolls the session back, which expires every loaded
    instance; before adaptive extraction that was harmless because a failure
    always ended the run, but subdivision continues working afterwards and would
    touch an expired `EvidenceItem`, triggering a lazy refresh from inside a
    place that cannot await one.

    Snapshotting is also the more honest shape. The recursion depends on the
    source *text* and its identity, both of which are immutable for the life of
    the call; it has no business holding a handle to a mutable session-bound
    row whose state something else can change underneath it.
    """

    evidence_id: uuid.UUID
    content: str
    #: The provenance lines that describe the source itself — type, label, URL,
    #: capture time. Fixed for every interval, so computed once.
    provenance: tuple[str, ...]

    @property
    def length(self) -> int:
        return len(self.content)


@dataclasses.dataclass
class _Progress:
    """What one `extract` call has accumulated so far.

    Carried through the recursion rather than returned from it, because a
    subdivided interval contributes several runs at a depth the caller does not
    know about, and threading return values back up would make the traversal
    order part of the result. Order is imposed once at the end, from the
    offsets, which is the only ordering that stays correct under subdivision.

    **Read models, not ORM rows, and for a specific reason.** Recording a failed
    attempt rolls the session back, and a rollback expires every instance in the
    identity map — including runs that committed successfully several intervals
    ago. Reading one of their columns afterwards would trigger a lazy refresh
    from synchronous code and fail as `MissingGreenlet`. Converting each run the
    moment it is known good, while it is certainly fresh, means nothing here can
    be invalidated by a later sibling's failure. `expire_on_commit=False` covers
    the commit path; only rollback needs this, and only adaptive extraction ever
    continues past one.
    """

    runs: list[ExtractionRunRead] = dataclasses.field(default_factory=list)
    #: True only while *every* interval so far was served from cache, so a
    #: partially cached source honestly reports that it spent money.
    reused_all: bool = True


def _in_source_order(runs: list[ExtractionRunRead]) -> list[ExtractionRunRead]:
    """Runs ordered as the source reads, by the interval each one covers."""
    return sorted(runs, key=lambda run: (run.window_start, run.window_end))


class ExtractionService:
    def __init__(
        self,
        session: AsyncSession,
        provider: ExtractionProvider,
        *,
        extractor_name: str = versioning.EXTRACTOR_NAME,
        extractor_version: str = versioning.EXTRACTOR_VERSION,
        schema_version: int = versioning.SCHEMA_VERSION,
        window_max_chars: int = DEFAULT_WINDOW_MAX_CHARS,
    ) -> None:
        self._session = session
        self._provider = provider
        self._repository = ExtractionRepository(session)
        self._evidence = EvidenceRepository(session)
        # Injectable so tests can prove a version bump forces reprocessing.
        self._extractor_name = extractor_name
        self._extractor_version = extractor_version
        self._schema_version = schema_version
        # Injectable so a test can force multi-window behaviour on small
        # fixtures rather than carrying a 60KB document to prove it.
        self._window_max_chars = window_max_chars

    async def extract(self, evidence_id: uuid.UUID) -> ExtractionResponse:
        """Extract one evidence item, in as many intervals as its density requires.

        The caller says `extract(evidence_id)` and nothing else: interval
        planning, reuse and subdivision are internal mechanics. There is no way
        to ask for a single interval, because an interval is not a thing anyone
        should have to know about to use the system.

        **Two different sizes, and only one of them is a guarantee.** The
        initial plan packs structural blocks up to a character budget, which is
        a *cost* choice: most intervals succeed, and each call pays a fixed
        prompt overhead, so packing generously is simply cheaper. It is not a
        correctness argument, and the real corpus proved three times that no
        budget could be. Correctness comes from what happens when an interval
        fails: if it failed *because it was too large for one bounded response*,
        it is deterministically split and its children are extracted instead.

        **Ordering, and why it stops at the first real failure.** Intervals are
        processed in source order, and a failure that is not an overflow aborts
        the rest. Every interval that already succeeded is committed and stays
        committed, so re-running resumes rather than restarts; continuing past a
        provider failure would instead keep paying a provider that has just
        demonstrated it is failing. The source is left honestly incomplete, its
        unread ranges are named, and the error propagates.
        """
        started = time.perf_counter()
        evidence = await self._evidence.get(evidence_id)
        if evidence is None:
            raise EvidenceNotFound(evidence_id)

        # Snapshotted before the first call: recording a failed attempt rolls
        # the session back, and subdivision keeps working afterwards.
        source = _Source(
            evidence_id=evidence.id,
            content=evidence.content,
            provenance=_source_provenance(evidence),
        )

        plan = plan_windows(source.content, max_chars=self._window_max_chars)
        state = _Progress()

        for window in plan:
            await self._extract_interval(source, window, state=state, depth=0, started=started)

        covered = coverage(
            ((run.window_start, run.window_end) for run in state.runs), length=source.length
        )
        return ExtractionResponse(
            reused=state.reused_all,
            complete=covered.is_complete,
            source_length=source.length,
            uncovered=list(covered.gaps),
            runs=_in_source_order(state.runs),
        )

    async def _extract_interval(
        self,
        source: _Source,
        window: Window,
        *,
        state: _Progress,
        depth: int,
        started: float,
    ) -> None:
        """Extract one interval, subdividing it if — and only if — it overflows.

        The recursion is the whole architecture in one method. An overflow is
        not a failure of this interval so much as evidence that the question was
        too big, so the interval is split and the children asked instead. Every
        other failure is a failure, and is re-raised without splitting anything:
        subdividing after a timeout, a rate limit or a fabricated quote would
        multiply calls against a problem that has nothing to do with how much
        text was sent.

        Children replace their parent exactly — they tile the same characters —
        so coverage is unchanged by the fact that a subdivision happened, and a
        sibling that already succeeded is neither re-read nor renumbered.
        """
        window_text = window.text(source.content)
        window_hash = content_fingerprint(window_text)

        identity = self._provider.identity
        reusable = await self._repository.find_reusable_run(
            evidence_id=source.evidence_id,
            input_content_hash=window_hash,
            extractor_name=self._extractor_name,
            extractor_version=self._extractor_version,
            schema_version=self._schema_version,
            provider=identity.provider,
            model=identity.model,
            window_start=window.start,
            window_end=window.end,
        )
        if reusable is not None:
            logger.info(
                "extraction.reused",
                extra=self._log_fields(run=reusable, reused=True, duration_ms=_elapsed_ms(started)),
            )
            state.runs.append(ExtractionRunRead.model_validate(reusable))
            return

        state.reused_all = False
        run = await self._record_attempt(source, window=window, window_hash=window_hash)

        try:
            output, provider_result = await self._call_provider(
                source, window_text=window_text, window=window
            )
        except ExtractionOverflow as exc:
            # The one failure answered by asking a smaller question. The parent
            # attempt stays on record as failed — it really did happen, and it
            # really did cost money — but it contributes no coverage, so the
            # children must succeed for this region to count as read.
            await self._mark_failed(run, exc)
            await self._subdivide_and_retry(
                source, window, state=state, depth=depth, started=started, cause=exc
            )
            return
        except ExtractionError as exc:
            await self._mark_failed(run, exc)
            logger.warning(
                "extraction.failed",
                extra=self._log_fields(
                    run=run,
                    reused=False,
                    duration_ms=_elapsed_ms(started),
                    error_code=exc.error_code,
                    intervals_completed=len(state.runs),
                ),
            )
            raise

        await self._persist_success(run, output, provider_result)
        logger.info(
            "extraction.succeeded",
            extra=self._log_fields(
                run=run,
                reused=False,
                duration_ms=_elapsed_ms(started),
                item_count=len(run.items),
                depth=depth,
            ),
        )
        state.runs.append(ExtractionRunRead.model_validate(run))

    async def _subdivide_and_retry(
        self,
        source: _Source,
        window: Window,
        *,
        state: _Progress,
        depth: int,
        started: float,
        cause: ExtractionOverflow,
    ) -> None:
        """Split an overflowing interval and extract its children in order.

        Two independent stopping conditions, both deterministic, either of which
        ends the descent by failing loudly rather than by guessing:

        - `MAX_SUBDIVISION_DEPTH`, an absolute bound on how far one interval may
          be pursued, so a pathological region costs a bounded number of calls;
        - `subdivide` returning `None`, meaning the interval is already at the
          smallest width worth splitting.

        Reaching either means a region of the source is denser than one bounded
        response can carry at the finest granularity this system will split to.
        That is reported as exactly what it is — an unread character range — and
        never papered over by truncating the item list, paginating the model, or
        inventing a second evidence item.
        """
        if depth >= MAX_SUBDIVISION_DEPTH:
            raise IrreducibleOverflow(
                window.start, window.end, f"subdivision depth limit {MAX_SUBDIVISION_DEPTH} reached"
            ) from cause

        children = subdivide(source.content, window.start, window.end)
        if children is None:
            raise IrreducibleOverflow(
                window.start,
                window.end,
                f"interval is at or below the {MIN_SUBDIVISIBLE_CHARS}-character minimum",
            ) from cause

        logger.info(
            "extraction.subdivided",
            extra={
                "evidence_id": str(source.evidence_id),
                "window_start": window.start,
                "window_end": window.end,
                "depth": depth,
                "child_count": len(children),
                "children": [(child.start, child.end) for child in children],
                "error_code": cause.error_code,
            },
        )
        for child in children:
            await self._extract_interval(
                source, child, state=state, depth=depth + 1, started=started
            )

    async def list_runs(
        self, evidence_id: uuid.UUID, *, limit: int = MAX_RUNS_LISTED
    ) -> ExtractionRunList:
        """Every attempt for one evidence item, newest first, failures included."""
        evidence = await self._evidence.get(evidence_id)
        if evidence is None:
            raise EvidenceNotFound(evidence_id)

        runs = await self._repository.list_runs_for_evidence(evidence_id, limit=limit)
        return ExtractionRunList(items=[ExtractionRunRead.model_validate(run) for run in runs])

    # -- internals ---------------------------------------------------------

    async def _record_attempt(
        self,
        source: _Source,
        *,
        window: Window,
        window_hash: str,
    ) -> ExtractionRun:
        identity = self._provider.identity
        run = self._repository.start_run(
            evidence_id=source.evidence_id,
            input_content_hash=window_hash,
            extractor_name=self._extractor_name,
            extractor_version=self._extractor_version,
            schema_version=self._schema_version,
            provider=identity.provider,
            model=identity.model,
            started_at=dt.datetime.now(dt.UTC),
            window_start=window.start,
            window_end=window.end,
        )
        try:
            await self._session.commit()
        except SQLAlchemyError as exc:
            await self._session.rollback()
            logger.exception("extraction.attempt.persist_failed")
            raise PersistenceError("Failed to record extraction attempt") from exc
        await self._session.refresh(run)
        return run

    async def _call_provider(
        self,
        source: _Source,
        *,
        window_text: str,
        window: Window,
    ) -> tuple[ExtractionOutput, ProviderResult]:
        """One provider call for one interval, validated against that interval.

        The model is sent the interval it owns and nothing else. No neighbouring
        text travels with it: supplying surrounding context was tried against
        the real corpus and measurably changed how items were *classified* —
        eleven of twenty-one flipping from `reported_claim` to `idea` — while
        every grounding quote still sat inside the target, so no quote check
        could have caught it. Since `idea` cannot enter an M4 relation, that
        silently deletes evidence downstream. The interval is the whole prompt.

        **Grounding is checked against the interval, not the whole document, and
        that is deliberately the stricter choice.** The interval is an exact
        slice of the content, so a quote found verbatim inside it is verbatim
        inside the source by construction — per-interval grounding *implies*
        source-level grounding. Checking against the whole document instead
        would additionally accept a quote the model reproduced from text it was
        never shown in this call, which is precisely the kind of unearned recall
        the grounding check exists to catch.

        The cost is named: a quote that straddles an interval boundary is
        rejected rather than trimmed. Boundaries fall on headings, so this is
        unlikely, and loud rejection is the correct failure for a system that
        refuses to normalise quoted content.

        **Truncation is checked before parsing, and that ordering matters.** A
        response cut off at the output ceiling is an incomplete answer for this
        interval whether or not the fragment that arrived happens to be valid
        JSON, so the stop reason decides, not the parse. Checking afterwards
        would let a response that truncated on a lucky boundary be stored as a
        complete reading of the interval.
        """
        result = await self._provider.extract(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(
                content=window_text,
                provenance=_describe_provenance(source, window=window),
            ),
            json_schema=wire_schema(),
        )
        if result.stop_reason is StopReason.MAX_OUTPUT:
            raise ExtractionOverflow(
                "Provider stopped at the output-token ceiling; the response is truncated"
            )
        output = parse_output(result.raw_output)
        verify_grounding(output, window_text)
        return output, result

    @staticmethod
    def _warn_on_identity_drift(run: ExtractionRun, result: ProviderResult) -> None:
        """Say so when the generator that answered is not the one that was asked.

        Not an error. A vendor resolving `claude-opus-5` to a dated snapshot is
        normal and the output is still valid, so failing the run would refuse
        good work. But it is exactly the condition under which "the same model"
        stops meaning one thing, and reuse keys on the requested side, so it must
        not pass unremarked. Every real run in the corpus to date reports back
        the string it was sent; this fires if that stops being true.
        """
        if run.provider == result.provider and run.model == result.model:
            return
        logger.warning(
            "extraction.provider.identity_drift",
            extra={
                "extraction_run_id": str(run.id),
                "requested_provider": run.provider,
                "requested_model": run.model,
                "responded_provider": result.provider,
                "responded_model": result.model,
            },
        )

    async def _persist_success(
        self, run: ExtractionRun, output: ExtractionOutput, provider_result: ProviderResult
    ) -> None:
        """Mark succeeded and insert every item in one transaction.

        `provider` and `model` are deliberately *not* written here. They were
        written at the attempt from the requested identity, and that is the value
        reuse compares against; overwriting them with the response's would make
        the row unfindable by the lookup that has to find it whenever a vendor
        resolves an alias to something else.

        A divergence is still worth knowing about, so it is logged rather than
        dropped — see `_warn_on_identity_drift`.
        """
        self._warn_on_identity_drift(run, provider_result)
        run.status = RunStatus.SUCCEEDED
        run.completed_at = dt.datetime.now(dt.UTC)
        run.input_tokens = provider_result.input_tokens
        run.output_tokens = provider_result.output_tokens

        for position, item in enumerate(output.items):
            self._repository.add_item(
                run=run,
                kind=item.kind,
                payload=item.data.model_dump(mode="json"),
                grounding_quote=item.grounding.quote,
                position=position,
            )

        try:
            await self._session.commit()
        except SQLAlchemyError as exc:
            # Nothing partial survives: the run stays `running` rather than
            # becoming a success with missing items.
            await self._session.rollback()
            logger.exception("extraction.success.persist_failed")
            raise PersistenceError("Failed to persist extraction results") from exc

    async def _mark_failed(self, run: ExtractionRun, exc: ExtractionError) -> None:
        """Record why the attempt failed. Never raises over the original error."""
        await self._session.rollback()
        try:
            run.status = RunStatus.FAILED
            run.completed_at = dt.datetime.now(dt.UTC)
            run.error_code = exc.error_code
            run.error_message = str(exc)[:MAX_ERROR_MESSAGE_LENGTH]
            await self._session.commit()
        except SQLAlchemyError:
            # The provider failure is the interesting error; losing the audit
            # row is worth a log, not a replaced exception.
            await self._session.rollback()
            logger.exception("extraction.failure.persist_failed")

    def _log_fields(
        self,
        *,
        run: ExtractionRun,
        reused: bool,
        duration_ms: float,
        **extra: object,
    ) -> dict[str, object]:
        """Operational fields only — never evidence text, never provider payloads."""
        fields: dict[str, object] = {
            "evidence_id": str(run.evidence_id),
            "extraction_run_id": str(run.id),
            "extractor_name": run.extractor_name,
            "extractor_version": run.extractor_version,
            "schema_version": run.schema_version,
            "provider": run.provider,
            "model": run.model,
            "status": run.status.value,
            "input_tokens": run.input_tokens,
            "output_tokens": run.output_tokens,
            "window_start": run.window_start,
            "window_end": run.window_end,
            "reused": reused,
            "duration_ms": duration_ms,
        }
        fields.update(extra)
        return fields


def _source_provenance(evidence: EvidenceItem) -> tuple[str, ...]:
    """The provenance lines describing the source itself.

    Read from the ORM object exactly once, while the session is known to be
    clean, and carried as plain strings thereafter — see `_Source`. Absent
    fields are simply omitted; writing "unknown" would invite the model to
    reason about the unknown.
    """
    parts: list[str] = []
    if evidence.source_type:
        parts.append(f"source_type: {evidence.source_type}")
    if evidence.source_label:
        parts.append(f"source_label: {evidence.source_label}")
    if evidence.source_url:
        parts.append(f"source_url: {evidence.source_url}")
    if evidence.captured_at:
        parts.append(f"captured_at: {evidence.captured_at.isoformat()}")
    return tuple(parts)


def _describe_provenance(source: _Source, *, window: Window) -> str:
    """A compact provenance summary for the prompt.

    Given to the model as interpretation context — who said this, when — but
    never as extractable content.

    The excerpt line appears **only when the interval is not the whole source**,
    decided by comparing the interval to the content length rather than by
    counting a plan. That is what keeps a document small enough to read in one
    pass producing a prompt byte-identical to the one it produced before any of
    this existed — so no stored run is invalidated and no extractor version bump
    is owed for evidence whose treatment did not change. When it does appear, it
    is honest about the partial view: a model told it is reading part of a
    conversation will not report the part as the whole.

    **The wording carries no ordinal and no total, deliberately.** It used to say
    "part N of M". Under subdivision M changes whenever any interval overflows,
    so the same interval text would be described differently between runs — and
    reuse is keyed on the hash of that text, not on the prompt, so the two runs
    would collide while having asked genuinely different questions. Saying only
    that this is an excerpt removes the moving part instead of adding it to the
    key, which is both smaller and the honest description: what the model needs
    to know is that it is reading a fragment, not where the fragment sits.
    """
    parts = list(source.provenance)
    if not _spans_whole(window, source.length):
        parts.append(
            "excerpt: an excerpt of this source, not the whole of it; "
            "extract only what this excerpt states"
        )
    return "\n".join(parts) if parts else "(no provenance supplied)"


def _spans_whole(window: Window, source_length: int) -> bool:
    """Whether this interval is the entire source rather than a part of it."""
    return window.start == 0 and window.end == source_length


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
