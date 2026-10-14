"""Intelligence application behaviour: analyse, review, read.

Owns the ordering that keeps a derived assertion honest and affordable:

    grounded material (active claimed_by resolutions only)
      -> deterministic candidate narrowing, bounded and capped
      -> settled-pair lookup       (no model call for a question already answered)
      -> record the attempt        (committed, so history survives a crash)
      -> bounded batches
      -> strict validation + batch identity binding
      -> one transaction per batch
      -> telemetry

Two ordering decisions differ from M2 on purpose.

**No run row when nothing needs classifying.** An analysis where every candidate
is already settled spends nothing and records nothing but a decision not to spend;
the response carries that. M2 reuse behaves the same way.

**Each batch is its own transaction.** M2 refuses partial acceptance because a
run is one indivisible extraction of one evidence item. Here, batches are
independent: pair 17 being answered badly says nothing about pairs 1-8, which are
individually valid and individually reusable. Discarding them would mean paying
for them twice. The run is marked `failed`, the caller gets a 502, and the work
that was sound stays.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import NamedTuple

from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from intelligence_engine.core.errors import PersistenceError
from intelligence_engine.entities.errors import EntityNotFound
from intelligence_engine.entities.models import Entity, EntityResolution
from intelligence_engine.entities.repository import EntityRepository
from intelligence_engine.entities.schemas import EntityRead
from intelligence_engine.entities.service import EntityService
from intelligence_engine.evidence.models import EvidenceItem
from intelligence_engine.extraction.models import ExtractionItem
from intelligence_engine.extraction.provider import ExtractionProvider, ProviderResult
from intelligence_engine.extraction.schemas import (
    ItemKind,
    ObservationData,
    ReportedClaimData,
)
from intelligence_engine.intelligence import versioning
from intelligence_engine.intelligence.candidates import (
    CandidatePair,
    RelatableItem,
    batched,
    generate_candidates,
)
from intelligence_engine.intelligence.errors import (
    IntelligenceAssemblyError,
    IntelligenceError,
    RelationAlreadyRejected,
    RelationNotFound,
)
from intelligence_engine.intelligence.models import (
    MAX_ERROR_MESSAGE_LENGTH,
    IntelligenceRelation,
    RelationAnalysisRun,
)
from intelligence_engine.intelligence.prompt import (
    SYSTEM_PROMPT,
    build_user_prompt,
    pair_fingerprint,
    render_pair,
)
from intelligence_engine.intelligence.repository import (
    MAX_RELATABLE_ITEMS,
    IntelligenceRepository,
)
from intelligence_engine.intelligence.schemas import (
    MEANINGFUL_RELATION_TYPES,
    AnalysisResponse,
    AnalysisStatus,
    AnalysisTelemetry,
    DerivedRelation,
    EntityIntelligence,
    IntelligenceRelationRead,
    RelationDirection,
    RelationReject,
    RelationSide,
    RelationStatus,
    RelationType,
    wire_schema,
)
from intelligence_engine.intelligence.validation import bind_results, parse_output

logger = logging.getLogger(__name__)

DEFAULT_MAX_CANDIDATE_PAIRS = 60
DEFAULT_BATCH_SIZE = 8
MAX_ANALYSIS_RUNS_LISTED = 100

_READING = {
    RelationType.REINFORCES: "left and right reinforce each other",
    RelationType.CONTRADICTS: "left and right contradict each other",
}


class IntelligenceService:
    def __init__(
        self,
        session: AsyncSession,
        provider: ExtractionProvider,
        *,
        classifier_name: str = versioning.CLASSIFIER_NAME,
        classifier_version: str = versioning.CLASSIFIER_VERSION,
        schema_version: int = versioning.SCHEMA_VERSION,
        max_candidate_pairs: int = DEFAULT_MAX_CANDIDATE_PAIRS,
        max_relatable_items: int = MAX_RELATABLE_ITEMS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._session = session
        self._provider = provider
        self._repository = IntelligenceRepository(session)
        self._entities = EntityRepository(session)
        # Injectable so tests can prove a version bump forces reclassification
        # and that the caps actually bind.
        self._classifier_name = classifier_name
        self._classifier_version = classifier_version
        self._schema_version = schema_version
        self._max_candidate_pairs = max_candidate_pairs
        # Injectable for the same reason as the version constants: proving that
        # truncation is *reported* needs a run that truncates, and the real
        # bound is 200 items. A test that had to build 201 resolutions to reach
        # it would be slow enough not to be written.
        self._max_relatable_items = max_relatable_items
        self._batch_size = batch_size

    # -- analysis ----------------------------------------------------------

    async def analyze(self, entity_id: uuid.UUID) -> AnalysisResponse:
        """Derive relations among one entity's grounded claims."""
        started = time.perf_counter()
        entity = await self._require_entity(entity_id)

        items, relatable_total = await self._load_relatable_items(entity_id)
        selection = generate_candidates(items, max_pairs=self._max_candidate_pairs)

        identity = self._provider.identity
        fingerprints = {pair.key: pair_fingerprint(pair) for pair in selection.pairs}
        settled = await self._repository.find_settled(
            selection.pairs,
            classifier_name=self._classifier_name,
            classifier_version=self._classifier_version,
            schema_version=self._schema_version,
            provider=identity.provider,
            model=identity.model,
            fingerprints=fingerprints,
        )
        pending = [pair for pair in selection.pairs if pair.key not in settled]

        logger.info(
            "intelligence.candidates.selected",
            extra={
                "entity_id": str(entity_id),
                "relatable_items_total": relatable_total,
                "relatable_items_loaded": len(items),
                "relatable_items_truncated": relatable_total > len(items),
                "theoretical_pairs": selection.theoretical_pairs,
                "filtered_pairs": selection.filtered_pairs,
                "candidate_pairs": len(selection.pairs),
                "capped": selection.capped,
                "reused_pairs": len(settled),
                "pending_pairs": len(pending),
            },
        )

        if not pending:
            # Nothing to ask, so nothing to record but the decision not to spend.
            return self._response(
                entity_id=entity_id,
                run=None,
                relatable_total=relatable_total,
                relatable_loaded=len(items),
                selection_theoretical=selection.theoretical_pairs,
                candidate_pairs=len(selection.pairs),
                reused=len(settled),
                classified=0,
                batch_count=0,
                call_count=0,
                input_tokens=None,
                output_tokens=None,
                relations=list(settled.values()),
                new_relations=[],
            )

        run = await self._record_attempt(
            entity_id=entity_id,
            theoretical_pairs=selection.theoretical_pairs,
            candidate_pairs=len(selection.pairs),
            reused_pairs=len(settled),
        )

        batches = batched(pending, size=self._batch_size)
        created: list[IntelligenceRelation] = []
        totals = _Totals()

        for index, batch in enumerate(batches):
            try:
                relations, result = await self._classify_batch(
                    run=run, entity=entity, batch=batch, fingerprints=fingerprints
                )
            except IntelligenceError as exc:
                await self._mark_failed(run, exc, totals)
                logger.warning(
                    "intelligence.analysis.failed",
                    extra=self._log_fields(
                        run=run,
                        duration_ms=_elapsed_ms(started),
                        error_code=exc.error_code,
                        failed_batch=index,
                        persisted_relations=len(created),
                    ),
                )
                raise

            created.extend(relations)
            totals.record(result, batch_size=len(batch))

        await self._mark_succeeded(run, totals)

        logger.info(
            "intelligence.analysis.succeeded",
            extra=self._log_fields(
                run=run,
                duration_ms=_elapsed_ms(started),
                new_relations=len(created),
                new_meaningful_relations=_count_meaningful(created),
            ),
        )

        return self._response(
            entity_id=entity_id,
            run=run,
            relatable_total=relatable_total,
            relatable_loaded=len(items),
            selection_theoretical=selection.theoretical_pairs,
            candidate_pairs=len(selection.pairs),
            reused=len(settled),
            classified=totals.classified,
            batch_count=totals.batch_count,
            call_count=totals.call_count,
            input_tokens=totals.input_tokens,
            output_tokens=totals.output_tokens,
            relations=[*settled.values(), *created],
            new_relations=created,
        )

    # -- review ------------------------------------------------------------

    async def reject_relation(
        self, relation_id: uuid.UUID, payload: RelationReject
    ) -> IntelligenceRelationRead:
        """Mark a derived relation wrong.

        It stops being served immediately, stays in the table as a record of what
        was asserted, and continues to satisfy the reuse lookup — so the next
        analysis does not pay to regenerate the same wrong answer. Bumping the
        classifier version is what re-opens the question.
        """
        relation = await self._repository.get_relation(relation_id)
        if relation is None:
            raise RelationNotFound(relation_id)
        if relation.status is not RelationStatus.CURRENT:
            raise RelationAlreadyRejected(
                f"Intelligence relation {relation_id} was already rejected at "
                f"{relation.rejected_at.isoformat() if relation.rejected_at else 'unknown'}"
            )

        self._repository.reject(relation, at=dt.datetime.now(dt.UTC), note=payload.note)
        await self._commit("intelligence.relation.reject_failed")
        await self._session.refresh(relation)

        logger.info(
            "intelligence.relation.rejected",
            extra={
                "relation_id": str(relation.id),
                "entity_id": str(relation.entity_id),
                "relation_type": relation.relation_type.value,
                "classifier_version": relation.classifier_version,
            },
        )
        return IntelligenceRelationRead.model_validate(relation)

    # -- read model --------------------------------------------------------

    async def get_intelligence(self, entity_id: uuid.UUID) -> EntityIntelligence:
        """Assemble grounded material and derived relations, at request time.

        Nothing here is stored. `grounded_claims` comes from the M3 profile's own
        read model rather than a second query, so the grounded half of this view
        cannot disagree with `/profile`.
        """
        entity = await self._require_entity(entity_id)
        profile = await EntityService(self._session).get_profile(entity_id)

        rows = await self._repository.list_current_relations(
            entity_id,
            classifier_name=self._classifier_name,
            classifier_version=self._classifier_version,
            schema_version=self._schema_version,
        )

        derived: list[DerivedRelation] = []
        for (
            relation,
            left_res,
            left_item,
            _left_run,
            left_evidence,
            right_res,
            right_item,
            _right_run,
            right_evidence,
        ) in rows:
            derived.append(
                DerivedRelation(
                    relation=IntelligenceRelationRead.model_validate(relation),
                    left=_side(left_res, left_item, left_evidence),
                    right=_side(right_res, right_item, right_evidence),
                    reading=_reading(relation.relation_type, relation.direction),
                )
            )

        return EntityIntelligence(
            entity=EntityRead.model_validate(entity),
            grounded_claims=profile.attributed_claims,
            derived_relations=derived,
        )

    async def list_analysis_runs(
        self, entity_id: uuid.UUID, *, limit: int = MAX_ANALYSIS_RUNS_LISTED
    ) -> list[RelationAnalysisRun]:
        await self._require_entity(entity_id)
        return await self._repository.list_runs_for_entity(entity_id, limit=limit)

    # -- internals ---------------------------------------------------------

    async def _load_relatable_items(self, entity_id: uuid.UUID) -> tuple[list[RelatableItem], int]:
        """The entity's relatable items, and how many it actually has.

        Two queries rather than one, and the second is the point. The loader is
        bounded by `MAX_RELATABLE_ITEMS`, so its result cannot report its own
        truncation; counting separately is the only way the caller can tell a
        complete analysis from one that silently dropped evidence.

        The count is taken after the load, so a concurrent write can make it
        larger than the loaded set by more than the limit explains. That reads as
        "more evidence exists than was analysed", which is true either way.
        """
        rows = await self._repository.list_relatable_items(
            entity_id, limit=self._max_relatable_items
        )
        total = await self._repository.count_relatable_items(entity_id)
        items = [
            _to_relatable(resolution, item, evidence) for resolution, item, _run, evidence in rows
        ]
        return items, total

    async def _record_attempt(
        self,
        *,
        entity_id: uuid.UUID,
        theoretical_pairs: int,
        candidate_pairs: int,
        reused_pairs: int,
    ) -> RelationAnalysisRun:
        identity = self._provider.identity
        run = self._repository.start_run(
            entity_id=entity_id,
            classifier_name=self._classifier_name,
            classifier_version=self._classifier_version,
            schema_version=self._schema_version,
            provider=identity.provider,
            model=identity.model,
            started_at=dt.datetime.now(dt.UTC),
            theoretical_pairs=theoretical_pairs,
            candidate_pairs=candidate_pairs,
            reused_pairs=reused_pairs,
        )
        try:
            await self._session.commit()
        except SQLAlchemyError as exc:
            await self._session.rollback()
            logger.exception("intelligence.attempt.persist_failed")
            raise PersistenceError("Failed to record relation analysis attempt") from exc
        await self._session.refresh(run)
        return run

    async def _classify_batch(
        self,
        *,
        run: RelationAnalysisRun,
        entity: Entity,
        batch: Sequence[CandidatePair],
        fingerprints: Mapping[tuple[uuid.UUID, uuid.UUID], str],
    ) -> tuple[list[IntelligenceRelation], ProviderResult]:
        """One provider call, validated and persisted as one transaction."""
        identity = self._provider.identity
        indexed = {f"pair_{position + 1:02d}": pair for position, pair in enumerate(batch)}
        rendered = [render_pair(pair_id, pair) for pair_id, pair in indexed.items()]

        result = await self._provider.extract(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(
                entity_label=f"{entity.canonical_name} ({entity.entity_type.value})",
                rendered_pairs=rendered,
            ),
            json_schema=wire_schema(),
        )
        bound = bind_results(parse_output(result.raw_output), indexed)

        relations = [
            self._repository.add_relation(
                analysis_run_id=run.id,
                entity_id=run.entity_id,
                pair=pair,
                relation_type=classified.relation_type,
                direction=direction,
                explanation=classified.explanation,
                classifier_name=self._classifier_name,
                classifier_version=self._classifier_version,
                schema_version=self._schema_version,
                # The requested identity, matching what `find_settled` looks
                # for. `result.provider` / `result.model` say who answered,
                # which is not necessarily what the next lookup will ask for.
                provider=identity.provider,
                model=identity.model,
                input_fingerprint=fingerprints[pair.key],
            )
            for pair, classified, direction in bound
        ]

        try:
            await self._session.commit()
        except SQLAlchemyError as exc:
            await self._session.rollback()
            logger.exception("intelligence.batch.persist_failed")
            raise PersistenceError("Failed to persist derived relations") from exc

        for relation in relations:
            await self._session.refresh(relation)
        return relations, result

    async def _mark_succeeded(self, run: RelationAnalysisRun, totals: _Totals) -> None:
        run.status = AnalysisStatus.SUCCEEDED
        run.completed_at = dt.datetime.now(dt.UTC)
        totals.apply(run)
        await self._commit("intelligence.analysis.persist_failed")

    async def _mark_failed(
        self, run: RelationAnalysisRun, exc: IntelligenceError, totals: _Totals
    ) -> None:
        """Record why the analysis failed. Never raises over the original error."""
        await self._session.rollback()
        try:
            run.status = AnalysisStatus.FAILED
            run.completed_at = dt.datetime.now(dt.UTC)
            run.error_code = exc.error_code
            run.error_message = str(exc)[:MAX_ERROR_MESSAGE_LENGTH]
            totals.apply(run)
            await self._session.commit()
        except SQLAlchemyError:
            await self._session.rollback()
            logger.exception("intelligence.failure.persist_failed")

    def _response(
        self,
        *,
        entity_id: uuid.UUID,
        run: RelationAnalysisRun | None,
        relatable_total: int,
        relatable_loaded: int,
        selection_theoretical: int,
        candidate_pairs: int,
        reused: int,
        classified: int,
        batch_count: int,
        call_count: int,
        input_tokens: int | None,
        output_tokens: int | None,
        relations: Sequence[IntelligenceRelation],
        new_relations: Sequence[IntelligenceRelation],
    ) -> AnalysisResponse:
        ordered = sorted(relations, key=lambda row: (row.created_at, row.id))
        return AnalysisResponse(
            performed_calls=call_count > 0,
            analysis_run_id=run.id if run is not None else None,
            entity_id=entity_id,
            classifier_name=self._classifier_name,
            classifier_version=self._classifier_version,
            schema_version=self._schema_version,
            telemetry=AnalysisTelemetry(
                relatable_items_total=relatable_total,
                relatable_items_loaded=relatable_loaded,
                relatable_items_truncated=relatable_total > relatable_loaded,
                theoretical_pairs=selection_theoretical,
                candidate_pairs=candidate_pairs,
                reused_pairs=reused,
                classified_pairs=classified,
                batch_count=batch_count,
                external_call_count=call_count,
                new_relations=len(new_relations),
                new_meaningful_relations=_count_meaningful(new_relations),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            relations=[IntelligenceRelationRead.model_validate(row) for row in ordered],
        )

    async def _require_entity(self, entity_id: uuid.UUID) -> Entity:
        entity = await self._entities.get(entity_id)
        if entity is None:
            raise EntityNotFound(entity_id)
        return entity

    async def _commit(self, event: str) -> None:
        try:
            await self._session.commit()
        except SQLAlchemyError as exc:
            await self._session.rollback()
            logger.exception(event)
            raise PersistenceError("Failed to persist intelligence state") from exc

    def _log_fields(
        self, *, run: RelationAnalysisRun, duration_ms: float, **extra: object
    ) -> dict[str, object]:
        """Operational fields only — never claim text, never provider payloads."""
        fields: dict[str, object] = {
            "entity_id": str(run.entity_id),
            "analysis_run_id": str(run.id),
            "classifier_name": run.classifier_name,
            "classifier_version": run.classifier_version,
            "schema_version": run.schema_version,
            "provider": run.provider,
            "model": run.model,
            "status": run.status.value,
            "classified_pairs": run.classified_pairs,
            "batch_count": run.batch_count,
            "external_call_count": run.external_call_count,
            "input_tokens": run.input_tokens,
            "output_tokens": run.output_tokens,
            "duration_ms": duration_ms,
        }
        fields.update(extra)
        return fields


class _Totals:
    """Running cost and volume counters for one analysis.

    Kept outside the ORM object because a failed batch rolls the session back,
    which would discard counters written straight onto the run row.
    """

    def __init__(self) -> None:
        self.classified = 0
        self.batch_count = 0
        self.call_count = 0
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None

    def record(self, result: ProviderResult, *, batch_size: int) -> None:
        """Counters only. The run's `provider`/`model` are not touched here.

        They were written at `start_run` from the requested identity, which is
        what `find_settled` compares against; writing the response's values over
        them on completion would leave the row describing one generator and
        matched under another.
        """
        self.classified += batch_size
        self.batch_count += 1
        self.call_count += 1
        self.input_tokens = _add(self.input_tokens, result.input_tokens)
        self.output_tokens = _add(self.output_tokens, result.output_tokens)

    def apply(self, run: RelationAnalysisRun) -> None:
        run.classified_pairs = self.classified
        run.batch_count = self.batch_count
        run.external_call_count = self.call_count
        run.input_tokens = self.input_tokens
        run.output_tokens = self.output_tokens


def _add(total: int | None, value: int | None) -> int | None:
    """Sum token counts, keeping `None` to mean "the provider did not report"."""
    if value is None:
        return total
    return value if total is None else total + value


def _count_meaningful(relations: Sequence[IntelligenceRelation]) -> int:
    return sum(1 for row in relations if row.relation_type in MEANINGFUL_RELATION_TYPES)


class _Assertion(NamedTuple):
    """One relatable item's payload, read according to its own kind."""

    text: str
    stance: str | None
    claimed_by: str | None
    subject: str | None


def _read_assertion(item: ExtractionItem) -> _Assertion:
    """Read a relatable item's payload through the M2 model for its kind.

    Dispatch on kind rather than assuming `reported_claim`: an `observation`
    reaching here is now normal, and validating it as a claim would either fail
    or - worse - succeed against a payload that means something else.

    The payload is re-validated rather than indexed directly, so JSONB written
    under an older schema version fails loudly here instead of yielding a
    silently wrong statement.

    `claimed_by` and `subject` are never both set. That is the epistemic
    invariant this function exists to preserve: a claim has a speaker, an
    observation has a subject, and neither becomes the other in transit.
    """
    if item.kind is ItemKind.REPORTED_CLAIM:
        try:
            claim = ReportedClaimData.model_validate(item.payload)
        except ValidationError as exc:
            raise IntelligenceAssemblyError(
                f"Extraction item {item.id} payload is not a readable reported_claim"
            ) from exc
        return _Assertion(
            text=claim.claim,
            stance=claim.stance,
            claimed_by=claim.claimed_by,
            subject=None,
        )

    if item.kind is ItemKind.OBSERVATION:
        try:
            observation = ObservationData.model_validate(item.payload)
        except ValidationError as exc:
            raise IntelligenceAssemblyError(
                f"Extraction item {item.id} payload is not a readable observation"
            ) from exc
        # No stance: an observation is not offered as advice or as a position,
        # so inventing one would tell the classifier something the evidence
        # never said.
        return _Assertion(
            text=observation.observation,
            stance=None,
            claimed_by=None,
            subject=observation.subject,
        )

    # Unreachable through the repository, which filters to RELATABLE_ITEM_KINDS.
    # Loud rather than silent: reaching here means that filter and this dispatch
    # have drifted apart.
    raise IntelligenceAssemblyError(
        f"Extraction item {item.id} has kind {item.kind.value}, which cannot stand in a relation"
    )


def _to_relatable(
    resolution: EntityResolution, item: ExtractionItem, evidence: EvidenceItem
) -> RelatableItem:
    """Flatten one grounded row into the shape candidate generation works on."""
    assertion = _read_assertion(item)

    return RelatableItem(
        item_id=item.id,
        kind=item.kind,
        role=resolution.role,
        text=assertion.text,
        stance=assertion.stance,
        claimed_by=assertion.claimed_by,
        subject=assertion.subject,
        grounding_quote=item.grounding_quote,
        extraction_run_id=item.extraction_run_id,
        evidence_id=evidence.id,
        resolution_id=resolution.id,
        source_label=evidence.source_label,
        captured_at=evidence.captured_at,
        evidence_created_at=evidence.created_at,
        position=item.position,
    )


def _side(
    resolution: EntityResolution, item: ExtractionItem, evidence: EvidenceItem
) -> RelationSide:
    assertion = _read_assertion(item)

    return RelationSide(
        extraction_item_id=item.id,
        extraction_run_id=item.extraction_run_id,
        evidence_id=evidence.id,
        resolution_id=resolution.id,
        item_kind=item.kind,
        text=assertion.text,
        stance=assertion.stance,
        claimed_by=assertion.claimed_by,
        subject=assertion.subject,
        grounding_quote=item.grounding_quote,
        source_label=evidence.source_label,
        captured_at=evidence.captured_at,
    )


def _reading(relation_type: RelationType, direction: RelationDirection) -> str:
    """A plain-language rendering of type + direction. Carries no new information."""
    if relation_type is RelationType.QUALIFIES:
        if direction is RelationDirection.LEFT_TO_RIGHT:
            return "left qualifies right"
        return "right qualifies left"
    return _READING.get(relation_type, relation_type.value)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
