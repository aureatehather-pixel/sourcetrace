"""Persistence for analysis runs and derived relations.

Like every other repository here, this one never commits — transaction
boundaries belong to the service.

Two queries carry most of the weight:

- `list_relatable_items` assembles the entity's grounded material by walking the
  same chain the M3 profile walks (resolution → item → run → evidence), so M4
  can never see material the profile would not show.
- `list_current_relations` re-walks that chain for *both* sides of every
  relation, which is what makes an M3 correction propagate into M4 with M4
  writing nothing.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Sequence

from sqlalchemy import Row, and_, func, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from intelligence_engine.entities.models import EntityResolution
from intelligence_engine.evidence.models import EvidenceItem
from intelligence_engine.extraction.models import ExtractionItem, ExtractionRun
from intelligence_engine.intelligence.candidates import CandidatePair
from intelligence_engine.intelligence.models import IntelligenceRelation, RelationAnalysisRun
from intelligence_engine.intelligence.schemas import (
    MEANINGFUL_RELATION_TYPES,
    RELATABLE_ITEM_KINDS,
    RELATABLE_ROLES,
    AnalysisStatus,
    RelationDirection,
    RelationStatus,
    RelationType,
)

#: A pathological entity should not be able to turn one request into an
#: unbounded query.
#:
#: The comment here used to read "well above anything the cap can consume",
#: which conflated two different bounds. `relation_max_candidate_pairs` caps
#: *pairs*; this caps *items*, and 200 items is 19,900 pairs, so the pair cap
#: cannot protect the item load. Above 200 items this limit discards evidence,
#: and `count_relatable_items` exists so that discarding is reported rather than
#: silent — see `list_relatable_items` for the ordering it discards under.
MAX_RELATABLE_ITEMS = 200
MAX_RELATIONS_LISTED = 500

MaterialRow = Row[tuple[EntityResolution, ExtractionItem, ExtractionRun, EvidenceItem]]

RelationRow = Row[
    tuple[
        IntelligenceRelation,
        EntityResolution,
        ExtractionItem,
        ExtractionRun,
        EvidenceItem,
        EntityResolution,
        ExtractionItem,
        ExtractionRun,
        EvidenceItem,
    ]
]


class IntelligenceRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # -- grounded material -------------------------------------------------

    async def list_relatable_items(
        self, entity_id: uuid.UUID, *, limit: int = MAX_RELATABLE_ITEMS
    ) -> Sequence[MaterialRow]:
        """Every assertion currently attributed to this entity, with provenance.

        Active `claimed_by` and `about` resolutions only — `RELATABLE_ROLES`.
        Those are the two ways M3 connects an assertion to an entity: the entity
        *said* it, or the evidence says the observation is *about* it. Both are
        deliberate assertions recorded against a grounded item. `mention` is
        excluded because co-occurrence is not attribution — a claim does not
        belong to a person because they were named in the same note.

        The role and kind filters are kept together even though the composite
        foreign key already guarantees they agree. The redundancy is the point: a
        fourth role added later cannot silently widen what M4 consumes.

        Ordered by evidence arrival then item position, so candidate generation
        and therefore batching are reproducible.

        **The ordering and the limit interact, and the interaction loses the
        newest evidence.** Oldest-first plus `LIMIT 200` means an entity past 200
        relatable items is analysed on its 200 oldest and never sees the rest.
        Reproducibility is a real requirement and this ordering delivers it, so
        which 200 to keep is a policy question rather than an obvious bug, and it
        is deliberately not answered here. What *is* answered is that the
        discarding is now visible: `count_relatable_items` reports the true
        total, and the analysis telemetry carries total, loaded and truncated
        separately.
        """
        stmt = (
            select(EntityResolution, ExtractionItem, ExtractionRun, EvidenceItem)
            .join(ExtractionItem, ExtractionItem.id == EntityResolution.extraction_item_id)
            .join(ExtractionRun, ExtractionRun.id == ExtractionItem.extraction_run_id)
            .join(EvidenceItem, EvidenceItem.id == ExtractionRun.evidence_id)
            .where(
                EntityResolution.entity_id == entity_id,
                EntityResolution.is_active.is_(True),
                EntityResolution.role.in_(sorted(RELATABLE_ROLES)),
                ExtractionItem.kind.in_(sorted(RELATABLE_ITEM_KINDS)),
            )
            .order_by(
                EvidenceItem.created_at.asc(),
                EvidenceItem.id.asc(),
                ExtractionItem.position.asc(),
            )
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return result.all()

    async def count_relatable_items(self, entity_id: uuid.UUID) -> int:
        """How many relatable items this entity has, before any limit applies.

        The same filters as `list_relatable_items` and deliberately no `ORDER BY`
        and no `LIMIT`: this is the number the limit is measured against, so a
        limit appearing in it would make it report itself.

        Its whole purpose is the difference between this and what was loaded. A
        telemetry field derived from the truncated set cannot describe the
        truncation, which is how `theoretical_pairs` came to be computed over 200
        items and reported as though it were the entity's whole population.
        """
        stmt = (
            select(func.count())
            .select_from(EntityResolution)
            .join(ExtractionItem, ExtractionItem.id == EntityResolution.extraction_item_id)
            .where(
                EntityResolution.entity_id == entity_id,
                EntityResolution.is_active.is_(True),
                EntityResolution.role.in_(sorted(RELATABLE_ROLES)),
                ExtractionItem.kind.in_(sorted(RELATABLE_ITEM_KINDS)),
            )
        )
        return int((await self._session.execute(stmt)).scalar_one())

    # -- runs --------------------------------------------------------------

    def start_run(
        self,
        *,
        entity_id: uuid.UUID,
        classifier_name: str,
        classifier_version: str,
        schema_version: int,
        provider: str,
        model: str,
        started_at: dt.datetime,
        theoretical_pairs: int,
        candidate_pairs: int,
        reused_pairs: int,
    ) -> RelationAnalysisRun:
        """Stage a `running` analysis. The caller commits it before calling out.

        `provider` and `model` are the requested identity, recorded at the
        attempt for M2's reason: a failed analysis then names the generator that
        failed instead of leaving both columns null.
        """
        run = RelationAnalysisRun(
            id=uuid.uuid4(),
            entity_id=entity_id,
            classifier_name=classifier_name,
            classifier_version=classifier_version,
            schema_version=schema_version,
            status=AnalysisStatus.RUNNING,
            provider=provider,
            model=model,
            started_at=started_at,
            theoretical_pairs=theoretical_pairs,
            candidate_pairs=candidate_pairs,
            reused_pairs=reused_pairs,
        )
        self._session.add(run)
        return run

    async def get_run(self, run_id: uuid.UUID) -> RelationAnalysisRun | None:
        return await self._session.get(RelationAnalysisRun, run_id)

    async def list_runs_for_entity(
        self, entity_id: uuid.UUID, *, limit: int
    ) -> list[RelationAnalysisRun]:
        """All analyses of one entity, newest first — failures included."""
        stmt = (
            select(RelationAnalysisRun)
            .where(RelationAnalysisRun.entity_id == entity_id)
            .order_by(RelationAnalysisRun.created_at.desc(), RelationAnalysisRun.id.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    # -- relations ---------------------------------------------------------

    async def find_settled(
        self,
        pairs: Sequence[CandidatePair],
        *,
        classifier_name: str,
        classifier_version: str,
        schema_version: int,
        provider: str,
        model: str,
        fingerprints: dict[tuple[uuid.UUID, uuid.UUID], str],
    ) -> dict[tuple[uuid.UUID, uuid.UUID], IntelligenceRelation]:
        """Pairs already answered under this classifier version *by this model*.

        "Settled" covers **rejected** relations as well as current ones. A human
        who marked a relation wrong has answered the question for this classifier
        version; regenerating it would pay for the same wrong answer again. A
        version bump is what re-opens it.

        `input_fingerprint` is part of the match. It cannot fail given M2's
        immutability, and that is the point: if it ever did, this lookup misses,
        the insert hits `unique_settled_pair_per_classifier`, and the request
        fails loudly instead of serving intelligence derived from text that has
        since changed.

        **`provider` and `model` are part of it too**, and their absence was a
        defect with a specific consequence: classify a pair with one model, point
        `RELATION_MODEL` at another, and the second model is never asked — the
        first one's answer is served as though it were the second's. A comparison
        between two classifiers run that way measures one classifier twice.

        The unique constraint moved with this lookup, deliberately and in the
        same change. Had only the lookup widened, a model swap would miss here,
        insert, and collide on a constraint that still said one answer per pair
        per classifier version — turning "classify this with the other model"
        into an integrity error. Reuse identity and settled-answer identity are
        the same identity, and they are kept the same identity.
        """
        if not pairs:
            return {}

        keys = [
            (pair.left.item_id, pair.right.item_id, fingerprints[pair.key])
            for pair in pairs
            if pair.key in fingerprints
        ]
        if not keys:
            return {}

        stmt = select(IntelligenceRelation).where(
            IntelligenceRelation.classifier_name == classifier_name,
            IntelligenceRelation.classifier_version == classifier_version,
            IntelligenceRelation.schema_version == schema_version,
            IntelligenceRelation.provider == provider,
            IntelligenceRelation.model == model,
            tuple_(
                IntelligenceRelation.left_extraction_item_id,
                IntelligenceRelation.right_extraction_item_id,
                IntelligenceRelation.input_fingerprint,
            ).in_(keys),
        )
        result = await self._session.execute(stmt)
        return {
            (row.left_extraction_item_id, row.right_extraction_item_id): row
            for row in result.scalars().all()
        }

    def add_relation(
        self,
        *,
        analysis_run_id: uuid.UUID,
        entity_id: uuid.UUID,
        pair: CandidatePair,
        relation_type: RelationType,
        direction: RelationDirection,
        explanation: str | None,
        classifier_name: str,
        classifier_version: str,
        schema_version: int,
        provider: str,
        model: str,
        input_fingerprint: str,
    ) -> IntelligenceRelation:
        """Stage one derived relation. Not durable until the caller commits.

        `provider` and `model` are required rather than optional. They are part
        of what makes a settled answer this answer, so a relation that could not
        say which model produced it would be a relation the reuse lookup could
        not honestly match — and, since they are in the unique constraint, a null
        one would silently stop conflicting with anything.
        """
        relation = IntelligenceRelation(
            id=uuid.uuid4(),
            analysis_run_id=analysis_run_id,
            entity_id=entity_id,
            left_extraction_item_id=pair.left.item_id,
            right_extraction_item_id=pair.right.item_id,
            left_item_kind=pair.left.kind,
            right_item_kind=pair.right.kind,
            relation_type=relation_type,
            direction=direction,
            explanation=explanation,
            classifier_name=classifier_name,
            classifier_version=classifier_version,
            schema_version=schema_version,
            provider=provider,
            model=model,
            input_fingerprint=input_fingerprint,
            status=RelationStatus.CURRENT,
        )
        self._session.add(relation)
        return relation

    async def get_relation(self, relation_id: uuid.UUID) -> IntelligenceRelation | None:
        return await self._session.get(IntelligenceRelation, relation_id)

    def reject(self, relation: IntelligenceRelation, *, at: dt.datetime, note: str | None) -> None:
        """Mark a relation wrong in place.

        The assertion itself is not edited: relation type, direction, explanation,
        classifier version and creation time all stay exactly as written, so the
        row remains a faithful record of what the system asserted and when. Only
        whether it is served changes.
        """
        relation.status = RelationStatus.REJECTED
        relation.rejected_at = at
        if note is not None:
            relation.rejection_note = note

    async def list_current_relations(
        self,
        entity_id: uuid.UUID,
        *,
        classifier_name: str,
        classifier_version: str,
        schema_version: int,
        limit: int = MAX_RELATIONS_LISTED,
    ) -> Sequence[RelationRow]:
        """The entity's current derived relations, with both sides fully provenanced.

        Four filters, each doing distinct work:

        - **status is current** — a rejected relation is history, not intelligence;
        - **classifier version matches the running one** — output from a retired
          classifier stays in the table and stops being served, exactly as M2
          treats extraction runs from a superseded extractor version;
        - **relation type is meaningful** — `no_meaningful_relation` is persisted
          so it never has to be paid for twice, and is never shown;
        - **both sides still resolve to this entity** — joined through *active*
          resolutions in `RELATABLE_ROLES`. Detaching an item in M3 therefore
          withdraws every relation built on it, without M4 writing a row. A
          derived assertion must not outlive the connection it was derived from.
          This role set must match the one candidate loading uses: if this join
          were narrower, relations would be classified, persisted, and then never
          served and never reused — invisible output, re-paid on every run.
        """
        left_res = aliased(EntityResolution, name="left_resolution")
        left_item = aliased(ExtractionItem, name="left_item")
        left_run = aliased(ExtractionRun, name="left_run")
        left_evidence = aliased(EvidenceItem, name="left_evidence")
        right_res = aliased(EntityResolution, name="right_resolution")
        right_item = aliased(ExtractionItem, name="right_item")
        right_run = aliased(ExtractionRun, name="right_run")
        right_evidence = aliased(EvidenceItem, name="right_evidence")

        stmt = (
            select(
                IntelligenceRelation,
                left_res,
                left_item,
                left_run,
                left_evidence,
                right_res,
                right_item,
                right_run,
                right_evidence,
            )
            .join(left_item, left_item.id == IntelligenceRelation.left_extraction_item_id)
            .join(left_run, left_run.id == left_item.extraction_run_id)
            .join(left_evidence, left_evidence.id == left_run.evidence_id)
            .join(
                left_res,
                and_(
                    left_res.extraction_item_id == IntelligenceRelation.left_extraction_item_id,
                    left_res.entity_id == IntelligenceRelation.entity_id,
                    left_res.is_active.is_(True),
                    left_res.role.in_(sorted(RELATABLE_ROLES)),
                ),
            )
            .join(right_item, right_item.id == IntelligenceRelation.right_extraction_item_id)
            .join(right_run, right_run.id == right_item.extraction_run_id)
            .join(right_evidence, right_evidence.id == right_run.evidence_id)
            .join(
                right_res,
                and_(
                    right_res.extraction_item_id == IntelligenceRelation.right_extraction_item_id,
                    right_res.entity_id == IntelligenceRelation.entity_id,
                    right_res.is_active.is_(True),
                    right_res.role.in_(sorted(RELATABLE_ROLES)),
                ),
            )
            .where(
                IntelligenceRelation.entity_id == entity_id,
                IntelligenceRelation.status == RelationStatus.CURRENT,
                IntelligenceRelation.classifier_name == classifier_name,
                IntelligenceRelation.classifier_version == classifier_version,
                IntelligenceRelation.schema_version == schema_version,
                IntelligenceRelation.relation_type.in_(sorted(MEANINGFUL_RELATION_TYPES)),
            )
            .order_by(
                IntelligenceRelation.created_at.asc(),
                IntelligenceRelation.id.asc(),
            )
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return result.all()
