"""Run the relation classifier against the hand-labelled corpus.

Deliberately a script rather than a pytest test. It costs money and needs
credentials, so it must never run as part of `uv run pytest`; what pytest does
check is that the corpus itself stays well-formed and keeps covering every trap
category (`tests/unit/test_gold_set.py`).

It exercises the **real** prompt, the real wire schema, the real batching and the
real validation path. Anything it measures is therefore a property of the
shipping classifier, not of an evaluation-only reimplementation of it.

Usage:

    uv run python -m evaluation.run_classifier_eval --model claude-haiku-4-5
    uv run python -m evaluation.run_classifier_eval --model claude-opus-5 --batch-size 8

Reports error categories first and aggregate agreement last, because the
aggregate is the least informative number on the page.

**It is a gate, and it exits non-zero.** It did not used to. It printed a report
and returned success whatever the numbers said, while being described as a gate —
so the only thing it mechanically proved was that a model had been reachable.

What it now rejects is stated entirely in terms of a baseline computed from the
corpus, never a threshold chosen here: an always-`no_meaningful_relation`
classifier is scored alongside the real one, and the real one must be no worse in
every failure category and strictly better on the pairs where declining is not an
acceptable answer. The abstention classifier scores 25/37 on aggregate agreement
and zero in every failure category, so aggregate agreement could never have been
the measure. See `gold_set.gate_failures` for why each condition is derived
rather than picked.

`--no-gate` prints the report and exits 0. It is for looking, not for CI.
"""

from __future__ import annotations

import argparse
import datetime as dt
import uuid
from collections.abc import Sequence

from evaluation.gold_set import (
    GoldPair,
    GoldStatement,
    Outcome,
    Report,
    abstention_baseline,
    gate_failures,
    has_causal_language,
    load_gold_set,
    meaningful_hits,
    requires_meaningful,
    score,
)
from intelligence_engine.core.config import get_settings
from intelligence_engine.core.platform import run_async
from intelligence_engine.entities.schemas import ResolutionRole
from intelligence_engine.extraction.anthropic_provider import AnthropicExtractionProvider
from intelligence_engine.extraction.schemas import ItemKind
from intelligence_engine.intelligence.candidates import CandidatePair, RelatableItem
from intelligence_engine.intelligence.errors import IntelligenceError
from intelligence_engine.intelligence.prompt import (
    SYSTEM_PROMPT,
    build_user_prompt,
    render_pair,
)
from intelligence_engine.intelligence.schemas import wire_schema
from intelligence_engine.intelligence.validation import bind_results, parse_output

_VERDICT_ORDER = (
    ("false_contradiction", "FALSE CONTRADICTIONS  (worst routine failure)"),
    ("false_relation", "false relations       (asserted where declining was right)"),
    ("wrong_meaningful", "wrong meaningful      (a relation, but the wrong one)"),
    ("wrong_direction", "wrong direction       (right label, backwards)"),
    ("missed_relation", "missed relations      (cheapest failure; acceptable in V1)"),
    ("malformed", "malformed             (batch failed validation)"),
    ("correct", "correct"),
)


def _to_item(statement: GoldStatement) -> RelatableItem:
    """A gold statement in the shape the real classifier consumes.

    `source_label` carries the corpus's `context` string so the rendered block is
    byte-identical in shape to a production one — the evaluation must not measure
    a prompt the service never sends.
    """
    kind = ItemKind(statement.kind)
    return RelatableItem(
        item_id=uuid.uuid4(),
        kind=kind,
        role=(ResolutionRole.ABOUT if kind is ItemKind.OBSERVATION else ResolutionRole.CLAIMED_BY),
        text=statement.statement,
        stance=statement.stance,
        claimed_by=statement.claimed_by,
        subject=statement.subject,
        grounding_quote=statement.quote,
        extraction_run_id=uuid.uuid4(),
        evidence_id=uuid.uuid4(),
        resolution_id=uuid.uuid4(),
        source_label=statement.context,
        captured_at=None,
        evidence_created_at=dt.datetime.now(dt.UTC),
        position=0,
    )


def _to_candidate(pair: GoldPair) -> CandidatePair:
    """Left is always A and right always B, so `expected_direction` stays readable.

    Canonical ordering by uuid is a persistence concern; the corpus is testing
    the classifier, which only ever sees A and B.
    """
    return CandidatePair(
        left=_to_item(pair.a),
        right=_to_item(pair.b),
        lexical_overlap=0.0,
        stance_complementary=False,
    )


async def _run(model: str, batch_size: int) -> tuple[Report, int, int, int]:
    settings = get_settings()
    if settings.anthropic_api_key is None:
        raise SystemExit("ANTHROPIC_API_KEY is not configured; this evaluation needs a real model")

    provider = AnthropicExtractionProvider(
        api_key=settings.anthropic_api_key.get_secret_value(),
        model=model,
        timeout_seconds=settings.relation_timeout_seconds,
        max_retries=settings.relation_max_retries,
        max_tokens=settings.relation_max_output_tokens,
    )

    gold = load_gold_set()
    report = Report()
    input_tokens = output_tokens = calls = 0

    # Grouped by entity label, because the label is part of what the classifier
    # is told and pairs analysed under different entities cannot share a call.
    grouped: dict[str, list[GoldPair]] = {}
    for pair in gold:
        grouped.setdefault(pair.entity_label, []).append(pair)
    chunks = [
        (label, pairs[start : start + batch_size])
        for label, pairs in grouped.items()
        for start in range(0, len(pairs), batch_size)
    ]

    for entity_label, chunk in chunks:
        indexed = {f"pair_{index + 1:02d}": _to_candidate(pair) for index, pair in enumerate(chunk)}
        by_id = dict(zip(indexed.keys(), chunk, strict=True))

        result = await provider.extract(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(
                entity_label=entity_label,
                rendered_pairs=[render_pair(pid, cand) for pid, cand in indexed.items()],
            ),
            json_schema=wire_schema(),
        )
        calls += 1
        input_tokens += result.input_tokens or 0
        output_tokens += result.output_tokens or 0

        try:
            bound = bind_results(parse_output(result.raw_output), indexed)
        except IntelligenceError as exc:
            print(f"  batch for {entity_label!r}: REJECTED - {exc}")  # noqa: T201
            for gold_pair in chunk:
                report.outcomes.append(
                    Outcome(
                        pair=gold_pair,
                        predicted=None,
                        direction=None,
                        explanation=None,
                        verdict="malformed",
                    )
                )
            continue

        # `bind_results` returns pairs in supplied order, which is `chunk` order.
        for (_candidate, classified, _direction), pair_id in zip(bound, indexed, strict=True):
            gold_pair = by_id[pair_id]
            verdict = score(gold_pair, classified.relation_type, classified.direction)
            report.outcomes.append(
                Outcome(
                    pair=gold_pair,
                    predicted=classified.relation_type,
                    direction=classified.direction,
                    explanation=classified.explanation,
                    verdict=verdict,
                    causal_language=has_causal_language(classified.explanation),
                )
            )

    return report, input_tokens, output_tokens, calls


def _print_report(
    report: Report,
    baseline: Report,
    model: str,
    batch_size: int,
    tokens: tuple[int, int, int],
) -> None:
    """Every count beside what a classifier that answers nothing would score.

    The baseline column is not decoration. Read on its own, "correct: 29/37"
    sounds like a result; read beside "25/37 for a `return` statement" it is a
    much smaller one, and the reader should not have to know that separately.
    """
    input_tokens, output_tokens, calls = tokens
    print(f"\n{'=' * 72}")  # noqa: T201
    print(f"model={model}  batch_size={batch_size}  examples={report.total}")  # noqa: T201
    print(  # noqa: T201
        f"calls={calls}  input_tokens={input_tokens}  output_tokens={output_tokens}"
    )
    print("=" * 72)  # noqa: T201
    print(f"{'':<58} {'this':>5} {'abstain':>8}")  # noqa: T201

    for verdict, label in _VERDICT_ORDER:
        print(  # noqa: T201
            f"{label:<58} {report.count(verdict):>5} {baseline.count(verdict):>8}"
        )

    causal = report.causal_explanations
    print(f"{'explanations containing causal language':<58} {len(causal):>5} {0:>8}")  # noqa: T201

    print(  # noqa: T201
        f"\n{'agreement with acceptable labels':<58} "
        f"{report.count('correct'):>5} {baseline.count('correct'):>8}"
    )
    required = sum(1 for outcome in report.outcomes if requires_meaningful(outcome.pair))
    print(  # noqa: T201
        f"{'relations found where declining was NOT acceptable':<58} "
        f"{meaningful_hits(report):>5} {meaningful_hits(baseline):>8}   of {required}"
    )
    print(  # noqa: T201
        "\n'abstain' is always-no_meaningful_relation over this same corpus, scored\n"
        "through the same `score` function. It reads nothing and costs nothing.\n"
        "Aggregate agreement is therefore not a measure of anything; the row above\n"
        "it is the only one abstention cannot reach.\n"
        "(n is small by design - this falsifies, it does not measure)"
    )

    _print_failures(report)
    _print_all(report)


def _print_failures(report: Report) -> None:
    failures = [outcome for outcome in report.outcomes if outcome.verdict != "correct"]
    if not failures:
        print("\nNo failures.")  # noqa: T201
        return
    print(f"\n{'-' * 72}\nFAILURES\n{'-' * 72}")  # noqa: T201
    for outcome in failures:
        predicted = outcome.predicted.value if outcome.predicted else "<none>"
        print(  # noqa: T201
            f"\n{outcome.pair.id} [{outcome.pair.category}] -> {outcome.verdict}\n"
            f"  expected  : {outcome.pair.expected.value} "
            f"(acceptable: {sorted(item.value for item in outcome.pair.acceptable)})\n"
            f"  predicted : {predicted} {outcome.direction or ''}\n"
            f"  A         : {outcome.pair.a.statement}\n"
            f"  B         : {outcome.pair.b.statement}\n"
            f"  said      : {outcome.explanation or '-'}"
        )


def _print_all(report: Report) -> None:
    print(f"\n{'-' * 72}\nALL RESULTS\n{'-' * 72}")  # noqa: T201
    for outcome in report.outcomes:
        predicted = outcome.predicted.value if outcome.predicted else "<none>"
        flag = "" if outcome.verdict == "correct" else f"  <<< {outcome.verdict}"
        print(f"{outcome.pair.id}  {predicted:<24}{flag}")  # noqa: T201
        if outcome.explanation:
            print(f"          {outcome.explanation}")  # noqa: T201


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Defaults to RELATION_MODEL from settings.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Print the report and exit 0 regardless of the result. For looking, not for CI.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    model = args.model or settings.relation_model
    batch_size = args.batch_size or settings.relation_batch_size

    gold = load_gold_set()
    baseline = abstention_baseline(gold)

    report, input_tokens, output_tokens, calls = run_async(_run(model, batch_size))
    _print_report(report, baseline, model, batch_size, (input_tokens, output_tokens, calls))

    failures = gate_failures(report, baseline)
    print(f"\n{'-' * 72}")  # noqa: T201
    if failures:
        print(f"REJECTED: {model} is not usable as the relation classifier.")  # noqa: T201
        for failure in failures:
            print(f"  - {failure}")  # noqa: T201
    else:
        print(  # noqa: T201
            f"ACCEPTED: {model} is no worse than declining in every failure category\n"
            "and strictly better where declining was not an acceptable answer.\n"
            "That is all this proves. It is not a quality measurement: 37 pairs,\n"
            "hand-labelled by the people who built the system, is a falsifier."
        )
    print("-" * 72)  # noqa: T201

    if failures and not args.no_gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
