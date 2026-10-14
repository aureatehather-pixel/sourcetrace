# SourceTrace

SourceTrace turns free-form notes into structured, attributable knowledge without losing track of where any of it came from.
Raw evidence goes in and is stored unchanged, with its provenance.
Everything derived from it (extracted statements, identity links, relations between statements) is stored separately, points back to the exact source text, and can be reviewed, corrected or rejected without touching the original.

The governing rule: **raw evidence is never lost or rewritten.**
Derived material can always be regenerated; an original observation cannot.

## What it checks, and what it does not

SourceTrace checks provenance mechanically.
It does not verify meaning.

- Every extracted item must quote its source **verbatim**, or the whole extraction run is rejected.
- Every citation in an answer must name an item **and** quote a fragment that appears in that item's quote and in no other.

Both are deterministic string checks.
**A quote occurring in a source does not prove that the quote supports the claim it is attached to.**
An answer can cite a real, verbatim, uniquely bound fragment and still misstate what that fragment says.
The demo below shows that case passing the gate, on purpose.
Whether a statement is a faithful reading of its quote remains a human judgement, or a separately evaluated one; nothing in this repository is a semantic verifier.

There are no confidence scores anywhere.
A model's self-rating is not calibrated against outcomes, so storing one would present precision the system does not have.
Uncertainty is kept only where the evidence itself expresses it, for example as the stance of a reported claim (asserted, recommended, questioned, disputed).

## The four layers

**1. Durable evidence ingestion with provenance.**
A note is stored as-is with its source type, label, URL and capture time, and fingerprinted by content hash.
There is no update or delete path for evidence.
Provenance is never guessed: a capture time without a timezone is refused, and a note with no recorded source is stored with explicitly empty provenance rather than an inferred one.

**2. Schema-validated extraction with verbatim quote checks.**
An LLM turns a note into typed items: entity mentions, reported claims (with speaker and stance), observations (with the subject they are about), and a few other kinds.
The response is parsed strictly: unknown kinds, missing fields and extra keys all fail.
Then every item's grounding quote must appear character for character in the source.
Any failure rejects the whole run and records it as failed; nothing partial is kept.
Long sources are read in windows that must tile the source exactly, and a source is reported as completely processed only when every character belonged to a window that was extracted successfully.
Successful runs are reused by an exact key (content hash, extractor and schema version, provider and model), so reprocessing unchanged evidence costs nothing.

**3. Conservative entity resolution with manual correction.**
Entities exist only because a person created them.
Linking a mention, a speaker or a subject to an entity is a separate, recorded *resolution*, with its method (`exact_match` or `manual`) and its history.
Automatic matching is deliberately narrow: Unicode NFKC normalisation, collapsed whitespace and case folding, and nothing else.
`Kai Lan` does not become `Kailan`, and `Bob` does not become `Robert`.
A match is made only when exactly one entity of the right type has that key; zero or several candidates leave the item unresolved and name the candidates for a person to choose.
A missed match costs one manual step, while a false merge fuses two real people.
Corrections never edit: superseding a resolution deactivates the old row, timestamps it and links it to its replacement.

**4. Bounded cross-evidence relations with review, rejection and versioned reuse.**
For one entity, SourceTrace looks for relations between statements from *different* notes: `reinforces`, `qualifies`, `contradicts` or `no_meaningful_relation`.
There is deliberately no causal relation type.
Which pairs are examined is decided deterministically and for free (a cross-evidence filter, a ranking over existing fields and an absolute cap), never by a model.
Each relation records both source items, the classifier name and version, the model that produced it, and an input fingerprint, and reuse is keyed on all of them.
A reviewer can reject a relation: it stops being served, stays on record, and counts as settled, so the same wrong answer is never paid for twice.
Bumping the classifier version is what reopens settled pairs.

## Quick start: the synthetic demo

The demo needs Docker and [uv](https://docs.astral.sh/uv/), and nothing else: no API key, no `.env`, no cost.

```bash
uv sync --all-groups
uv run python -m demo
```

It starts a disposable PostgreSQL 16 container, applies the real migrations, walks through all four layers on three invented notes, and removes the container.
The two kinds of model call are replaced by scripted responses.
Everything else is the real system: schema, constraints, validation, matching, reuse and the citation gate.
The script is wrong on purpose, so the checks are shown catching it:

1. An extraction that paraphrases its quote is rejected; a verbatim one is accepted.
2. `Dana Reyes` resolves by exact match.
   An ambiguous `Alex Lee` (two on file) is declined, then linked by hand, then corrected by supersession.
3. A cross-note `qualifies` relation is derived, a trivial `reinforces` is rejected by a reviewer, and a re-run reuses every settled pair with zero model calls.
4. The citation gate accepts a verbatim, unique fragment, and rejects a paraphrased fragment and an ambiguous one.
   It then **accepts** a verbatim citation attached to a claim the source contradicts, which is the limit described above.

`tests/integration/test_demo.py` asserts each of these outcomes, so the narration cannot drift from the system.
Every person, place and event in the demo, the examples and the test fixtures is invented.

## Using it on your own notes

Requirements: Python 3.12+, uv, and Docker for the local database.

```bash
uv sync --all-groups
cp .env.example .env              # local defaults match compose.yaml
docker compose up -d              # PostgreSQL 16 on localhost:5433
uv run alembic upgrade head       # an empty database reaches the schema this way alone
```

Set `ANTHROPIC_API_KEY` in `.env` to use a real model.
Without a key, a deterministic fake provider is used: it exercises the whole path but extracts nothing.
`ENVIRONMENT=production` refuses to start without a key.

```bash
uv run pie process examples/notes/01-workshop.md
```

`pie process` ingests the note, extracts items, resolves the names it finds, derives relations and prints the result.
Re-running it on the same file is safe and nearly free.
When the matcher declines (no entity on file, or more than one), it asks rather than guesses.
With `--non-interactive` it asks nothing and prints the exact command that would settle each open question.
An attribution such as `claimed_by = "an engineer at the meetup"` names nobody, so creating a person from it is never offered.

A note may carry front matter recording its provenance.
Only the body is stored as content, so relabelling a note does not change its fingerprint.

```
---
source_type: conversation_note
source_label: Follow-up call
captured_at: 2026-04-15T09:00:00+00:00
---
Follow-up call with Dana Reyes. She said the first month was ...
```

The granular commands are for inspection, correction and reruns:

```bash
pie ingest note.md                       # evidence only
pie extract <evidence-id>                # re-run or inspect extraction
pie show evidence|items|entity|intelligence <id>
pie entity new --type person --name "Dana Reyes" --disambiguator "maker workshop"
pie entity list [--type person]
pie resolve <item-id> --role mention [--entity <entity-id>]
pie analyze <entity-id>                  # derive relations
pie reject <relation-id> --note "technically true and useless"
pie packet [--limit N]                   # the evidence packet an answer is written against
pie validate <answer-file> [--limit N]   # the citation gate
```

Every command takes `--json`.
Exit status is `0` on success, `2` for a usage error and `1` for a failure.

### The evidence packet and the citation gate

`pie packet` renders every grounded item with a packet-local handle (`E1`, `E2`, ...) and its quote.
An answer cites an item as `[E3 "verbatim fragment"]`.
`pie validate` accepts a citation only if the handle exists, the fragment is verbatim inside that item's quote, the fragment is long enough to identify anything, and **no other item's quote contains it**.
Uniqueness is what makes a citation identify one item rather than merely quote something.
Because uniqueness is a property of the whole packet, `pie validate` checks against the whole corpus by default, and refuses rather than silently checking a subset when the corpus will not fit in one packet.
Database ids are never citable: the packet prints none, so an id in an answer was not read from it.

## HTTP API

The same service layer is exposed over HTTP for anything that is not a terminal.

```bash
uv run python -m intelligence_engine     # 127.0.0.1:8000; set HOST and PORT to change
```

OpenAPI docs are at `/docs`, liveness at `/health`, and readiness (PostgreSQL reachable) at `/ready`.
Prefer the module entrypoint to the `uvicorn` CLI on Windows: psycopg's async mode cannot use the default Windows event loop, and the entrypoint selects a compatible one.

## Evaluation

Relation quality is measured against a hand-labelled corpus, not asserted.
`evaluation/relation_gold_set.json` holds 37 labelled pairs.
They cover clear reinforcement, qualification and contradiction, unrelated pairs, and the traps that matter: false contradictions from different times or scopes, attribution traps (a speaker's statement about the *listener*), shared vocabulary with no relationship, and relations that are true but worthless.
Each pair has an `expected` label, a set of `acceptable` labels for genuinely borderline cases, and `forbidden` labels, which are what the traps test.
Results are reported by error category, never as one accuracy number, because the errors differ in cost.
Some pairs record real classifier failures; where the original inputs were private, the pair is a synthetic re-creation and its note says so.

```bash
# Candidate narrowing. Deterministic, no credentials, no cost.
uv run python -m evaluation.run_candidate_eval

# Classifier quality against the labelled pairs. Calls a paid model and needs a key.
uv run python -m evaluation.run_classifier_eval --model claude-sonnet-5
```

No classifier accuracy figure is published with this repository.

## Tests and quality gates

```bash
uv run pytest                        # everything; Docker required
uv run pytest -m "not integration"   # no Docker
uv run ruff check .
uv run ruff format --check .
uv run mypy                          # strict
```

The full suite is 920 tests (595 without Docker, 325 integration), and a green run has **0 skipped**.
If Docker is unavailable while integration tests are selected, the run fails with an explicit message instead of skipping them.
Integration tests run against a disposable PostgreSQL 16 container with the real migrations.
SQLite is never substituted, because it does not share PostgreSQL's JSONB, `TIMESTAMPTZ` or constraint semantics.
Negative cases are tested explicitly, including the false-merge regressions: two different people both called `Alex Lee` must never be merged automatically.

## Configuration

| Variable | Default | Notes |
| --- | --- | --- |
| `DATABASE_URL` | required | Must use the `postgresql+psycopg` scheme |
| `ENVIRONMENT` | `local` | `local`, `test` or `production` |
| `LOG_LEVEL` | `INFO` | |
| `EXTRACTION_PROVIDER` | `anthropic` | `anthropic` or `deepseek` |
| `ANTHROPIC_API_KEY` | unset | Unset selects the fake provider; required in production |
| `EXTRACTION_MODEL` | `claude-opus-5` | |
| `DEEPSEEK_API_KEY` | unset | Read only when `EXTRACTION_PROVIDER=deepseek` |
| `EXTRACTION_WINDOW_MAX_CHARS` | `6000` | Window size for long sources |
| `RELATION_MODEL` | `claude-sonnet-5` | Configured separately from extraction |
| `RELATION_MAX_CANDIDATE_PAIRS` | `60` | The absolute cost bound on one analysis |
| `RELATION_BATCH_SIZE` | `8` | Pairs per model call |

Settings are typed and validated at startup; a missing or malformed `DATABASE_URL` fails immediately.

## Layout

```
src/intelligence_engine/
  evidence/      ingestion, fingerprinting, provenance
  extraction/    prompt, strict validation, grounding check, windowing, providers
  entities/      matching, resolutions, supersession, profiles
  intelligence/  candidate narrowing, relation classification, review, reuse
  packet/        evidence packet, packet-local handles, citation gate
  cli/           the `pie` command
  api/ db/ core/ HTTP routes, sessions, configuration and logging
migrations/      Alembic; all schema change goes through here
demo/            the synthetic walkthrough
examples/notes/  the same synthetic notes, as files for `pie process`
evaluation/      the labelled relation corpus and its runners
tests/           unit/ (no infrastructure) and integration/ (real PostgreSQL)
docs/DESIGN.md   the reasoning behind the main decisions
```

In every layer the service owns the transaction boundary and the repository never commits.
An extraction or analysis attempt is committed before the provider call, so a crash mid-call leaves an honest `running` record instead of nothing.

## Privacy

Logs carry ids, fingerprints, lengths, versions and token counts.
They never carry evidence content or entity names, because both can identify private individuals, and never API keys or provider payloads.
Treat any database you point SourceTrace at as private data.

## License

Licensed under the [Apache License 2.0](LICENSE).
