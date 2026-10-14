# Design notes

This document records the reasoning behind SourceTrace's main decisions.
The README says what the system does; this says why it is built that way, and what each guarantee does not cover.
Every example is invented.

## Evidence is the only thing that cannot be regenerated

Raw evidence is immutable and lives in its own table.
Extraction runs, entity resolutions and relations are separate layers that point back to it and never write to it.
The consequence is that any derived layer can be thrown away and rebuilt, and no bug in a derived layer can damage the source.

Work that calls a model records its attempt *before* the call.
If the process dies mid-call, the database holds an honest `running` row rather than no record at all, and nothing claims a result that never arrived.

## No confidence scores

A model's self-reported confidence is not calibrated against outcomes in this system, so storing it would present precision that does not exist.
Uncertainty is represented only where the evidence expresses it: hedged wording is kept as written, and a reported claim records its stance (asserted, recommended, questioned or disputed).
For entity resolution the stored field is `method` (`exact_match` or `manual`), which is checkable, rather than a probability, which would not be.

## What a verbatim quote proves

Every extracted item carries a grounding quote that must appear character for character in the source, or the whole run is rejected.
This is the strongest hallucination check available without a second model: an invented quote cannot pass it.

It proves less than it might appear to.
It shows that the quoted words exist in the source.
It does not show that the extracted statement is a faithful reading of them, that the claim is true, or that the quote is the best support available.
Those remain judgements, and the system is built so that a person can make them, with the quote and its source one step away.

The same limit applies to the citation gate (below), and the demo deliberately shows it.

## Reading long sources

A long source is divided into windows, and each window is extracted separately.
The windows are exact, contiguous, non-overlapping slices that tile the whole source, so per-window grounding implies source-level grounding.
When a window overflows the model's output, it is subdivided deterministically rather than resized by guesswork.

A source is reported `complete` only when the successful windows tile it with no gap.
That is processing completeness and nothing more: every character was read by a successful extraction.
It is not a claim that the model found every meaningful item in those characters.

## Mention, entity and resolution are three different things

`"Dana Reyes"` appearing in a note is a mention.
A row in `entities` is an identity that a person decided exists.
Deciding that this mention refers to that identity is a resolution: derived, fallible, and therefore stored as its own row with its own history.

Nothing creates an entity as a side effect of ingestion or extraction.
An attribution such as "an engineer at the meetup" identifies nobody, and the system will not manufacture a person from it.

### Conservative matching

The matching key is Unicode NFKC normalisation, collapsed whitespace and case folding.
Nothing else is applied, on purpose:

- `Kai Lan` and `Kailan` stay different, because removing a space guesses at a naming convention.
- `Bob` and `Robert` stay different, because a nickname is a claim, not a normalisation.
- `Acme Inc.` and `Acme` stay different, because legal suffixes can distinguish real organisations.

A match is asserted only when exactly one candidate of the right type exists.
Zero or several candidates leave the item unresolved and name the candidates.
The asymmetry is the point: a missed match costs one manual resolution, while a false merge silently fuses two people's records.
Unresolved is a valid, permanent state.

### Corrections append

Superseding a resolution deactivates the old row, stamps it with `superseded_at` and links it to its replacement.
Re-resolving an item that already has an active resolution is refused rather than overwritten.
Entity profiles are assembled by query from active resolutions on every request, so there is no stored profile to drift, and detaching a wrong link immediately withdraws everything derived through it.

## A speaker is not a subject

A reported claim reaches an entity through `claimed_by`: the entity *said* it.
An observation reaches an entity through `about`: the evidence records it as being *about* them, and nobody asserted it.
The two are never merged into one "attributed to" field, and the database enforces which role each item kind may take.

The distinction matters because speakers very often talk about someone else.
"Ines told me that I would lose my first few contests" is said by Ines and is about the listener.
Treating it as a statement about Ines, and then relating it to facts about Ines, invents a history nobody reported.
The relation classifier is shown each statement's connection explicitly, and the labelled corpus contains attribution traps built on exactly this shape.

## Choosing which pairs to classify

All-pairs classification is ruled out: 100 statements make 4,950 pairs.
Candidate selection is deterministic and free: a hard cross-evidence filter (two statements from one note are not cross-evidence intelligence), a ranking over fields the schema already has, and an absolute cap.

Vocabulary overlap is a tiebreak, never a gate.
The planted relations in the candidate evaluation score at or near zero word overlap, while unrelated pairs score far higher; a similarity threshold would discard exactly the relations this layer exists to find.
`uv run python -m evaluation.run_candidate_eval` reproduces that measurement.

## Relations are assertions, so they are reviewable and versioned

A relation is a new claim that neither source makes, so it can be wrong.
It is stored once per pair in canonical order, so "A contradicts B" and "B contradicts A" cannot both exist and disagree.
Its explanation is derived reasoning and is never presented as a quote.

The errors are not equally costly.
In order, worst first: invented causality, a false contradiction, a false relation, a trivial relation presented as insight, and only then a missed relation.
`no_meaningful_relation` is a correct and common answer, and it is stored, because caching the negative is what makes re-analysis free.

A settled pair is reused on an exact key: both item ids, classifier name and version, schema version, the provider and model that produced it, and an input fingerprint.
Leaving the model out of that key would let a comparison between two models silently measure the first one twice.
A rejected relation counts as settled, so a wrong answer is never paid for twice.
Bumping the classifier version is what reopens the question.

## Prompts are versioned behaviour

The extraction and relation prompts are part of their component's behaviour, and stored output is reused by version.
If a prompt could change without its version changing, the system would serve results produced by instructions that no longer exist.
So each prompt's fingerprint is pinned to its version in a test, and changing the text without a version bump fails the suite.

## Citations bind a handle to a quote

The evidence packet gives every item a short packet-local handle (`E1`, `E2`, ...).
A citation must carry both a handle and a fragment of that item's quote: `[E3 "verbatim fragment"]`.

Three rules make a citation identify one item rather than merely look plausible:

1. The fragment must be verbatim inside the named item's own quote.
2. The fragment must appear in no other item's quote.
   Containment proves the fragment came from somewhere; only uniqueness proves it came from there.
   Items whose quotes are identical to, or contained in, another item's are reported as unbindable rather than passed.
3. Database ids are not citable.
   The packet prints none, so an id in an answer was not read from the packet it claims to cite.

Uniqueness is a property of the whole packet, so validating against a smaller packet than the answer was written against could accept an ambiguous citation.
Validation therefore uses the whole corpus by default and refuses, rather than quietly checking a prefix, when the corpus does not fit in one packet.

None of this checks meaning.
A citation can satisfy all three rules and still be attached to a claim its source contradicts.

## Tests that can fail

Integration tests run against a real, disposable PostgreSQL 16 with the real migrations, never SQLite.
If integration tests are selected and Docker is unavailable, the run fails loudly instead of reporting skipped tests as a green result.
The supported way to run without Docker is to deselect them explicitly with `-m "not integration"`.
A green run has zero skipped tests, so any skip means part of the suite did not run.
