"""The trust boundary for anything that cites a packet.

This module exists because of three failures, in the order they were found. The
first two were real runs, not hypotheticals; the third was measured against the
real corpus after the second check was believed sufficient. Each produced one of
the three checks below.

## Failure 1 — a fabricated id (the fair-context synthesis experiment)

A model cited `5995bdaf-9b4b-4e9f-9a9b-4605a0e2caac`. The real id was
`5995bdaf-94b3-4e9f-9a9b-4605a0e2caac` - one hex group transposed, `94b3` ->
`9b4b`. It then wrote a parenthetical vouching for its own handling of that id.
The result is a well-formed UUID, spelled consistently, attached to a fabricated
justification, pointing at nothing. A reader cannot detect it; a set lookup
catches it every time.

That gave **id membership**: every cited id must be an item in the packet. It is
the same guarantee `intelligence.validation.bind_results` gives M4, generalised
from a closed request/response pairing to an open set of cited ids.

## Failure 2 — a real id on the wrong claim (the first closed-loop run)

Membership was then believed sufficient, on the reasoning that ids are a closed
set. **That was wrong, and the first real closed-loop run proved it.** Four
citations, four genuine ids, validation passing cleanly - and two of them
describing content that lives in the packet under *different* ids:

    cited e11a3fd5 as "direct engineering contribution was effectively zero"
      - that item says "This was my first hackathon, Emberline DevGuild Lab 1."
      - the quoted content is item 8609f6d6

The model read the right evidence and attached the wrong ids. Membership testing
catches **fabrication**; it cannot catch **misattribution**, because a
misattributed id is a real one.

That gave **quote binding**, below.

## Failure 3 — a real quote that identifies nothing (measured, not observed)

Quote binding was then believed sufficient, on the reasoning that a quote comes
from exactly one row. **That is false for a fragment.** Binding checked only
that the cited fragment occurs *inside* the named item's grounding quote, and
`MIN_CITATION_QUOTE_LENGTH` is 8 characters, so the question "does this fragment
identify this item?" was never asked.

Measured against the 154 packet-eligible items of the Event 2 corpus:

- 151 of 154 items have a quote sharing at least one 8-character fragment with
  some other item's quote;
- 2,109 of the 11,781 item pairs share at least one such fragment;
- 11 items have a quote that is *identical to, or contained in*, another item's
  quote, so no fragment of theirs can ever identify them.

Real collisions from that corpus, both bindable to more than one row:
`'**Qualif'` (many items quote a `**Qualification:**` line) and `':** User'`.

The failing shape is exactly failure 2 wearing a valid costume. Prose describes
item B, the citation carries item A's id, and the fragment is one of the 2,109
collisions — present in A's quote, so containment passes, while identifying
neither. A reader sees an id, a quote, and a green validator.

That gave **unique binding**: the fragment must occur in the cited item's quote
**and in no other packet item's quote**.

## What quote binding is

A citation is no longer an id. It is an identifier *and* a verbatim fragment of
that item's own grounding quote, written together:

    [E17 "My direct engineering contribution"]

The fragment must appear verbatim inside the grounding quote of **the item the
identifier names**. Nothing else is consulted. This is a pure string containment
test against one known row - deterministic, free, and no model call.

A bare identifier anywhere in the answer is rejected. If bare ids were tolerated
the binding would be optional, and an optional check is not a check.

## Failure 4 — a corrupted identifier (the criterion 4 evaluation)

The identifier above used to be a UUID, and answers got it wrong in ways that
membership alone reported as "unknown id" without saying why. A real item id was
written with one extra character appended; an *evidence* id was written where an
item id belonged, four times across two runs.

That gave **packet-local handles**: `E1`, `E2`, ... assigned by packet order,
rendered in place of the UUID, and resolved back to the item id by
`resolve_bound_citations` before any check runs. A uuid is 36 characters of
incompressible noise; `E17` has no interior to corrupt, and a wrong handle fails
resolution loudly instead of pointing at nothing. An evidence id can no longer be
spelled as a citation of an item at all.

**And a uuid is no longer a citable identifier at all** - not even a real item
id, correctly transcribed, carrying a fragment that would pass containment and
uniqueness on its own merits. The packet prints no uuid, so a uuid in an answer
cannot have been read off the evidence it names; it came from somewhere the
packet is not. What binding buys is *evidence that the model read that row*, and
an identifier the model could not have read there is not that evidence, however
well it checks out afterwards. It also leaves one spelling of the identifier
rather than two, so every citation goes through handle resolution and no second
path exists to keep in step with the first.

**Handles are ergonomics and change no guarantee.** The identifier is still
asserted by the answer, exactly as the UUID was. The design that stops at
"resolve the handle, check the quote is inside the resolved item" is failure 3
all over again with a shorter identifier — nothing in it bounds how many *other*
items contain the fragment. So resolution happens first, and containment and
uniqueness then apply to the resolved item unchanged. That equivalence is what
`tests/unit/test_packet_handles.py` asserts, case by case.

## What quote binding does and does not do

**Does:** rejects a citation whose quote belongs to a different item. Under this
contract the misattribution above is caught mechanically, because the model must
either copy `e11a3fd5`'s real quote - and then its own sentence visibly
contradicts the quote sitting next to it - or copy `8609f6d6`'s quote under
`e11a3fd5`'s id, which fails the binding.

**Does:** rejects a fragment that fits more than one item, whichever item it was
filed under. A citation that cannot single out a row is not evidence that the
prose beside it came from that row.

**Does not:** make every item citable. Where two items were extracted from one
sentence and carry the same grounding quote verbatim, *neither* can be uniquely
bound, and this rejects both. That is a deliberate reversal of the earlier
decision to allow it, and the reversal is the honest direction: the previous
behaviour did not make those two items distinguishable, it only declined to
check. An item that cannot be told apart from another is reported as ambiguous
rather than passed as though it had been identified. `unbindable_items` names
them, so the condition is inspectable instead of only discoverable by rejection.

**Does not:** check that the surrounding prose claim *follows from* the quote.
That is semantic entailment, it needs a judge, and it is deliberately not built
here. What binding buys is that the reader is shown the evidence beside the
claim, and that the id and the quote cannot come from different rows.

Stated plainly so nobody reads more into a green result than it means:

    binding proves the citation points at the evidence it quotes.
    It does not prove the sentence around it is true.

## Deliberate non-behaviours

- **No repair.** A transposed id is not corrected to the nearest real one.
- **No fuzzy matching.** No edit distance, no prefix matching, no "5995bdaf...
  is probably this one".
- **No inference of intent.**
- **No length floor standing in for uniqueness.** Raising
  `MIN_CITATION_QUOTE_LENGTH` would reduce collisions without bounding them; a
  long fragment of a long shared quote is still shared. Length is a separate
  and much weaker question, and it is kept separate.

Each of those would take the exact failure this exists to catch and convert it
back into a citation a reader would trust.

## One tolerance, named

Comparison normalises **runs of whitespace, and nothing else**. That is not
leniency about content: this package's own renderer line-wraps long quotes to
fit a terminal, so a model copying a quote verbatim out of a rendered packet
reproduces line breaks and indentation *we* inserted. Refusing those would fail
correct citations for a defect of our own formatting, and a validator that
rejects correct input gets switched off. Case, punctuation and every character
are compared exactly.

## Rejection is whole-answer, not per-citation

Exactly as `bind_results` rejects the whole batch. Dropping the bad citation and
keeping the rest would leave the sentence it supported standing with its support
silently removed - a claim that reads as evidence-backed and is not. An answer
that cited something imaginary, or bound a quote to the wrong row, is an answer
produced by a process already known to be unreliable.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from intelligence_engine.extraction.schemas import MIN_QUOTE_LENGTH
from intelligence_engine.packet.errors import InvalidCitation
from intelligence_engine.packet.schemas import EvidencePacket

#: Matches the canonical 8-4-4-4-12 hyphenated form and nothing else. Case is
#: tolerated because a model may echo an id upper-cased; layout is not, because a
#: string that is not a UUID was never a citation of an item id.
#:
#: `\b` boundaries keep an id embedded in prose or parentheses findable while
#: refusing a longer hex run that merely contains one.
_UUID_PATTERN = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
_UUID_RE = re.compile(rf"\b{_UUID_PATTERN}\b")

#: A packet-local handle: `E1`, `E17`, ... Assigned by position in the packet and
#: rendered in place of the uuid, for one measured reason. A uuid is 36
#: characters of incompressible noise, and answers corrupted them: a real id was
#: written with one extra character appended, and evidence ids were written where
#: item ids belonged. A handle has no interior to corrupt, and a wrong one fails
#: resolution loudly instead of pointing at nothing.
#:
#: **A handle is ergonomics, not a guarantee.** It is asserted by the answer
#: exactly as a uuid was, and it is resolved to the item id before any check
#: runs. Containment and packet-wide uniqueness are applied to the resolved item
#: unchanged - see `_check_quote`. Replacing the identifier's spelling does not
#: and cannot establish which row a fragment came from; only uniqueness does
#: that, and removing it to make duplicate-quote items citable would reopen the
#: defect unique binding was introduced to close.
_HANDLE_PATTERN = r"E[1-9][0-9]{0,3}"
_HANDLE_RE = re.compile(rf"\b{_HANDLE_PATTERN}\b")

#: Handle-shaped, not handle-valid: anything beginning `E` and a digit. Used only
#: by the bare-identifier check, so `E1x` and `E0` are rejected as malformed
#: rather than parsed as prose and dropped.
_HANDLE_LIKE_RE = re.compile(r"\bE[0-9][A-Za-z0-9]*")


def handle_for_index(index: int) -> str:
    """The handle of the item at `index` in packet order. Zero-based in, 1-based out."""
    return f"E{index + 1}"


def packet_handles(packet: EvidencePacket) -> dict[str, uuid.UUID]:
    """Handle -> item id, in packet order. A pure function of the packet.

    Deterministic and total: every item has exactly one handle and every handle
    resolves to exactly one item. Regenerating the packet regenerates the map,
    which is why the map is not stored anywhere.
    """
    return {handle_for_index(i): item.item_id for i, item in enumerate(packet.items)}


#: A bound citation: `[<id> "<quote>"]`.
#:
#: The tolerances here are all about *our* syntax, never about the quoted
#: content. Curly quotation marks are accepted as delimiters because a model
#: typesetting prose will produce them; an optional `:`, `,`, `;` or dash between
#: the id and the quote is accepted for the same reason. The text between the
#: delimiters is then compared exactly (modulo whitespace runs, see the module
#: docstring).
#:
#: Angle brackets around the id are accepted for a reason that was **observed,
#: not anticipated**: the packet states the citation shape as
#: `[<id> "<verbatim quote>"]`, and on the first rerun a model copied the
#: placeholder's angle brackets straight through into its answer. The id inside
#: was exact and the quote was exact - only our own notation had leaked.
#: Rejecting that would have failed a substantively correct citation on a defect
#: of our instructions, and a gate that rejects correct input is a gate someone
#: switches off. The packet text now says the brackets are placeholders; this
#: accepts them anyway, because both halves belong in the fix.
_BOUND_RE = re.compile(
    rf"""\[\s*
        <?\s*(?P<id>{_UUID_PATTERN}|{_HANDLE_PATTERN})\s*>?
        \s*[:,;\-\u2013\u2014]?\s*
        ["\u201c\u201d](?P<quote>[^"\u201c\u201d]*)["\u201c\u201d]
        \s*\]""",
    re.VERBOSE,
)

#: Bounded so a long list of bad citations cannot become a large log line.
MAX_REPORTED_INVALID = 3

#: The shortest fragment that counts as a quote. Reused from M2 rather than
#: chosen here: `MIN_QUOTE_LENGTH` is already this system's answer to "how short
#: can a quote be before it stops being one", and a citation fragment is a quote.
MIN_CITATION_QUOTE_LENGTH = MIN_QUOTE_LENGTH


@dataclass(frozen=True, slots=True)
class BoundCitation:
    """One `[id "quote"]` pair as it was written in the answer."""

    item_id: uuid.UUID
    quote: str


@dataclass(frozen=True, slots=True)
class CitationAudit:
    """What an answer cited, checked against the packet it was given.

    `uncited` is reported rather than enforced. An answer is not required to use
    every item - silence about an item is a legitimate editorial choice, whereas
    citing one that does not exist is not - but knowing what went unused is how
    a reader judges whether the packet was actually read.
    """

    cited: tuple[uuid.UUID, ...]
    uncited: tuple[uuid.UUID, ...]
    bound: tuple[BoundCitation, ...] = ()

    @property
    def cited_count(self) -> int:
        return len(self.cited)


def _squeeze(text: str) -> str:
    """Collapse every run of whitespace to one space. The only normalisation."""
    return " ".join(text.split())


def extract_cited_ids(text: str) -> tuple[str, ...]:
    """Every UUID-shaped string in an answer, in order of first appearance.

    Deduplicated, because citing one item twice is ordinary prose, not an error -
    unlike M4, where a pair answered twice means the model lost track of its own
    batch.
    """
    seen: dict[str, None] = {}
    for match in _UUID_RE.findall(text):
        seen.setdefault(match.lower(), None)
    return tuple(seen)


def extract_bound_citations(text: str) -> tuple[BoundCitation, ...]:
    """Every well-formed `[id "quote"]` in an answer, in order of appearance.

    Not deduplicated: the same item quoted twice at different lengths is two
    bindings, and both have to hold.
    """
    citations: list[BoundCitation] = []
    for match in _BOUND_RE.finditer(text):
        try:
            item_id = uuid.UUID(match.group("id"))
        except ValueError:
            # A handle, not a uuid. Resolving it needs the packet, so it is left
            # to `resolve_bound_citations`; this function stays a pure text scan.
            continue
        citations.append(BoundCitation(item_id=item_id, quote=match.group("quote")))
    return tuple(citations)


def resolve_bound_citations(text: str, packet: EvidencePacket) -> tuple[BoundCitation, ...]:
    """Every `[E17 "quote"]`, with the handle resolved to an item id.

    Resolution happens here and nowhere else, so everything downstream — the
    membership check, containment, and packet-wide uniqueness — sees an item id
    and behaves exactly as it did before handles existed. That is the whole point
    of doing it this way: a handle is a spelling of the identifier, and changing
    a spelling must not change what is guaranteed.

    An unresolvable handle is rejected here rather than passed on as a missing
    item, because "E900 is not in this packet" is a different and more useful
    finding than "some id was unknown".

    **A uuid is not an identifier an answer may use.** The renderer prints no
    uuid anywhere in the packet, so a uuid in an answer cannot have been read off
    the evidence it claims to cite: it came from somewhere the packet is not.
    Accepting a *real* item id — which passes membership, containment and
    uniqueness on its own merits — would mean accepting a citation that is not
    evidence the rendered packet was read, which is the one thing binding exists
    to establish. It is rejected with its own message rather than falling through
    to the bare-id check, because "you used a uuid, cite by handle" is actionable
    and "an id appeared without a bound quote" would be a false description of
    what the answer actually did.
    """
    handles = packet_handles(packet)
    citations: list[BoundCitation] = []
    unknown: list[str] = []
    uuids: list[str] = []
    for match in _BOUND_RE.finditer(text):
        ident = match.group("id")
        try:
            uuid.UUID(ident)
        except ValueError:
            resolved = handles.get(ident)
            if resolved is None:
                unknown.append(ident)
                continue
            citations.append(BoundCitation(item_id=resolved, quote=match.group("quote")))
            continue
        uuids.append(ident)
    if uuids:
        raise InvalidCitation(
            f"{len(uuids)} citation(s) use a uuid instead of a packet handle: "
            f"{_sample(uuids)}. This packet prints no uuid, so a uuid in an answer "
            f"was not read from it. This packet has {len(handles)} item(s), "
            f'E1 to E{len(handles)}; cite as [E<n> "<verbatim quote>"].'
        )
    if unknown:
        raise InvalidCitation(
            f"{len(unknown)} handle(s) do not name an item in this packet: "
            f"{_sample(unknown)}. This packet has {len(handles)} item(s), "
            f"E1 to E{len(handles)}."
        )
    return tuple(citations)


def validate_citations(
    cited: Iterable[str], packet: EvidencePacket, *, require_any: bool = False
) -> CitationAudit:
    """Require every cited id to be an item in `packet`, or reject the answer.

    The membership floor, and the primitive `validate_bound_answer` builds on. It
    catches fabricated ids and nothing else; on its own it is **not** a
    sufficient gate, which is what the first closed-loop run established.

    Raises `InvalidCitation` on the first violation class found, reporting a
    bounded list. Malformed and unknown ids are reported separately because they
    are different failures: a malformed string means the model emitted something
    that was never an id, an unknown one means it emitted a plausible id for an
    item that does not exist - the laundering case.

    `require_any` is off by default. An answer that legitimately concludes "no
    evidence in this packet bears on this" cites nothing and is correct to;
    turning that into a failure would pressure the next answer to cite something.
    """
    malformed: list[str] = []
    unknown: list[str] = []
    parsed: list[uuid.UUID] = []
    known = packet.item_ids

    for raw in cited:
        try:
            candidate = uuid.UUID(raw)
        except ValueError:
            malformed.append(raw)
            continue
        if candidate not in known:
            unknown.append(raw)
            continue
        if candidate not in parsed:
            parsed.append(candidate)

    if malformed:
        raise InvalidCitation(
            f"{len(malformed)} citation(s) are not item ids: {_sample(malformed)}"
        )
    if unknown:
        raise InvalidCitation(
            f"{len(unknown)} cited id(s) are not in the supplied packet of "
            f"{len(known)} items: {_sample(unknown)}"
        )
    if require_any and not parsed:
        raise InvalidCitation("the answer cites no item in the packet")

    return _audit(parsed, packet)


def validate_bound_answer(
    answer: str, packet: EvidencePacket, *, require_any: bool = False
) -> CitationAudit:
    """The full gate: every id bound to a verbatim fragment of its own quote.

    Rejections, in the order a failing answer usually hits them:

    0. **A uuid used as an identifier.** The packet prints none, so it was not
       read from the packet. See `resolve_bound_citations`.
    1. **A bare id.** Every UUID in the answer must sit inside `[id "quote"]`.
       Tolerating a loose id would make the binding optional.
    2. **An unknown id.** Delegated to `validate_citations` - failure 1 above.
    3. **A fragment too short to be a quote.** Below `MIN_CITATION_QUOTE_LENGTH`
       a fragment identifies nothing.
    4. **A fragment that is not in that item's grounding quote.** Failure 2
       above.
    5. **A fragment that is in that item's quote and in others too.** Failure 3
       above. Containment proves the fragment came from somewhere; only
       uniqueness proves it came from *there*.

    On (4) the error names the item the fragment *does* belong to when one
    exists. That is a plain substring search over the packet, not an inference
    about meaning, and it turns "this citation is wrong" into "you meant this
    row" for the human reading the rejection. On (5) it names the items the
    fragment collides with, and says whether a longer fragment would fix it or
    whether the item is intrinsically unbindable in this packet.
    """
    bound = resolve_bound_citations(answer, packet)
    _reject_bare_ids(answer, bound)

    audit = validate_citations(
        [str(citation.item_id) for citation in bound], packet, require_any=require_any
    )

    quotes = _Quotes(packet)
    for citation in bound:
        _check_quote(citation, quotes)

    return CitationAudit(cited=audit.cited, uncited=audit.uncited, bound=bound)


def _reject_bare_ids(answer: str, bound: Sequence[BoundCitation]) -> None:
    """Every identifier in the text must fall inside a bound-citation span.

    Handles are held to the same rule as uuids, for the same reason: a bare
    identifier makes the binding optional, and an optional check is not a check.

    A handle is a short token, so this is deliberately strict about a cost it
    could otherwise hide — any `E12`-shaped word outside a citation is rejected
    even if the packet has no twelfth item. Prose that legitimately needs to
    write `E5` cannot, and that is the trade: a false rejection is visible and
    arguable, whereas a bare handle silently reintroduces unbound citation.
    """
    spans = [match.span() for match in _BOUND_RE.finditer(answer)]

    def outside(match: re.Match[str]) -> bool:
        return not any(start <= match.start() and match.end() <= end for start, end in spans)

    loose = [match.group(0) for match in _UUID_RE.finditer(answer) if outside(match)]
    if loose:
        raise InvalidCitation(
            f"{len(loose)} id(s) appear without a bound quote: {_sample(loose)}. "
            'Every citation must be written as [E<n> "<verbatim quote>"].'
        )

    # Handle-*shaped* rather than handle-valid, deliberately. `[E1x "quote"]`
    # matches no citation pattern and no valid handle, so a narrower check let it
    # through as text and the citation vanished without a word — the silent drop
    # this validator exists to prevent. Anything starting `E` followed by a digit
    # is therefore claimed by this check, and a malformed one is rejected rather
    # than ignored.
    loose_handles = [match.group(0) for match in _HANDLE_LIKE_RE.finditer(answer) if outside(match)]
    if loose_handles:
        raise InvalidCitation(
            f"{len(loose_handles)} handle(s) appear without a bound quote: "
            f"{_sample(loose_handles)}. "
            'Every citation must be written as [E<n> "<verbatim quote>"].'
        )


def _check_quote(citation: BoundCitation, quotes: _Quotes) -> None:
    """One citation against every quote in the packet. Two questions, in order.

    Containment first, because "you quoted a different row" and "you quoted a row
    that cannot be told from another" are different findings and a reader needs
    to know which one happened. Uniqueness second, over the same normalised
    quotes, so the two answers can never be drawn from different readings of the
    same text.
    """
    fragment = _squeeze(citation.quote)
    if len(fragment) < MIN_CITATION_QUOTE_LENGTH:
        raise InvalidCitation(
            f"the quote bound to {citation.item_id} is {len(fragment)} characters, "
            f"below the {MIN_CITATION_QUOTE_LENGTH} needed to identify an item"
        )

    others = quotes.containing(fragment, excluding=citation.item_id)

    if fragment not in quotes.of(citation.item_id):
        detail = (
            f" That quote is from item {others[0]}."
            if others
            else " No item in the packet has that quote."
        )
        raise InvalidCitation(
            f"the quote bound to {citation.item_id} does not appear in that item's "
            f"grounding quote.{detail}"
        )

    if not others:
        return

    # The fragment is in the right row and in others too, so it names none of
    # them. Which of the two repairs is available is a property of the packet
    # rather than of the answer, so the rejection says which one this is.
    if quotes.has_unique_fragment(citation.item_id):
        raise InvalidCitation(
            f"the quote bound to {citation.item_id} also appears in "
            f"{len(others)} other item(s): {_sample([str(other) for other in others])}. "
            "A fragment that fits more than one item identifies none of them; "
            "quote a longer stretch that appears only in the item you are citing."
        )
    raise InvalidCitation(
        f"item {citation.item_id} cannot be uniquely cited in this packet: its whole "
        f"grounding quote also appears in item {others[0]}, so no fragment of it "
        "identifies this item rather than that one. This is a property of the stored "
        "evidence, not of the citation."
    )


class _Quotes:
    """Every packet item's grounding quote, normalised once per answer.

    Built once and passed down rather than recomputed per citation. Uniqueness is
    a question about the whole packet, so every citation in an answer must ask it
    against the same normalised text; `_squeeze` on the fly would be correct and
    quadratic, and would leave two call sites free to normalise differently.
    """

    __slots__ = ("_order", "_quotes")

    def __init__(self, packet: EvidencePacket) -> None:
        # Packet order, so every diagnostic below is deterministic.
        self._order: tuple[uuid.UUID, ...] = tuple(item.item_id for item in packet.items)
        self._quotes: dict[uuid.UUID, str] = {
            item.item_id: _squeeze(item.grounding_quote) for item in packet.items
        }

    def of(self, item_id: uuid.UUID) -> str:
        return self._quotes[item_id]

    def containing(self, fragment: str, *, excluding: uuid.UUID) -> tuple[uuid.UUID, ...]:
        """Every other item whose quote contains this fragment, in packet order.

        A diagnostic and a decision, never a repair: it can only turn an
        acceptance into a rejection, and it never rewrites a citation.
        """
        return tuple(
            item_id
            for item_id in self._order
            if item_id != excluding and fragment in self._quotes[item_id]
        )

    def has_unique_fragment(self, item_id: uuid.UUID) -> bool:
        """Whether *any* fragment of this item's quote identifies it.

        Exactly: whether the item's whole quote fails to appear inside another
        item's. If it stands alone, the whole quote is itself a uniquely
        identifying fragment. If it is contained in another item's quote, then
        every fragment of it is contained there too and none can identify it. So
        one containment test on the full quote answers the question for every
        fragment at once, with no search.
        """
        quote = self._quotes[item_id]
        return not any(
            quote in other for candidate, other in self._quotes.items() if candidate != item_id
        )


def unbindable_items(packet: EvidencePacket) -> tuple[uuid.UUID, ...]:
    """Items no citation can ever bind, because another item's quote contains theirs.

    Reported rather than repaired, and deliberately public: the condition is a
    property of the extracted evidence — one sentence yielding two items — and
    the only honest handling is to name it. On the Event 2 corpus this is 11 of
    154 items. Nothing in the validation path calls this; it exists so the
    condition can be inspected before an answer is written rather than only
    discovered when one is rejected.
    """
    quotes = _Quotes(packet)
    return tuple(
        item.item_id for item in packet.items if not quotes.has_unique_fragment(item.item_id)
    )


def _audit(cited: Sequence[uuid.UUID], packet: EvidencePacket) -> CitationAudit:
    seen = set(cited)
    return CitationAudit(
        cited=tuple(cited),
        uncited=tuple(item.item_id for item in packet.items if item.item_id not in seen),
    )


def _sample(values: Sequence[str]) -> str:
    shown = ", ".join(repr(value) for value in values[:MAX_REPORTED_INVALID])
    if len(values) > MAX_REPORTED_INVALID:
        shown += f", and {len(values) - MAX_REPORTED_INVALID} more"
    return shown
