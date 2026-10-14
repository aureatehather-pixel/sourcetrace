"""Rendering a packet as text, deterministically and without a model.

This is the provider-independent half of the capability. It runs at an Anthropic
balance of zero, with no credentials configured and no network — which is the
point: the experiment measured PIE's advantage as retrieval and provenance, and
both live entirely on this side of the model call.

The output is written to be pasted somewhere. It states the citation rule in the
packet itself rather than assuming the caller's prompt will, so a packet handed
to a person, a chat window or a provider carries the same contract with it.

Two rendering rules follow from the epistemics rather than from taste:

- **Absent stays absent.** A field the evidence did not supply prints no line at
  all. It never prints "unknown", which would be a claim the evidence did not
  make.
- **`claimed_by` and `subject` are labelled differently and never merged.** "Zi
  Lan said X" and "X is recorded about Zi Lan" are rendered as the different
  statements they are, because a reader who conflates them has been misled by
  this function rather than by the evidence.
"""

from __future__ import annotations

import textwrap

from intelligence_engine.packet.citations import CitationAudit, handle_for_index
from intelligence_engine.packet.schemas import EvidencePacket, PacketItem, PacketSource

WIDTH = 96

#: Stated in the packet rather than left to the caller's prompt, so a packet
#: pasted into a chat window carries the same contract as one handed to a
#: provider.
#:
#: The example is written as a *concrete* handle, `E17`, and that is deliberate
#: even though a model may echo it. The two candidate mistakes are not equal.
#: An echoed `[E17 "verbatim quote"]` parses, resolves, and then fails
#: containment loudly, naming the problem. An abstract placeholder like `E<n>`
#: would not parse as a citation at all, so an echo of it would be silently
#: dropped and the answer would simply appear to cite less. A rejection a reader
#: can act on beats a citation that quietly disappears.
_CONTRACT_LINES = (
    "Every item below is stored evidence, and carries a handle (E1, E2, ...) and a quote.",
    'Cite an item by writing BOTH, together, like this:  [E17 "verbatim quote"]',
    "Write the handle exactly as it appears on the item. The quote must be copied "
    "exactly from that same item's own quote: line - any unbroken stretch of it, "
    "long enough to identify the item.",
    "A handle written on its own, a handle that does not appear below, or a quote "
    "taken from a different item than the handle names, rejects the whole answer.",
    "A fragment that also appears in another item's quote identifies neither, and "
    "is rejected: quote a longer stretch that appears only in the item you are citing.",
)


def render_packet(packet: EvidencePacket) -> str:
    """The whole packet as text. A pure function of the packet."""
    lines: list[str] = ["GROUNDED EVIDENCE PACKET", ""]
    lines.extend(_wrap(_headline(packet), indent=""))
    for line in _CONTRACT_LINES:
        lines.append("")
        lines.extend(_wrap(line, indent=""))

    # Handles are assigned over `packet.items` in packet order, before the items
    # are grouped for display. Grouping is presentation; the handle must follow
    # the same order the resolver uses, or E17 would mean different rows to the
    # renderer and to the validator.
    handles = {item.item_id: handle_for_index(i) for i, item in enumerate(packet.items)}

    by_evidence: dict[str, list[PacketItem]] = {}
    for item in packet.items:
        by_evidence.setdefault(str(item.evidence_id), []).append(item)

    for source in packet.sources:
        lines.append("")
        lines.extend(_source_lines(source))
        for item in by_evidence.get(str(source.evidence_id), []):
            lines.append("")
            lines.extend(_item_lines(item, handles[item.item_id]))

    if not packet.items:
        # Stated, not implied. An empty packet is a real answer — "nothing
        # stored bears on this" — and must not read as a rendering failure.
        lines.extend(["", "  No grounded items are stored. The packet is empty, not filtered."])

    return "\n".join(lines)


def _headline(packet: EvidencePacket) -> str:
    shown = len(packet.items)
    sources = len(packet.sources)
    if packet.truncated:
        return (
            f"{shown} of {packet.total_items_available} grounded items, from {sources} "
            f"source(s). TRUNCATED: {packet.total_items_available - shown} stored item(s) "
            "are not shown. Raise the limit before relying on this packet as complete."
        )
    return (
        f"{shown} grounded item(s) from {sources} source(s). "
        "Complete: every stored grounded item is included."
    )


def _source_lines(source: PacketSource) -> list[str]:
    label = source.source_label or "(no source label recorded)"
    lines = [f"SOURCE  {source.evidence_id}", f"        {label}"]
    detail = [f"{source.item_count} item(s)"]
    if source.source_type:
        detail.append(source.source_type)
    if source.captured_at:
        detail.append(f"captured {source.captured_at.isoformat()}")
    if source.source_url:
        detail.append(source.source_url)
    if not source.complete:
        # Stated in characters of one source, never as a source count. A reader
        # must not be able to read partial coverage as several documents.
        detail.append(f"read {source.covered_chars} of {source.source_length} characters")
    lines.append("        " + " · ".join(detail))
    if not source.complete:
        # Loud, and above the items rather than in a footnote: every claim built
        # on this source is built on a source PIE knows it did not finish
        # reading, and nothing downstream can infer that on its own. The unread
        # ranges are named, because "something failed" cannot be checked and a
        # character range can.
        ranges = ", ".join(f"[{start}:{end})" for start, end in source.uncovered)
        lines.append(
            f"        INCOMPLETE: {source.source_length - source.covered_chars} character(s) "
            f"of this source were never extracted — {ranges}. "
            f"Items below are a partial reading."
        )
    return lines


def _item_lines(item: PacketItem, handle: str) -> list[str]:
    # The handle is the citation token and the uuid is not printed. A uuid in
    # this position was copied wrong in real answers - once as a real id with an
    # extra character appended, and repeatedly as an evidence id where an item id
    # belonged. An identifier the answer never sees cannot be mis-transcribed.
    lines = [f"  [{handle}]  {item.kind.value}"]
    lines.extend(_wrap(item.text, indent="    "))

    if item.claimed_by is not None:
        lines.append(f"      asserted by (as worded): {item.claimed_by}")
    if item.subject is not None:
        lines.append(f"      observed about (as worded): {item.subject}")
    if item.stance is not None:
        lines.append(f"      stance: {item.stance}")
    for name, value in item.attributes:
        lines.extend(_wrap(f"{name}: {value}", indent="      "))
    for context in item.entities:
        # The role is printed, never collapsed into "linked to". `claimed_by`
        # and `about` are different assertions and are the reason this field
        # exists at all.
        lines.append(
            f"      entity: {context.canonical_name} ({context.role.value}, {context.entity_id})"
        )
    if item.source_label:
        lines.append(f"      source: {item.source_label}")
    if item.captured_at:
        lines.append(f"      captured: {item.captured_at.isoformat()}")
    lines.extend(_wrap(f'quote: "{item.grounding_quote}"', indent="      "))
    return lines


def _wrap(text: str, *, indent: str) -> list[str]:
    return textwrap.wrap(
        text, width=WIDTH, initial_indent=indent, subsequent_indent=indent + "  "
    ) or [indent + text]


def render_audit(audit: CitationAudit, packet: EvidencePacket) -> str:
    """What a passing answer actually cited, each claim beside its evidence.

    Only reached when validation passed, so every line here is a citation whose
    id and quote came from the same row. It is printed rather than assumed
    because the point of binding is that a human can see the evidence next to
    the claim — a green exit status alone would put the reader back to trusting
    the model, which is what this whole path exists to stop.

    The scope line exists because "PASSED" means different things over different
    packets. Uniqueness is packet-wide, so a verdict reached over a truncated
    packet is a weaker statement than the same word over the whole corpus — a
    fragment can be unique among 200 items and ambiguous among 496. Printing the
    scope makes that visible instead of leaving the reader to assume the stronger
    reading.
    """
    lines = ["CITATION VALIDATION: PASSED"]
    lines.extend(_wrap(f"scope: {_audit_scope(packet)}", indent="  "))
    lines.extend(
        [
            f"  {len(audit.bound)} bound citation(s) over {audit.cited_count} "
            f"of {len(packet.items)} packet items",
            f"  {len(audit.uncited)} item(s) went uncited",
        ]
    )
    by_id = {item.item_id: item for item in packet.items}
    for citation in audit.bound:
        item = by_id[citation.item_id]
        lines.append("")
        lines.append(f"  [{citation.item_id}]  {item.kind.value}")
        lines.extend(_wrap(f'cited: "{_squeeze(citation.quote)}"', indent="    "))
        lines.extend(_wrap(f'stored: "{item.grounding_quote}"', indent="    "))
        if item.source_label:
            lines.append(f"    source: {item.source_label}")
    return "\n".join(lines)


def _audit_scope(packet: EvidencePacket) -> str:
    """What the verdict above covers. Truncation is stated, never implied."""
    if packet.truncated:
        return (
            f"{len(packet.items)} of {packet.total_items_available} stored item(s) — "
            "TRUNCATED. Uniqueness was checked over this prefix only, so a fragment "
            "colliding with an item outside it would not have been caught."
        )
    return f"all {packet.total_items_available} stored item(s) — complete corpus"


def _squeeze(text: str) -> str:
    return " ".join(text.split())
