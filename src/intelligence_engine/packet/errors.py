"""Packet error model. Short stable `error_code`s, as everywhere else here."""

from __future__ import annotations

from intelligence_engine.core.errors import ApplicationError


class PacketError(ApplicationError):
    """Base for anything the packet layer raises deliberately."""

    error_code = "packet_error"


class InvalidCitation(PacketError):
    """A cited item id is not in the packet that was supplied.

    Rejection is the whole behaviour. There is no repair path, no nearest-match,
    and no attempt to infer what was meant: the failure this exists to catch is a
    plausible-looking id that points at nothing, and every one of those
    strategies would turn it back into a citation a reader would trust.
    """

    error_code = "invalid_citation"


class CorpusExceedsPacketLimit(PacketError):
    """The stored corpus no longer fits in one packet, so validation refuses.

    Unique binding is a **packet-wide** property: a fragment is accepted only if
    it occurs in no other item's grounding quote *in the packet being checked*.
    Compute that over a prefix of the corpus and the check silently weakens —
    a fragment colliding only with an item past the cut is reported unique, which
    is failure 3 in `citations.py` reintroduced by the validator itself rather
    than by the model.

    So when the whole corpus cannot be assembled, `pie validate` declines to
    return a verdict instead of returning a weaker one under the same name. The
    operator can still validate against an explicit `--limit`, which makes the
    narrowed scope a stated choice that the audit output then names.
    """

    error_code = "corpus_exceeds_packet_limit"
