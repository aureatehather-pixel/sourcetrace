"""Realistic two-evidence material for M3.

Two genuinely independent notes about the same named person, captured weeks
apart in different settings. Not "hello world": the point of M3 is that
information about one identity accumulates across separate observations, and a
fixture where both items say the same thing would not exercise that.

The second note also *qualifies* the first - Zilan walks back part of his own
earlier story. A profile that surfaces both, each with its own provenance, is
doing the job. A profile that silently merged them into one smoothed statement
would be doing the opposite.

Every `quote` below is a verbatim substring of its evidence text, so the M2
grounding check is exercised for real rather than around.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------
# Evidence A - meetup conversation
# --------------------------------------------------------------------------

EVIDENCE_A_CONTENT = (
    "Zilan told me at the Kestrel Python meetup that the fastest way to get good is to "
    "reverse-engineer a serious project instead of starting one from scratch. "
    "He said he rebuilt a small Postgres query planner over three months and that it "
    "taught him more than two years of tutorials. "
    "I asked him how he picks what to rebuild and he said he only picks things he has "
    "already used and been annoyed by."
)

A_QUOTE_MENTION = "Zilan told me at the Kestrel Python meetup"
A_QUOTE_ADVICE = "reverse-engineer a serious project instead of starting one from scratch"
A_QUOTE_PLANNER = "he rebuilt a small Postgres query planner over three months"

# --------------------------------------------------------------------------
# Evidence B - follow-up, weeks later
# --------------------------------------------------------------------------

EVIDENCE_B_CONTENT = (
    "Follow-up coffee with Zilan. He walked back the query planner story slightly - "
    "said the first six weeks were mostly reading, not writing, and that people who "
    "copy his approach usually quit during that part. "
    "He also mentioned he is now mentoring two juniors at Glyde and that the same "
    "reading-first advice is what he gives them."
)

B_QUOTE_MENTION = "Follow-up coffee with Zilan"
B_QUOTE_READING = "the first six weeks were mostly reading, not writing"
B_QUOTE_GLYDE = "he is now mentoring two juniors at Glyde"


def _mention(
    *, name: str, entity_type: str, quote: str, described_as: str | None = None
) -> dict[str, Any]:
    return {
        "kind": "entity_mention",
        "data": {"name": name, "entity_type": entity_type, "described_as": described_as},
        "grounding": {"quote": quote},
    }


def _claim(*, claim: str, claimed_by: str | None, stance: str, quote: str) -> dict[str, Any]:
    return {
        "kind": "reported_claim",
        "data": {"claim": claim, "claimed_by": claimed_by, "stance": stance},
        "grounding": {"quote": quote},
    }


def _observation(*, observation: str, subject: str | None, quote: str) -> dict[str, Any]:
    return {
        "kind": "observation",
        "data": {"observation": observation, "subject": subject},
        "grounding": {"quote": quote},
    }


# --------------------------------------------------------------------------
# Evidence C - a public profile, read rather than heard
# --------------------------------------------------------------------------
#
# This note exists to exercise the distinction the `about` role was added for.
# Nothing here was said by Zilan. Every line is something the author *noticed*,
# and the entity is the subject of each one rather than its source.
#
# It is also the adversarial case for the co-occurrence trap. The AWS sentence
# names four entities - Zilan, AWS, TVU and Kestrel Bay - and only Zilan
# is its subject. The other three are part of what was observed. A design that
# derived subjects from names appearing nearby would attach this observation to
# all four; this fixture is what proves we do not.
#
# The last observation is first-person on purpose: `subject = "I"` is honest,
# names nobody on file, and must stay unresolved rather than quietly becoming an
# entity. Author modelling is deliberately not built.

EVIDENCE_C_CONTENT = (
    "Checked Zilan's public profile today. It lists Solidity and Rust under skills, "
    "alongside project management and public speaking. "
    "The profile shows he was secretary of the TVU Robotics Society. "
    "It also shows two Open Data Basics certificates. "
    "Separately I noticed he used AWS at TVU in Kestrel Bay for the club's node. "
    "I have never shipped anything on Solidity myself."
)

C_QUOTE_MENTION = "Checked Zilan's public profile today"
C_QUOTE_SKILLS = "It lists Solidity and Rust under skills"
C_QUOTE_SECRETARY = "he was secretary of the TVU Robotics Society"
C_QUOTE_AWS = "he used AWS at TVU in Kestrel Bay for the club's node"
C_QUOTE_SELF = "I have never shipped anything on Solidity myself"


def evidence_c_payload() -> dict[str, Any]:
    """Observations about a resolved person, plus the traps that must not fire."""
    return {
        "items": [
            _mention(
                name="Zilan",
                entity_type="person",
                described_as="Public profile",
                quote=C_QUOTE_MENTION,
            ),
            _observation(
                observation="His public profile lists Solidity and Rust as skills.",
                subject="Zilan",
                quote=C_QUOTE_SKILLS,
            ),
            _observation(
                observation="His profile shows he was secretary of the TVU Robotics Society.",
                subject="Zilan",
                quote=C_QUOTE_SECRETARY,
            ),
            # Four names in one sentence, one subject.
            _observation(
                observation="He used AWS at TVU in Kestrel Bay for the club's node.",
                subject="Zilan",
                quote=C_QUOTE_AWS,
            ),
            # First person: must reach the matcher as written and stay unresolved.
            _observation(
                observation="I have never shipped anything on Solidity myself.",
                subject="I",
                quote=C_QUOTE_SELF,
            ),
        ]
    }


def evidence_a_payload() -> dict[str, Any]:
    """Extraction output for evidence A, shaped like real M2 output."""
    return {
        "items": [
            _mention(
                name="Zilan",
                entity_type="person",
                described_as="Met at the Kestrel Python meetup",
                quote=A_QUOTE_MENTION,
            ),
            _claim(
                claim=(
                    "The fastest way to get good is to reverse-engineer a serious project "
                    "rather than start one from scratch."
                ),
                claimed_by="Zilan",
                stance="recommended",
                quote=A_QUOTE_ADVICE,
            ),
            _claim(
                claim=(
                    "Rebuilding a small Postgres query planner over three months taught him "
                    "more than two years of tutorials."
                ),
                claimed_by="Zilan",
                stance="asserted",
                quote=A_QUOTE_PLANNER,
            ),
        ]
    }


def evidence_b_payload() -> dict[str, Any]:
    """Extraction output for evidence B.

    Includes a `Glyde` organisation mention, so tests can prove that resolving
    the person does not drag unrelated mentions from the same evidence into the
    person's profile.
    """
    return {
        "items": [
            _mention(
                name="Zilan",
                entity_type="person",
                described_as="Met again for coffee",
                quote=B_QUOTE_MENTION,
            ),
            _claim(
                claim=(
                    "The first six weeks of the query planner rebuild were mostly reading "
                    "rather than writing, and most people who copy the approach quit then."
                ),
                claimed_by="Zilan",
                stance="asserted",
                quote=B_QUOTE_READING,
            ),
            _mention(
                name="Glyde",
                entity_type="organization",
                described_as="Where he mentors two juniors",
                quote=B_QUOTE_GLYDE,
            ),
        ]
    }


# --------------------------------------------------------------------------
# False-merge material: two different people who share a name
# --------------------------------------------------------------------------

EVIDENCE_ALEX_ONE_CONTENT = (
    "Alex Lee from the design team said the onboarding flow tests badly with "
    "first-time users and that nobody has owned it since the last reorg."
)
ALEX_ONE_QUOTE = "Alex Lee from the design team"

EVIDENCE_ALEX_TWO_CONTENT = (
    "Different Alex Lee entirely - this one runs the payments infrastructure at a "
    "bank in Singapore and has never met the design one."
)
ALEX_TWO_QUOTE = "Different Alex Lee entirely"


def alex_payload(quote: str, described_as: str) -> dict[str, Any]:
    return {
        "items": [
            _mention(
                name="Alex Lee",
                entity_type="person",
                described_as=described_as,
                quote=quote,
            )
        ]
    }


# --------------------------------------------------------------------------
# Unnamed attribution: the real M2 observation carried forward
# --------------------------------------------------------------------------

EVIDENCE_UNNAMED_CONTENT = (
    "Engineer at DevGuild told me serious projects mattered more than LeetCode. "
    "He never gave his name and I did not ask."
)
UNNAMED_QUOTE = "serious projects mattered more than LeetCode"


def unnamed_payload() -> dict[str, Any]:
    """A claim attributed to a role, not a person. Must stay unresolved."""
    return {
        "items": [
            _claim(
                claim="Serious projects matter more than LeetCode.",
                claimed_by="Engineer at DevGuild",
                stance="asserted",
                quote=UNNAMED_QUOTE,
            )
        ]
    }
