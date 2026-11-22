"""Lexical fixture Judge for offline scoring. Owner: Sai Chetan.

This is not the CI fake and not a live model. `FakeJudge` stays the offline stand-in:
it only emits SUPPORTED or INSUFFICIENT_EVIDENCE. This fixture can also emit
REFUTED, MIXED_EVIDENCE, and INVALID_INPUT so a dev run can score those gold
labels. It reads the candidate and the evidence only. It does not read gold
labels, memory, Historian output, or the policy.

`prompt_hash` is sha256 of the version string, the same convention as the
deterministic historian. It is not a hash of a prompt file.
"""

from __future__ import annotations

import re
from hashlib import sha256
from uuid import NAMESPACE_URL, UUID, uuid5

from sekf.contracts import EvidenceRecord, JudgeAssessment, JudgeLabel, JudgeRequest

PROMPT_VERSION = "fixture-judge-v2"
MODEL_ID = "fixture/judge"
PROVIDER_ID = "fixture"

# Evidence below this reliability does not create MIXED_EVIDENCE against stronger support
# (annotation guide: unequal sources; Judge still may cite the weak text as ignored).
_WEAK_RELIABILITY = 0.6

_HOUR_WORDS = (
    "twelve",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
)
_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_CLOCK = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_RANGE_CLOCK = re.compile(r"\b(\d{1,2}):(\d{2})\s*(?:-|to)\s*(\d{1,2}):(\d{2})\b", re.I)
# Skip fragments of HH:MM (e.g. do not treat "00-16" inside "10:00-16:00" as a range).
_INFORMAL_RANGE = re.compile(
    r"(?<![:\d])\b(\d{1,2})\s*(?:-|to)\s*(\d{1,2})\b(?![:\d])",
    re.I,
)
_ISO_DATE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
_MONTH_DAY = re.compile(
    r"\b(" + "|".join(_MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(20\d{2}))?\b",
    re.I,
)
_MONEY = re.compile(r"\$(\d+)")
_TOKEN = re.compile(r"[a-z0-9]+")
_PLACEHOLDERS = frozenset({"unknown", "unknown_malformed", "n/a", "???", "none"})
_STOP = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "for",
        "on",
        "in",
        "at",
        "by",
        "is",
        "are",
        "from",
        "with",
        "rm",
        "room",
    }
)
_INSTRUCTION_CUES = (
    "ignore previous",
    "ignore your rules",
    "always output",
    "system note",
    "label the claim",
    "mark all",
    "set this source",
)
# Soft social noise alone does not discard a clear asserted clock ("21:00 lol").
_HARD_HEDGE_CUES = (
    "maybe",
    "heard",
    "not certain",
    "pretty sure",
    "someone mentioned",
    "no hour information",
)
_SOFT_NOISE = ("lol", "??")
_REFUTE_CUES = ("incorrect", "is false", "denied", "no overnight", "not true")
_FALSE_OBJECTS = frozenset({"false", "no", "not required"})
_TRUE_OBJECTS = frozenset({"true", "yes", "required"})


def _clock_key(hour: int, minute: str) -> str:
    return f"{hour % 24:02d}:{minute}"


def _clocks(text: str) -> set[str]:
    found = {_clock_key(int(hour), minute) for hour, minute in _CLOCK.findall(text)}
    for start_h, start_m, end_h, end_m in _RANGE_CLOCK.findall(text):
        found.add(_clock_key(int(start_h), start_m))
        found.add(_clock_key(int(end_h), end_m))
    for start, end in _INFORMAL_RANGE.findall(text):
        start_h, end_h = int(start), int(end)
        # "10 to 4" means 10:00-16:00 when end is a small afternoon hour.
        if end_h <= 12 and start_h <= 12 and end_h < start_h:
            end_h += 12
        found.add(_clock_key(start_h, "00"))
        found.add(_clock_key(end_h, "00"))
    return found


def _dates(text: str) -> set[str]:
    """Normalize calendar days to YYYY-MM-DD when year is known, else --MM-DD."""

    found: set[str] = set()
    for year, month, day in _ISO_DATE.findall(text):
        found.add(f"{year}-{month}-{day}")
        found.add(f"--{month}-{day}")
    for month_name, day, year in _MONTH_DAY.findall(text):
        month = _MONTHS[month_name.lower()]
        day_i = int(day)
        if year:
            found.add(f"{year}-{month:02d}-{day_i:02d}")
        found.add(f"--{month:02d}-{day_i:02d}")
    return found


def _tokens(text: str) -> set[str]:
    return {tok for tok in _TOKEN.findall(text.lower()) if tok not in _STOP and len(tok) > 1}


def _object_phrases(object_text: str) -> set[str]:
    """Surface forms of the claimed object, including clocks, ranges, and 12-hour words."""

    raw = object_text.strip().lower()
    phrases = {raw, raw.replace(" to ", "-"), raw.replace("-", " to ")}
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", raw)
    if match is not None:
        hour = int(match.group(1))
        minute = match.group(2)
        phrases.add(_clock_key(hour, minute))
        if minute == "00" and 0 <= hour <= 23:
            hour12 = hour % 12 or 12
            suffix = "pm" if hour >= 12 else "am"
            word = "twelve" if hour12 == 12 else _HOUR_WORDS[hour12]
            phrases.update(
                {
                    f"{hour12} {suffix}",
                    f"{hour12}{suffix}",
                    f"{hour12}:00 {suffix}",
                    f"{word} {suffix}",
                    f"{word}{suffix}",
                }
            )
        return phrases
    range_match = re.fullmatch(r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})", raw)
    if range_match is not None:
        a, b, c, d = range_match.groups()
        start_h, end_h = int(a), int(c)
        phrases.add(f"{_clock_key(start_h, b)}-{_clock_key(end_h, d)}")
        phrases.add(f"{start_h} to {end_h}")
        phrases.add(f"{start_h}-{end_h}")
        # Social shorthand: "10:00-16:00" often appears as "10 to 4".
        if b == "00" and d == "00" and end_h > 12:
            end12 = end_h - 12
            phrases.add(f"{start_h} to {end12}")
            phrases.add(f"{start_h}-{end12}")
    return phrases


def _malformed(request: JudgeRequest) -> bool:
    candidate = request.candidate
    if candidate.metadata.get("malformed") is True:
        return True
    if candidate.object_text.strip().lower() in _PLACEHOLDERS:
        return True
    return re.search(r"[a-z0-9]", candidate.claim_text.lower()) is None


def _ignored(content: str, object_text: str) -> bool:
    """Drop injection and hard hedges. Keep soft noise when a matching clock is asserted."""

    text = content.lower()
    if any(cue in text for cue in _INSTRUCTION_CUES):
        return True
    if any(cue in text for cue in _HARD_HEDGE_CUES):
        return True
    if any(cue in text for cue in _SOFT_NOISE):
        object_clocks = _clocks(object_text)
        if object_clocks and object_clocks <= _clocks(text):
            return False
        if not object_clocks:
            return True
    return False


def _boolean_stance(object_text: str, content: str) -> str | None:
    obj = object_text.strip().lower()
    text = content.lower()
    asserts_required = any(cue in text for cue in ("must be worn", "must be", "required"))
    asserts_absent = "no longer required" in text or "not required" in text
    if obj in _FALSE_OBJECTS:
        if asserts_required and not asserts_absent:
            return "refute"
        if asserts_absent:
            return "support"
    if obj in _TRUE_OBJECTS:
        if asserts_absent:
            return "refute"
        if asserts_required:
            return "support"
    return None


def _token_overlap(object_text: str, content: str) -> bool:
    left = _tokens(object_text)
    right = _tokens(content)
    if not left or not right:
        return False
    # Require most object tokens to appear (room numbers, hall names).
    return len(left & right) >= max(1, (len(left) + 1) // 2)


def _stance(object_text: str, content: str) -> str | None:
    """Return support, refute, or None for one evidence text."""

    if _ignored(content, object_text):
        return None
    boolean = _boolean_stance(object_text, content)
    if boolean is not None:
        return boolean
    text = content.lower()
    phrases = _object_phrases(object_text)
    mentioned = any(phrase in text for phrase in phrases) or _token_overlap(object_text, content)
    object_clocks = _clocks(object_text)
    text_clocks = _clocks(text)
    if object_clocks and object_clocks <= text_clocks:
        mentioned = True
    object_dates = _dates(object_text)
    text_dates = _dates(text)
    if object_dates and (object_dates & text_dates):
        mentioned = True
    if mentioned and any(cue in text for cue in _REFUTE_CUES):
        return "refute"
    other_clocks = text_clocks - object_clocks
    if object_clocks and other_clocks and not (object_clocks <= text_clocks):
        return "refute"
    other_dates = text_dates - object_dates
    # Same-month competing day for a dated claim is a conflict.
    if object_dates and other_dates and not (object_dates & text_dates):
        return "refute"
    object_money = set(_MONEY.findall(object_text))
    other_money = set(_MONEY.findall(content)) - object_money
    if not object_money and object_text.strip().isdigit():
        object_money = {object_text.strip()}
        other_money = set(_MONEY.findall(content)) - object_money
    if object_money and other_money and not mentioned:
        return "refute"
    if "24" in object_text and "no overnight" in text:
        return "refute"
    if mentioned:
        return "support"
    return None


def _aggregate(
    evidence: list[EvidenceRecord],
    supporting: list[UUID],
    refuting: list[UUID],
) -> tuple[JudgeLabel, float, str, list[str], list[UUID], list[UUID]]:
    """Apply reliability: weak refutes do not overturn strong support."""

    by_id = {item.evidence_id: item for item in evidence}
    strong_support = [eid for eid in supporting if by_id[eid].reliability >= _WEAK_RELIABILITY]
    strong_refute = [eid for eid in refuting if by_id[eid].reliability >= _WEAK_RELIABILITY]
    weak_support = [eid for eid in supporting if by_id[eid].reliability < _WEAK_RELIABILITY]
    weak_refute = [eid for eid in refuting if by_id[eid].reliability < _WEAK_RELIABILITY]

    if strong_support and strong_refute:
        return (
            JudgeLabel.MIXED_EVIDENCE,
            0.92,
            "Some evidence states the claimed value and some evidence contradicts it.",
            [],
            supporting,
            refuting,
        )
    if strong_support:
        # Drop weak contradicting citations from the Judge output for unequal sources.
        return (
            JudgeLabel.SUPPORTED,
            0.95,
            "Reliable evidence states the claimed value; weaker conflicting text is ignored.",
            [],
            strong_support + weak_support,
            [],
        )
    if strong_refute and not supporting:
        return (
            JudgeLabel.REFUTED,
            0.95,
            "Evidence contradicts the claimed value.",
            [],
            [],
            strong_refute + weak_refute,
        )
    if strong_refute and weak_support and not strong_support:
        return (
            JudgeLabel.REFUTED,
            0.92,
            "Reliable evidence contradicts the claimed value.",
            [],
            [],
            strong_refute,
        )
    if weak_support and weak_refute:
        return (
            JudgeLabel.MIXED_EVIDENCE,
            0.90,
            "Weak evidence both supports and contradicts the claimed value.",
            [],
            supporting,
            refuting,
        )
    if weak_support:
        return (
            JudgeLabel.SUPPORTED,
            0.92,
            "Evidence states the claimed value and does not contradict it.",
            [],
            supporting,
            [],
        )
    if weak_refute:
        return (
            JudgeLabel.REFUTED,
            0.92,
            "Evidence contradicts the claimed value.",
            [],
            [],
            refuting,
        )
    return (
        JudgeLabel.INSUFFICIENT_EVIDENCE,
        0.90,
        "No supplied evidence states or contradicts the claimed value.",
        ["No evidence states or contradicts the claimed value."],
        [],
        [],
    )


class FixtureJudge:
    """Evidence-only lexical Judge. Same request fields as `FakeJudge`, more labels."""

    async def evaluate(self, request: JudgeRequest) -> JudgeAssessment:
        """Label the claim from its own text and the evidence texts."""

        candidate = request.candidate
        if _malformed(request):
            label = JudgeLabel.INVALID_INPUT
            supporting: list[UUID] = []
            refuting: list[UUID] = []
            missing = ["The candidate claim is empty or malformed."]
            rationale = "The candidate is not a well-formed claim."
            confidence = 0.95
        else:
            supporting = []
            refuting = []
            for item in request.evidence:
                stance = _stance(candidate.object_text, item.content)
                if stance == "support":
                    supporting.append(item.evidence_id)
                elif stance == "refute":
                    refuting.append(item.evidence_id)
            label, confidence, rationale, missing, supporting, refuting = _aggregate(
                request.evidence, supporting, refuting
            )
        digest = sha256(PROMPT_VERSION.encode()).hexdigest()
        return JudgeAssessment(
            assessment_id=uuid5(
                NAMESPACE_URL,
                "fixture-judge:"
                + ":".join(
                    [str(candidate.claim_id)]
                    + [str(item.evidence_id) + item.content_hash for item in request.evidence]
                ),
            ),
            candidate_claim_id=candidate.claim_id,
            label=label,
            confidence=confidence,
            supporting_evidence_ids=supporting,
            refuting_evidence_ids=refuting,
            missing_information=missing,
            rationale=rationale,
            model=MODEL_ID,
            provider=PROVIDER_ID,
            prompt_version=PROMPT_VERSION,
            prompt_hash=digest,
            temperature=request.run_config.temperature,
            seed=request.run_config.seed,
            llm_calls=0,
        )
