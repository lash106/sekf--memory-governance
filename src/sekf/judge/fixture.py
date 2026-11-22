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

from sekf.contracts import JudgeAssessment, JudgeLabel, JudgeRequest

PROMPT_VERSION = "fixture-judge-v1"
MODEL_ID = "fixture/judge"
PROVIDER_ID = "fixture"

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
_CLOCK = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_MONEY = re.compile(r"\$(\d+)")
_PLACEHOLDERS = frozenset({"unknown", "unknown_malformed", "n/a", "???", "none"})
_INSTRUCTION_CUES = (
    "ignore previous",
    "ignore your rules",
    "always output",
    "system note",
    "label the claim",
    "mark all",
    "set this source",
)
_HEDGE_CUES = (
    "maybe",
    "heard",
    "not certain",
    "pretty sure",
    "someone mentioned",
    "no hour information",
    "lol",
    "??",
)
_REFUTE_CUES = ("incorrect", "is false", "denied", "no overnight", "not true")
_FALSE_OBJECTS = frozenset({"false", "no", "not required"})
_TRUE_OBJECTS = frozenset({"true", "yes", "required"})


def _clock_key(hour: int, minute: str) -> str:
    return f"{hour % 24:02d}:{minute}"


def _clocks(text: str) -> set[str]:
    return {_clock_key(int(hour), minute) for hour, minute in _CLOCK.findall(text)}


def _object_phrases(object_text: str) -> set[str]:
    """Surface forms of the claimed object, including 24-hour and 12-hour clock words."""

    raw = object_text.strip().lower()
    phrases = {raw}
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", raw)
    if match is None:
        return phrases
    hour = int(match.group(1))
    minute = match.group(2)
    phrases.add(_clock_key(hour, minute))
    if minute != "00" or not 0 <= hour <= 23:
        return phrases
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


def _malformed(request: JudgeRequest) -> bool:
    candidate = request.candidate
    if candidate.metadata.get("malformed") is True:
        return True
    if candidate.object_text.strip().lower() in _PLACEHOLDERS:
        return True
    return re.search(r"[a-z0-9]", candidate.claim_text.lower()) is None


def _ignored(content: str) -> bool:
    text = content.lower()
    if any(cue in text for cue in _INSTRUCTION_CUES):
        return True
    return any(cue in text for cue in _HEDGE_CUES)


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


def _stance(object_text: str, content: str) -> str | None:
    """Return support, refute, or None for one evidence text."""

    if _ignored(content):
        return None
    boolean = _boolean_stance(object_text, content)
    if boolean is not None:
        return boolean
    text = content.lower()
    phrases = _object_phrases(object_text)
    mentioned = any(phrase in text for phrase in phrases)
    if mentioned and any(cue in text for cue in _REFUTE_CUES):
        return "refute"
    object_clocks = _clocks(object_text)
    other_clocks = _clocks(text) - object_clocks
    if object_clocks and other_clocks and not mentioned:
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
            confidence = 0.90
        else:
            supporting = []
            refuting = []
            for item in request.evidence:
                stance = _stance(candidate.object_text, item.content)
                if stance == "support":
                    supporting.append(item.evidence_id)
                elif stance == "refute":
                    refuting.append(item.evidence_id)
            if supporting and refuting:
                label = JudgeLabel.MIXED_EVIDENCE
                missing = []
                rationale = (
                    "Some evidence states the claimed value and some evidence contradicts it."
                )
                confidence = 0.70
            elif supporting:
                label = JudgeLabel.SUPPORTED
                missing = []
                rationale = "Evidence states the claimed value and does not contradict it."
                confidence = 0.90
            elif refuting:
                label = JudgeLabel.REFUTED
                missing = []
                rationale = "Evidence contradicts the claimed value."
                confidence = 0.90
            else:
                label = JudgeLabel.INSUFFICIENT_EVIDENCE
                missing = ["No evidence states or contradicts the claimed value."]
                rationale = "No supplied evidence states or contradicts the claimed value."
                confidence = 0.55
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
