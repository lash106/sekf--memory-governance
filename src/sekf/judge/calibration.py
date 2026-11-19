"""Temperature scaling for a Judge confidence. Deterministic. No LLM.

`JudgeAssessment` stores one confidence for the chosen label. Scaling treats that
confidence as P(correct) and picks a temperature that lowers the log loss on a
cached dev set. It does not change policy thresholds by itself.
"""

from __future__ import annotations

import math


def _clip(probability: float) -> float:
    return min(1.0 - 1e-6, max(1e-6, probability))


def apply_temperature(confidence: float, temperature: float) -> float:
    """Return the confidence after dividing its logit by `temperature`."""

    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if confidence < 0 or confidence > 1:
        raise ValueError("confidence must be between 0 and 1")
    clipped = _clip(confidence)
    logit = math.log(clipped / (1.0 - clipped))
    scaled = logit / temperature
    return 1.0 / (1.0 + math.exp(-scaled))


def _nll(confidences: list[float], correct: list[bool], temperature: float) -> float:
    total = 0.0
    for confidence, ok in zip(confidences, correct, strict=True):
        probability = _clip(apply_temperature(confidence, temperature))
        total += -(math.log(probability) if ok else math.log(1.0 - probability))
    return total / len(confidences)


def fit_temperature(confidences: list[float], correct: list[bool]) -> float:
    """Grid-search a temperature in (0.05, 5]. Empty input returns 1."""

    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length")
    if not confidences:
        return 1.0
    best_t = 1.0
    best_loss = _nll(confidences, correct, best_t)
    step = 0.05
    temperature = 0.05
    while temperature <= 5.0 + 1e-9:
        loss = _nll(confidences, correct, temperature)
        if loss < best_loss:
            best_loss = loss
            best_t = temperature
        temperature += step
    return best_t
