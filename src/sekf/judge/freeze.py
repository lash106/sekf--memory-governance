"""Pinned part-one prompt hashes for the Judge and single_judge. Owner: Sai Chetan.

SPEC.md Sprint 3 freezes prompts before the first official held-out run. These
hashes are the bytes `load_prompt` records in call metadata. A mismatch is an
error: this module does not fall back to another prompt file.
"""

from __future__ import annotations

from sekf.providers.prompts import load_prompt

# sha256 of the UTF-8 text, matching `LoadedPrompt.prompt_hash`.
FROZEN_PROMPT_HASHES: dict[str, str] = {
    "judge/v1.txt": "6695b6071e8c004aaeaf5cc7fa5311b55f3f2229e888cee174b1990b8e5270d1",
    "baselines/single_judge/v1.txt": (
        "39e28d9870a37df8b9dba4ddb2023560059336b41e2b8d227fd154c0d68495f4"
    ),
}


def frozen_prompt_mismatches() -> list[str]:
    """Return human-readable mismatches. An empty list means the files still match."""

    mismatches: list[str] = []
    for relative_path, expected in FROZEN_PROMPT_HASHES.items():
        loaded = load_prompt(relative_path)
        if loaded.prompt_hash != expected:
            mismatches.append(f"{relative_path}: on disk {loaded.prompt_hash} frozen {expected}")
    return mismatches


def assert_prompts_frozen() -> None:
    """Raise ValueError when a pinned prompt file no longer matches its hash."""

    mismatches = frozen_prompt_mismatches()
    if mismatches:
        raise ValueError("frozen prompt mismatch: " + "; ".join(mismatches))
