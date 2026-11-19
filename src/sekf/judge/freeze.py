"""Pinned part-one prompt hashes for the Judge and single_judge. Owner: Sai Chetan.

SPEC.md Sprint 3 freezes prompts before the first official held-out run. These
hashes are the bytes `load_prompt` records in call metadata. A mismatch is an
error: this module does not fall back to another prompt file.
"""

from __future__ import annotations

from sekf.providers.prompts import load_prompt

# sha256 of the UTF-8 text, matching `LoadedPrompt.prompt_hash`.
# These pin the prompts after 30fb350 added the instruction-boundary lines.
FROZEN_PROMPT_HASHES: dict[str, str] = {
    "judge/v1.txt": "da2bfe3727e347098f0c57745028995eeb2ec8eda71e5f74914df5fd24b7dac3",
    "baselines/single_judge/v1.txt": (
        "df160414eb3d1e3de4c08c38e07b3828c295f4a7b979abd88ed6ce400bd3e9cb"
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
