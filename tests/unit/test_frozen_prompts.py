"""Frozen prompts (SPEC.md section 18, Sprint 3: freeze prompts before the first test run).

Owner: Mahalakshmi (Historian, Synthesizer). Every official result records the prompt hash
(`CallMetadata.prompt_hash`), so editing a frozen prompt in place would make two runs with the
same `prompt_version` use different prompts.

To change a frozen prompt, do not edit it: add `prompts/<role>/v2.txt`, point the role's
default at v2, and add v2 here. Then re-run `dev` before any `test` run.
"""

import pytest

from sekf.providers.prompts import load_role_prompt

FROZEN = {
    ("historian", "v1"): "afb9e3e6af41327d4d6769a99f4eac7c34e38ebeb8fc42923f1d721295dbdc9c",
    ("synthesizer", "v1"): "99596d421c465d2d01544917cd9e06f8d9403f5cad4704edf39c5a7ae5335060",
}


@pytest.mark.parametrize(("role", "version"), sorted(FROZEN), ids=lambda v: str(v))
def test_frozen_prompt_is_unchanged(role: str, version: str) -> None:
    prompt = load_role_prompt(role, version)
    assert prompt.prompt_hash == FROZEN[(role, version)], (
        f"prompts/{role}/{version}.txt changed after it was frozen; "
        "add a new version instead of editing this one"
    )


def test_roles_default_to_a_frozen_prompt() -> None:
    from sekf.historian.llm import DEFAULT_PROMPT_VERSION as HISTORIAN_VERSION
    from sekf.synthesizer.llm import DEFAULT_PROMPT_VERSION as SYNTHESIZER_VERSION

    assert ("historian", HISTORIAN_VERSION) in FROZEN
    assert ("synthesizer", SYNTHESIZER_VERSION) in FROZEN
