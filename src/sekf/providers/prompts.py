"""Load versioned prompt templates from the repo `prompts/` tree.

Owner: Sai Chetan. Layout is role-oriented so later versions and roles can be added
without renaming call sites:

    prompts/<role>/<version>.txt
    prompts/baselines/<baseline_name>/<version>.txt

Templates use `{named}` placeholders only. JSON payloads must be substituted with
`render_prompt` (not `str.format`) so braces in the JSON stay intact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")


@dataclass(frozen=True)
class LoadedPrompt:
    """Prompt file contents plus the hash recorded in `CallMetadata.prompt_hash`."""

    relative_path: str
    version: str
    text: str
    prompt_hash: str


def prompts_dir() -> Path:
    """Walk up from this file until a `prompts/` directory is found."""

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "prompts"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("could not locate the repository prompts/ directory")


def prompt_version_for(relative_path: str) -> str:
    """Stable audit id for a relative prompt path (`judge/v1.txt` -> `judge/v1`)."""

    return Path(relative_path).with_suffix("").as_posix()


def load_prompt(relative_path: str) -> LoadedPrompt:
    """Read `prompts/<relative_path>` and return text plus sha256 of the UTF-8 bytes.

    `relative_path` may include subdirectories (for example `judge/v1.txt`).
    """

    path = prompts_dir().joinpath(*Path(relative_path).parts)
    if not path.is_file():
        raise FileNotFoundError(f"prompt not found: {relative_path}")
    text = path.read_text(encoding="utf-8")
    return LoadedPrompt(
        relative_path=Path(relative_path).as_posix(),
        version=prompt_version_for(relative_path),
        text=text,
        prompt_hash=sha256(text.encode("utf-8")).hexdigest(),
    )


def load_role_prompt(role: str, version: str = "v1") -> LoadedPrompt:
    """Load `prompts/<role>/<version>.txt` (council roles such as `judge`)."""

    return load_prompt(f"{role}/{version}.txt")


def load_baseline_prompt(baseline: str, version: str = "v1") -> LoadedPrompt:
    """Load `prompts/baselines/<baseline>/<version>.txt`."""

    return load_prompt(f"baselines/{baseline}/{version}.txt")


def render_prompt(template: str, **values: str) -> str:
    """Replace `{name}` placeholders. Missing or unused names are errors, not silent fallbacks."""

    needed = set(_PLACEHOLDER.findall(template))
    provided = set(values)
    missing = needed - provided
    extra = provided - needed
    if missing:
        raise KeyError(f"prompt is missing values for: {sorted(missing)}")
    if extra:
        raise KeyError(f"unused prompt values: {sorted(extra)}")
    rendered = template
    for name, value in values.items():
        rendered = rendered.replace("{" + name + "}", value)
    return rendered
