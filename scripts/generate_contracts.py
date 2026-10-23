"""Generate contracts/v1/*.json from the Pydantic contracts (`make contracts`).

`generated_schemas()` is also used by tests/contract to fail when committed files are stale.
"""

import inspect
import json
from pathlib import Path

from sekf.contracts import models

OUT_DIR = Path(__file__).resolve().parent.parent / "contracts" / "v1"


def generated_schemas() -> dict[str, str]:
    """Return {filename: json text} for every concrete contract model."""
    schemas: dict[str, str] = {}
    for name, cls in inspect.getmembers(models, inspect.isclass):
        if cls is models.Contract or not issubclass(cls, models.Contract):
            continue
        if cls.__module__ != models.__name__:
            continue
        schemas[f"{name}.json"] = (
            json.dumps(cls.model_json_schema(), indent=2, sort_keys=True) + "\n"
        )
    return schemas


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wanted = generated_schemas()
    for stale in OUT_DIR.glob("*.json"):
        if stale.name not in wanted:
            stale.unlink()
    for filename, text in wanted.items():
        (OUT_DIR / filename).write_text(text)
    print(f"wrote {len(wanted)} schemas to {OUT_DIR}")


if __name__ == "__main__":
    main()
