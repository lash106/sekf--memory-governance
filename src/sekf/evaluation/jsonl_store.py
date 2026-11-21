"""Resumable per-case results file (SPEC.md sections 15 and 18, Sprint 2). Owner: Shailen.

A run evaluates many (case, system) pairs, some of them against a paid model. If it dies at
case 30 it must continue at case 31, not start again, and it must never mix two different
experiments in one file. This module provides that:

- One JSON object per line, appended and fsynced as each result finishes, so a crash loses at
  most the pairs that were in flight.
- The first line is a header holding the run id and the manifest hash. Opening an existing file
  with a different manifest raises `ManifestMismatch` instead of silently combining runs.
- A pair is *done* once it has an `ok` record. `error` records are kept for the audit trail and
  the pair is retried on resume. The first `ok` record for a pair wins (like a committed
  decision: a replay never overwrites it).
- A crash can leave a half-written last line. Opening repairs that by dropping only that line.

`run_resumable` ties it together: it runs only the pending pairs through an async function,
records each outcome as it completes, turns an ordinary exception into an `error` record, and
carries on.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable, Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

Key = tuple[str, str]  # (case_id, system)
STORE_VERSION = "1"


class ManifestMismatch(Exception):
    """The results file belongs to a different experiment than the one being run."""


class JsonlResultStore:
    def __init__(self, path: Path, *, run_id: str, manifest_hash: str) -> None:
        self.path = path
        self.run_id = run_id
        self.manifest_hash = manifest_hash
        self._done: dict[Key, dict[str, Any]] = {}
        self._errors: list[dict[str, Any]] = []
        self._open()

    # -------------------------------------------------------------------------- opening

    def _open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or self.path.stat().st_size == 0:
            self._write_line(
                {
                    "type": "header",
                    "store_version": STORE_VERSION,
                    "run_id": self.run_id,
                    "manifest_hash": self.manifest_hash,
                    "created_at": _now(),
                }
            )
            return
        records, good_bytes = self._read_valid_prefix()
        if good_bytes < self.path.stat().st_size:
            with self.path.open("r+b") as fh:  # drop the half-written tail only
                fh.truncate(good_bytes)
        header = records[0] if records and records[0].get("type") == "header" else None
        if header is None:
            raise ManifestMismatch(f"{self.path} has no header; refusing to append to it")
        if header.get("manifest_hash") != self.manifest_hash:
            raise ManifestMismatch(
                f"{self.path} was written for manifest {header.get('manifest_hash')!s:.12} "
                f"but this run is {self.manifest_hash[:12]}; use a new results file or "
                "restore the code, data and prompts of the original run"
            )
        for record in records[1:]:
            if record.get("type") != "result":
                continue
            key = (record["case_id"], record["system"])
            if record["status"] == "ok":
                self._done.setdefault(key, record)  # first ok wins
            else:
                self._errors.append(record)

    def _read_valid_prefix(self) -> tuple[list[dict[str, Any]], int]:
        """Parse lines until the first one that is not complete JSON; return them and their size."""
        records: list[dict[str, Any]] = []
        good = 0
        with self.path.open("rb") as fh:
            for raw in fh:
                if not raw.endswith(b"\n"):
                    break  # a crash cut this line short
                try:
                    records.append(json.loads(raw))
                except json.JSONDecodeError:
                    break
                good += len(raw)
        return records, good

    # --------------------------------------------------------------------------- writing

    def _write_line(self, obj: dict[str, Any]) -> None:
        line = json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n"
        with self.path.open("ab") as fh:
            fh.write(line.encode("utf-8"))
            fh.flush()
            os.fsync(fh.fileno())

    def record_ok(self, case_id: str, system: str, result: dict[str, Any]) -> bool:
        """Store a finished result. Returns False (and writes nothing) if the pair was done."""
        key = (case_id, system)
        if key in self._done:
            return False
        record = {
            "type": "result",
            "status": "ok",
            "case_id": case_id,
            "system": system,
            "result": result,
            "recorded_at": _now(),
        }
        self._write_line(record)
        self._done[key] = record
        return True

    def record_error(self, case_id: str, system: str, error_type: str, message: str) -> None:
        """Keep a failure for the audit trail. The pair stays pending and is retried on resume."""
        record = {
            "type": "result",
            "status": "error",
            "case_id": case_id,
            "system": system,
            "error": {"type": error_type, "message": message},
            "recorded_at": _now(),
        }
        self._write_line(record)
        self._errors.append(record)

    # ---------------------------------------------------------------------------- reading

    def is_done(self, case_id: str, system: str) -> bool:
        return (case_id, system) in self._done

    def pending(self, keys: Iterable[Key]) -> list[Key]:
        return [k for k in keys if k not in self._done]

    def results(self) -> Iterator[dict[str, Any]]:
        """Finished results, one per pair, in the order they were first recorded."""
        return iter(self._done.values())

    def errors(self) -> list[dict[str, Any]]:
        return list(self._errors)

    def __len__(self) -> int:
        return len(self._done)


def _now() -> str:
    return datetime.now(UTC).isoformat()


async def run_resumable(
    store: JsonlResultStore,
    keys: Iterable[Key],
    work: Callable[[Key], Awaitable[dict[str, Any]]],
    *,
    concurrency: int = 1,
) -> dict[str, int]:
    """Run `work` for every pair that is not done yet; record each outcome as it finishes.

    An ordinary `Exception` becomes an `error` record and the run continues. Anything else (for
    example Ctrl-C or a crash) propagates immediately, and everything already recorded is safe
    on disk, so calling this again resumes. Returns counts of what happened in this call.
    """
    wanted = list(
        dict.fromkeys(keys)
    )  # `keys` may be a one-shot iterator; keep order, drop repeats
    todo = store.pending(wanted)
    counts = {"ran": 0, "ok": 0, "error": 0, "skipped": len(wanted) - len(todo)}
    gate = asyncio.Semaphore(max(1, concurrency))
    write_lock = asyncio.Lock()

    async def one(key: Key) -> None:
        async with gate:
            try:
                result = await work(key)
            except Exception as exc:
                async with write_lock:
                    store.record_error(key[0], key[1], type(exc).__name__, str(exc))
                counts["error"] += 1
            else:
                async with write_lock:
                    store.record_ok(key[0], key[1], result)
                counts["ok"] += 1
            counts["ran"] += 1

    if not todo:
        return counts
    tasks = [asyncio.create_task(one(k)) for k in todo]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in pending:  # a crash stops the run: do not keep starting new work
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            if (crash := task.exception()) is not None:
                raise crash
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return counts
