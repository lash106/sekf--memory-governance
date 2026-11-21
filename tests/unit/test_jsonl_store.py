"""Resumable results file: crash-safe, refuses to mix experiments, retries only what failed."""

import asyncio
import json
from pathlib import Path

import pytest

from sekf.evaluation.jsonl_store import JsonlResultStore, ManifestMismatch, run_resumable

KEYS = [(f"case-{i}", system) for i in range(4) for system in ("policy-v1", "direct_storage")]


def open_store(path: Path, manifest_hash: str = "m" * 64) -> JsonlResultStore:
    return JsonlResultStore(path, run_id="run-1", manifest_hash=manifest_hash)


class Crash(BaseException):
    """Stands in for a killed process: not an ordinary Exception, so nothing catches it."""


def test_new_file_gets_a_header_and_records_results(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    store = open_store(path)
    assert store.record_ok("c1", "policy-v1", {"action": "PROMOTE"}) is True
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert lines[0]["type"] == "header" and lines[0]["manifest_hash"] == "m" * 64
    assert lines[1]["status"] == "ok" and lines[1]["result"] == {"action": "PROMOTE"}
    assert store.is_done("c1", "policy-v1") and not store.is_done("c1", "other")


def test_first_ok_result_wins_and_a_replay_writes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    store = open_store(path)
    store.record_ok("c1", "s", {"v": 1})
    size = path.stat().st_size
    assert store.record_ok("c1", "s", {"v": 2}) is False
    assert path.stat().st_size == size
    assert [r["result"] for r in store.results()] == [{"v": 1}]


def test_reopening_restores_what_is_done(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    first = open_store(path)
    first.record_ok("c1", "s", {"v": 1})
    first.record_error("c2", "s", "ProviderUnavailable", "rate limited")
    again = open_store(path)
    assert len(again) == 1 and again.is_done("c1", "s") and not again.is_done("c2", "s")
    assert again.pending([("c1", "s"), ("c2", "s")]) == [("c2", "s")]
    assert again.errors()[0]["error"]["type"] == "ProviderUnavailable"


def test_a_different_manifest_is_refused_not_mixed(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    open_store(path).record_ok("c1", "s", {})
    with pytest.raises(ManifestMismatch, match="new results file"):
        open_store(path, manifest_hash="x" * 64)
    assert len(open_store(path)) == 1  # the original run is untouched


def test_a_file_without_a_header_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    path.write_text('{"type":"result","status":"ok","case_id":"c","system":"s","result":{}}\n')
    with pytest.raises(ManifestMismatch, match="no header"):
        open_store(path)


def test_a_half_written_last_line_is_repaired_and_only_that_line_is_lost(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    store = open_store(path)
    store.record_ok("c1", "s", {"v": 1})
    store.record_ok("c2", "s", {"v": 2})
    with path.open("ab") as fh:  # the process died in the middle of writing c3
        fh.write(b'{"type":"result","status":"ok","case_id":"c3","sys')
    resumed = open_store(path)
    assert len(resumed) == 2 and not resumed.is_done("c3", "s")
    assert resumed.record_ok("c3", "s", {"v": 3}) is True  # appends cleanly after the repair
    for line in path.read_text().splitlines():
        json.loads(line)  # every remaining line is complete JSON
    assert len(open_store(path)) == 3


async def work_ok(key: tuple[str, str]) -> dict[str, str]:
    return {"action": f"{key[0]}:{key[1]}"}


def test_run_resumable_runs_everything_once_then_nothing(tmp_path: Path) -> None:
    store = open_store(tmp_path / "r.jsonl")
    first = asyncio.run(run_resumable(store, KEYS, work_ok))
    assert first == {"ran": 8, "ok": 8, "error": 0, "skipped": 0}
    calls: list[tuple[str, str]] = []

    async def spy(key: tuple[str, str]) -> dict[str, str]:
        calls.append(key)
        return {}

    second = asyncio.run(run_resumable(open_store(tmp_path / "r.jsonl"), KEYS, spy))
    assert second == {"ran": 0, "ok": 0, "error": 0, "skipped": 8} and calls == []


def test_a_crash_midway_then_resume_finishes_without_redoing_finished_work(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    started: list[tuple[str, str]] = []

    async def dies_on_fifth(key: tuple[str, str]) -> dict[str, str]:
        started.append(key)
        if len(started) == 5:
            raise Crash
        return {"k": key[0]}

    with pytest.raises(Crash):
        asyncio.run(run_resumable(open_store(path), KEYS, dies_on_fifth))
    survived = open_store(path)
    finished = len(survived)
    assert 4 <= finished < 8  # what completed before the crash is safely on disk
    assert not survived.is_done(*started[4])  # the pair that crashed is NOT marked done

    redone: list[tuple[str, str]] = []

    async def finish(key: tuple[str, str]) -> dict[str, str]:
        redone.append(key)
        return {"k": key[0]}

    counts = asyncio.run(run_resumable(open_store(path), KEYS, finish))
    assert counts == {"ran": 8 - finished, "ok": 8 - finished, "error": 0, "skipped": finished}
    done_before = {(r["case_id"], r["system"]) for r in survived.results()}
    assert set(redone).isdisjoint(done_before)  # nothing already finished was run again
    assert len(open_store(path)) == 8


def test_an_ordinary_error_is_recorded_and_the_run_continues_then_retries_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "r.jsonl"

    async def flaky(key: tuple[str, str]) -> dict[str, str]:
        if key == ("case-1", "policy-v1"):
            raise RuntimeError("HTTP 429")
        return {}

    counts = asyncio.run(run_resumable(open_store(path), KEYS, flaky))
    assert counts["ok"] == 7 and counts["error"] == 1
    store = open_store(path)
    assert store.pending(KEYS) == [("case-1", "policy-v1")]
    assert store.errors()[0]["error"] == {"type": "RuntimeError", "message": "HTTP 429"}

    retry = asyncio.run(run_resumable(store, KEYS, work_ok))
    assert retry["ran"] == 1 and retry["ok"] == 1
    final = open_store(path)
    assert len(final) == 8 and len(final.errors()) == 1  # the failure stays in the audit trail


def test_concurrent_workers_never_interleave_lines(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    keys = [(f"c{i}", "s") for i in range(40)]

    async def slow(key: tuple[str, str]) -> dict[str, str]:
        await asyncio.sleep(0.001 * (hash(key) % 5))
        return {"payload": "x" * 200}

    counts = asyncio.run(run_resumable(open_store(path), keys, slow, concurrency=8))
    assert counts["ok"] == 40
    parsed = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(parsed) == 41  # header + 40 intact lines
    assert len(open_store(path)) == 40


def test_duplicate_and_one_shot_keys_are_handled(tmp_path: Path) -> None:
    store = open_store(tmp_path / "r.jsonl")
    counts = asyncio.run(run_resumable(store, iter([("a", "s"), ("a", "s"), ("b", "s")]), work_ok))
    assert counts == {"ran": 2, "ok": 2, "error": 0, "skipped": 0}
