# Reproducing SEKF

How a reviewer gets from a fresh clone to the same results. Written against `SPEC.md`
section 18 (Sprint 4 acceptance). Steps marked **(pending)** depend on work that has not
merged yet; they are listed so the guide shows the full path, and each is removed from
"pending" when it lands.

## 1. Get the code at a fixed point

```bash
git clone https://github.com/lash106/sekf--memory-governance.git
cd sekf--memory-governance
git checkout v0.1.0            # (pending) the release tag; until it exists, use the commit in the manifest
```

## 2. Set up (Python 3.12)

```bash
make setup                     # creates .venv, installs the exact versions in constraints.txt
make lint typecheck test       # everything runs offline with fake roles; no key needed
```

`constraints.txt` pins every dependency. To change a dependency, edit `pyproject.toml`, then
`make setup && make lock` and commit the new `constraints.txt`.

## 3. Start the application

Local:

```bash
make run                       # http://localhost:8000   (interactive docs at /docs)
```

Or Docker (same pinned dependencies, runs as a non-root user, data in a volume):

```bash
docker build -t sekf .
docker run --rm -p 8000:8000 -v sekf-data:/app/data sekf
```

Defaults need no key and no network: SQLite store, rule-based Historian and Synthesizer, and a
fake Judge. See the README for the `SEKF_*` settings. An unknown value stops the app with an
error instead of silently picking another.

## 4. Submit a claim and inspect the result

Send the worked example from `SPEC.md` section 4 and read the Historian, Judge, Synthesizer and
policy outputs, the rules fired, and the memory change in the response:

```bash
python3 -c "import json;r=json.load(open('tests/fixtures/lab_closing_revise.json'))['request'];r['mode']='commit';print(json.dumps(r))" > /tmp/req.json
curl -s localhost:8000/v1/decide -H 'content-type: application/json' -d @/tmp/req.json
```

Repeating the call returns `"replayed": true`: the first committed decision is returned and
nothing runs twice. To see the stored state and audit trail directly:

```bash
sqlite3 data/sekf.sqlite3 "select substr(claim_id,1,8), status, object_text, version from claims order by version"
```

With Docker the database is inside the volume: `docker exec <container> python -c "..."` or
mount the volume on the host.

## 5. Re-run the experiment and reproduce the summary **(pending)**

Needs the evaluation runner (merging from the `sai-chetan/review-fixes` branch) wired to the
run manifest and the resumable results file already in `src/sekf/evaluation/`:

- `manifest.py` records the git commit (and whether the tree was dirty), the dataset hash, every
  prompt file's hash, the model, the policy and thresholds, the run configuration, and the
  relevant `SEKF_*` settings. No API key is ever written.
- `jsonl_store.py` writes one line per (case, system) as it finishes, so a crashed run resumes
  where it stopped, and refuses to append to a file that belongs to a different manifest.

The final instructions here will be the exact command that runs the pilot on `datasets/dev/`
and the command that rebuilds the summary table from the saved results file alone.

## 6. Labeling and agreement

Gold labels follow `datasets/ANNOTATION_GUIDE.md`. Each frozen test case is labeled
independently by two people, then:

```bash
.venv/bin/python scripts/check_agreement.py <annotator-A-folder> <annotator-B-folder>
```

prints percent agreement and Cohen's kappa per field and exits non-zero if a case is missing,
the inputs differ, or kappa is below 0.6. Resolve disagreements, fix the guide if a rule was
unclear, and re-run before freezing `datasets/test/`.

## What is deliberately not in the image

`.env`, tests, notebooks, experiment outputs, and anything under `data/`. Pass settings with
`-e KEY=value` or `--env-file`.
