# RAGE Research Agent Instructions

This file is authoritative for humans and automated agents working in this repository.

## Before changing anything

Observe, do not assume:

```bash
git status --short
git branch --show-current
git remote -v
find .. -name AGENTS.md -print
```

Read every applicable nested `AGENTS.md`. Do not overwrite unrelated work. Current source, tests, scripts, and tracked repository state override stale prose or remembered behavior.

## RAGE

RAGE means **Recursive Analysis with Gated Execution**.

Every loop must be grounded in observed state. Analysis identifies evidence, constraints, contradictions, and unknowns. Execution is admitted only by an explicit gate. Evaluation records what actually happened. The next loop starts from that evidence rather than from the previous plan.

Do not treat a plan, generated patch, existing file, merge, or green build as proof that a gate was approved.

## Repository model

- `roam/research/` owns research and findings.
- `roam/design/` owns canonical designs.
- `roam/implement/` owns active implementation working copies.
- `roam/indexes/` owns project indexes and roadmaps.
- `scripts/` owns capture, lifecycle, synchronization, and validation.
- `tests/` owns regression coverage for repository mechanics.

Use existing scripts before inventing parallel workflow machinery.

## Document contract

Every substantive Org document must contain:

```org
:PROPERTIES:
:ID: stable-id
:END:
#+title:
#+description:
#+status:
#+filetags:
```

It must also contain an approval table and changelog.

Use this approval header:

```org
* Approval Table

| Approval area | Required authority | State | Evidence required | Evidence reference |
|---------------+--------------------+-------+-------------------+--------------------|
```

Allowed approval states are `PENDING`, `NOT STARTED`, `APPROVED`, `REJECTED`, `SUPERSEDED`, and `NOT APPLICABLE`.

Never fabricate approval or evidence. `NOT APPLICABLE` needs a reason. Preserve stable IDs. Duplicate IDs and unresolved `id:` links are validation failures.

Use this changelog header:

```org
* Changelog

| Date | Change | Author or actor | Evidence |
|------+--------+-----------------+----------|
```

Research must separate verified facts, inference, contradictions, and unresolved questions. For externally changing facts, prefer current primary sources and record retrieval dates.

## Implementation gates

Each immediate project subtree under `roam/implement/` may contain at most one active Org design.

Promote through `scripts/implement.py`; do not manually create competing active copies. Record terminal outcomes through `scripts/mark-design.py`. Implemented and rejected ledgers are append-only evidence.

Retries and repeated runs must be idempotent where practical. A stale or failed run must not silently become success.

## Generated and sensitive material

Do not commit `.cache/`, `_site/`, generated databases, credentials, authorization headers, private evidence, session material, or secrets.

Never hand-edit generated output. Change source or generators instead.

## Validation

Before completion run and observe:

```bash
git diff --check
git diff --stat
python3 -m py_compile scripts/*.py
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 scripts/sync.py
python3 scripts/sync.py --check
python3 scripts/validate-docs.py
```

Never claim a command passed unless it was executed and its result was observed. Never bypass a failing wrapper with a lower-level command just to obtain a green exit code.
