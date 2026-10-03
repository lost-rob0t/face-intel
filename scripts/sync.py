#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from _roamlib import TREE_NAMES, implementation_slot_problems, project_root, visible_dirs


def expected_missing(roam):
    problems: list[str] = []
    for tree in TREE_NAMES:
        if not (roam / tree).is_dir():
            problems.append(f"missing directory: roam/{tree}")
    for ledger in (".implemented", ".rejected"):
        if not (roam / ledger).exists():
            problems.append(f"missing ledger: roam/{ledger}")

    rels = set()
    for tree in TREE_NAMES:
        rels.update(visible_dirs(roam / tree))
    for rel in sorted(rels):
        for tree in TREE_NAMES:
            path = roam / tree / rel
            if not path.is_dir():
                problems.append(f"missing mirrored directory: {path.relative_to(project_root())}")
    problems.extend(implementation_slot_problems(roam))
    return problems


def synchronize(roam):
    for tree in TREE_NAMES:
        (roam / tree).mkdir(parents=True, exist_ok=True)
    for ledger in (".implemented", ".rejected"):
        (roam / ledger).touch(exist_ok=True)

    rels = set()
    for tree in TREE_NAMES:
        rels.update(visible_dirs(roam / tree))
    for rel in rels:
        for tree in TREE_NAMES:
            (roam / tree / rel).mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    root = project_root()
    roam = root / "roam"
    if args.check:
        problems = expected_missing(roam)
        if problems:
            print("\n".join(problems), file=sys.stderr)
            return 1
        print("sync check: ok")
        return 0

    synchronize(roam)
    problems = expected_missing(roam)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print("sync: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
