#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from _roamlib import ensure_roam, implementation_slot_problems, project_root, validate_org_headers


def status(roam: Path, root: Path) -> int:
    active = sorted((roam / "implement").rglob("*.org"))
    if not active:
        print("no active implementations")
        return 0
    for path in active:
        print(path.relative_to(root))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("design", nargs="?")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    root = project_root()
    roam = ensure_roam(root)

    if args.status:
        return status(roam, root)
    if not args.design:
        parser.error("design is required unless --status is used")

    source = Path(args.design)
    if not source.is_absolute():
        source = (root / source).resolve()
    design_root = (roam / "design").resolve()
    try:
        rel = source.relative_to(design_root)
    except ValueError as exc:
        raise SystemExit(f"design must be under {design_root}") from exc
    if not source.is_file():
        raise SystemExit(f"missing design: {source}")

    validate_org_headers(source)
    if len(rel.parts) < 2:
        raise SystemExit("design must be inside a project directory")

    project = rel.parts[0]
    active_dir = roam / "implement" / project
    active_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(active_dir.rglob("*.org"))
    if existing:
        rendered = ", ".join(str(path.relative_to(root)) for path in existing)
        raise SystemExit(f"implementation slot for {project} is occupied: {rendered}")

    destination = roam / "implement" / rel
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)

    problems = implementation_slot_problems(roam)
    if problems:
        destination.unlink(missing_ok=True)
        raise SystemExit("\n".join(problems))

    print(destination.relative_to(root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
