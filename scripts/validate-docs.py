#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from _roamlib import implementation_slot_problems, project_root

TREE_NAMES = ("research", "design", "implement", "indexes")
ALLOWED_APPROVAL_STATES = {
    "PENDING",
    "NOT STARTED",
    "APPROVED",
    "REJECTED",
    "SUPERSEDED",
    "NOT APPLICABLE",
}
REQUIRED_HEADERS = ("title", "description", "status", "filetags")
RESEARCH_SECTIONS = (
    "Verified Facts",
    "Inference",
    "Contradictions",
    "Unresolved Questions",
)


def substantive_org_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for tree in TREE_NAMES:
        base = root / "roam" / tree
        if base.exists():
            files.extend(path for path in base.rglob("*.org") if path.is_file())
    return sorted(set(files))


def metadata(text: str, key: str) -> str | None:
    match = re.search(rf"(?im)^\#\+{re.escape(key)}:\s*(\S.*)$", text)
    return match.group(1).strip() if match else None


def org_id(text: str) -> str | None:
    block = re.search(r"(?ms)^:PROPERTIES:\s*$.*?^:END:\s*$", text)
    if not block:
        return None
    match = re.search(r"(?im)^:ID:\s*(\S+)\s*$", block.group(0))
    return match.group(1) if match else None


def approval_states(text: str) -> list[str]:
    heading = re.search(r"(?im)^\* Approval Table\s*$", text)
    if not heading:
        return []
    tail = text[heading.end():]
    next_heading = re.search(r"(?m)^\*\s+", tail)
    block = tail[: next_heading.start()] if next_heading else tail
    states: list[str] = []
    for line in block.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        if cells[0] in {"Approval area", "---------------"} or set(cells[0]) <= {"-", "+"}:
            continue
        states.append(cells[2])
    return states


def id_links(text: str) -> set[str]:
    return set(re.findall(r"\[\[id:([^]\s]+)", text))


def tracked_generated_paths(root: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    bad = []
    for line in result.stdout.splitlines():
        if line == ".cache" or line.startswith(".cache/") or line == "_site" or line.startswith("_site/"):
            bad.append(line)
    return bad


def validate_file(path: Path, root: Path) -> tuple[list[str], str | None, set[str]]:
    text = path.read_text(encoding="utf-8")
    rel = path.relative_to(root)
    errors: list[str] = []

    for key in REQUIRED_HEADERS:
        if metadata(text, key) is None:
            errors.append(f"{rel}: missing #+{key}")

    identifier = org_id(text)
    if not identifier:
        errors.append(f"{rel}: missing stable :ID:")

    if not re.search(r"(?im)^\* Approval Table\s*$", text):
        errors.append(f"{rel}: missing Approval Table")
    else:
        states = approval_states(text)
        if not states:
            errors.append(f"{rel}: approval table has no data rows")
        for state in states:
            if state not in ALLOWED_APPROVAL_STATES:
                errors.append(f"{rel}: invalid approval state {state!r}")

    if not re.search(r"(?im)^\* Changelog\s*$", text):
        errors.append(f"{rel}: missing Changelog")

    try:
        tree = rel.parts[1]
    except IndexError:
        tree = ""
    if tree == "research":
        for section in RESEARCH_SECTIONS:
            if not re.search(rf"(?im)^\* {re.escape(section)}\s*$", text):
                errors.append(f"{rel}: missing research section {section!r}")

    return errors, identifier, id_links(text)


def main() -> int:
    root = project_root()
    roam = root / "roam"
    errors: list[str] = []
    ids: dict[str, Path] = {}
    links: list[tuple[Path, str]] = []

    for path in substantive_org_files(root):
        file_errors, identifier, outgoing = validate_file(path, root)
        errors.extend(file_errors)
        if identifier:
            if identifier in ids:
                errors.append(
                    f"duplicate ID {identifier}: {ids[identifier].relative_to(root)} and {path.relative_to(root)}"
                )
            else:
                ids[identifier] = path
        links.extend((path, target) for target in outgoing)

    for path, target in links:
        if target not in ids:
            errors.append(f"{path.relative_to(root)}: unresolved id link {target}")

    errors.extend(implementation_slot_problems(roam))
    for path in tracked_generated_paths(root):
        errors.append(f"generated output is tracked: {path}")

    for ledger in (roam / ".implemented", roam / ".rejected"):
        if not ledger.exists():
            errors.append(f"missing ledger: {ledger.relative_to(root)}")

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"validated {len(substantive_org_files(root))} Org documents")
    return 0


if __name__ == "__main__":
    sys.exit(main())
