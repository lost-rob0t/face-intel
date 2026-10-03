from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import uuid
from pathlib import Path

TREE_NAMES = ("design", "research", "implement", "indexes")


def project_root() -> Path:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        return Path(result.stdout.strip()).resolve()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return Path.cwd().resolve()


def ensure_roam(root: Path | None = None) -> Path:
    roam = (root or project_root()) / "roam"
    roam.mkdir(parents=True, exist_ok=True)
    for tree in TREE_NAMES:
        (roam / tree).mkdir(parents=True, exist_ok=True)
    for ledger in (".implemented", ".rejected"):
        (roam / ledger).touch(exist_ok=True)
    mirror_structure(roam)
    return roam


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        raise SystemExit("invalid slug")
    return slug


def visible_dirs(tree: Path) -> set[Path]:
    result: set[Path] = set()
    if not tree.exists():
        return result
    for path in tree.rglob("*"):
        if not path.is_dir():
            continue
        rel = path.relative_to(tree)
        if not any(part.startswith(".") for part in rel.parts):
            result.add(rel)
    return result


def mirror_structure(roam: Path) -> set[Path]:
    for tree in TREE_NAMES:
        (roam / tree).mkdir(parents=True, exist_ok=True)
    rels: set[Path] = set()
    for tree in TREE_NAMES:
        rels.update(visible_dirs(roam / tree))
    for rel in rels:
        for tree in TREE_NAMES:
            (roam / tree / rel).mkdir(parents=True, exist_ok=True)
    return rels


def active_org_files(roam: Path) -> list[Path]:
    return sorted((roam / "implement").rglob("*.org"))


def implementation_project(path: Path, roam: Path) -> str:
    rel = path.resolve().relative_to((roam / "implement").resolve())
    return rel.parts[0] if len(rel.parts) > 1 else "."


def active_org_files_by_project(roam: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in active_org_files(roam):
        grouped.setdefault(implementation_project(path, roam), []).append(path)
    return grouped


def implementation_slot_problems(roam: Path) -> list[str]:
    problems: list[str] = []
    for project, paths in sorted(active_org_files_by_project(roam).items()):
        if len(paths) <= 1:
            continue
        rendered = ", ".join(str(path.relative_to(roam / "implement")) for path in paths)
        problems.append(
            f"implementation slot for project {project} contains {len(paths)} Org files: {rendered}"
        )
    return problems


def canonical_from_active(active: Path, roam: Path) -> Path:
    rel = active.resolve().relative_to((roam / "implement").resolve())
    return roam / "design" / rel


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    values: list[dict] = []
    if not path.exists():
        return values
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid JSONL in {path}:{number}: {exc}") from exc
        if not isinstance(value, dict):
            raise SystemExit(f"invalid ledger record in {path}:{number}")
        values.append(value)
    return values


def new_event_id() -> str:
    return str(uuid.uuid4())


def new_org_id() -> str:
    return str(uuid.uuid4())


def org_list(values: list[str], empty: str = "None") -> str:
    if values:
        return "\n".join(f"- {value}" for value in values)
    return f"- {empty}"


def validate_org_headers(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [
        key
        for key in ("title", "description", "status", "filetags")
        if not re.search(rf"(?im)^\#\+{re.escape(key)}:\s*\S", text)
    ]
    if not re.search(r"(?ms)^:PROPERTIES:\s*$.*?^:ID:\s*\S+.*?^:END:\s*$", text):
        missing.append("ID")
    if missing:
        raise SystemExit(f"invalid Org file {path}; missing: {', '.join(missing)}")
