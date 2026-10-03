#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from _roamlib import append_jsonl, canonical_from_active, ensure_roam, new_event_id, now_iso, project_root


def active_for_project(roam: Path, project: str) -> Path:
    base = roam / "implement" / project
    paths = sorted(base.rglob("*.org")) if base.exists() else []
    if not paths:
        raise SystemExit(f"no active implementation for project {project}")
    if len(paths) > 1:
        rendered = ", ".join(str(path.relative_to(roam)) for path in paths)
        raise SystemExit(f"multiple active implementations for {project}: {rendered}")
    return paths[0]


def set_header(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?im)^\#\+{re.escape(key)}:\s*.*$")
    line = f"#+{key}: {value}"
    if pattern.search(text):
        return pattern.sub(line, text, count=1)
    return line + "\n" + text


def render_event(event: dict) -> str:
    lines = [
        "",
        f"* {event['status']} RAGE Execution Record",
        ":PROPERTIES:",
        f":STATUS_EVENT_ID: {event['event_id']}",
        f":RECORDED_AT: {event['timestamp']}",
        ":END:",
        "",
    ]
    if event["status"] == "IMPLEMENTED":
        lines.extend(["** Summary", event["summary"] or "No summary recorded.", "", "** Files"])
        lines.extend(f"- {value}" for value in event["files"] or ["None"])
        lines.extend(["", "** Tests"])
        lines.extend(f"- {value}" for value in event["tests"] or ["None"])
        lines.extend(["", "** Commits"])
        lines.extend(f"- {value}" for value in event["commits"] or ["None"])
    else:
        lines.extend(["** Reason", event["reason"], "", "** Evidence"])
        lines.extend(f"- {value}" for value in event["evidence"] or ["None"])
        lines.extend(["", "** Replacement", event["replacement"] or "None"])
    lines.extend(["", "** Notes"])
    lines.extend(f"- {value}" for value in event["notes"] or ["None"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    implemented = sub.add_parser("implemented")
    implemented.add_argument("--project", required=True)
    implemented.add_argument("--summary", required=True)
    implemented.add_argument("--file", action="append", default=[])
    implemented.add_argument("--test", action="append", default=[])
    implemented.add_argument("--commit", action="append", default=[])
    implemented.add_argument("--note", action="append", default=[])

    rejected = sub.add_parser("rejected")
    rejected.add_argument("--project", required=True)
    rejected.add_argument("--reason", required=True)
    rejected.add_argument("--evidence", action="append", default=[])
    rejected.add_argument("--replacement", default="")
    rejected.add_argument("--note", action="append", default=[])

    args = parser.parse_args()
    root = project_root()
    roam = ensure_roam(root)
    active = active_for_project(roam, args.project)
    canonical = canonical_from_active(active, roam)
    if not canonical.exists():
        raise SystemExit(f"canonical design missing: {canonical.relative_to(root)}")

    event = {
        "event_id": new_event_id(),
        "timestamp": now_iso(),
        "project": args.project,
        "active": str(active.relative_to(root)),
        "canonical": str(canonical.relative_to(root)),
        "status": "IMPLEMENTED" if args.command == "implemented" else "REJECTED",
        "notes": args.note,
    }
    if args.command == "implemented":
        event.update(summary=args.summary, files=args.file, tests=args.test, commits=args.commit)
        ledger = roam / ".implemented"
    else:
        event.update(reason=args.reason, evidence=args.evidence, replacement=args.replacement)
        ledger = roam / ".rejected"

    append_jsonl(ledger, event)
    text = canonical.read_text(encoding="utf-8")
    text = set_header(text, "status", event["status"])
    canonical.write_text(text.rstrip() + "\n" + render_event(event), encoding="utf-8")
    active.unlink()

    print(f"{event['status'].lower()}: {canonical.relative_to(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
