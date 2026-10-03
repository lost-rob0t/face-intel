#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import sys

from _roamlib import ensure_roam, mirror_structure, new_org_id, project_root, slugify


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--finding", action="append", default=[])
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--repository", action="append", default=[])
    parser.add_argument("--commit", action="append", default=[])
    parser.add_argument("--design-file", action="append", default=[])
    parser.add_argument("--next-action", default="")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--draft", action="store_true")
    mode.add_argument("--final", action="store_true")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    root = project_root()
    roam = ensure_roam(root)
    directory = roam / "research" / slugify(args.project)
    directory.mkdir(parents=True, exist_ok=True)
    mirror_structure(roam)
    path = directory / f"{slugify(args.title)}.org"

    if path.exists() and not args.append:
        raise SystemExit(f"refusing overwrite: {path.relative_to(root)}")
    if args.append and not path.exists():
        raise SystemExit(f"cannot append missing note: {path.relative_to(root)}")

    now = dt.datetime.now().astimezone()
    timestamp = now.isoformat(timespec="seconds")
    date = now.strftime("%Y-%m-%d")
    state = "DRAFT" if args.draft else "REVIEW"

    if not path.exists():
        description = args.description or "TODO"
        path.write_text(
            f""":PROPERTIES:\n:ID: {new_org_id()}\n:END:\n#+title: {args.title}\n#+description: {description}\n#+status: {state}\n#+filetags: :rage:research:{slugify(args.project)}:\n\n* Approval Table\n\n| Approval area | Required authority | State | Evidence required | Evidence reference |\n|---------------+--------------------+-------+-------------------+--------------------|\n| Research      | Project owner      | PENDING | Sources and findings reviewed | |\n| Execution     | Project owner      | NOT STARTED | Approved bounded next action | |\n\n* Objective\n\n{description}\n\n* Verified Facts\n\n* Inference\n\n* Contradictions\n\n* Unresolved Questions\n\n* Findings\n\n* Changelog\n\n| Date | Change | Author or actor | Evidence |\n|------+--------+-----------------+----------|\n| {date} | Created research note | save-research.py | repository diff |\n""",
            encoding="utf-8",
        )

    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n** Research update {timestamp}\n")
        for value in args.finding or ["TODO"]:
            handle.write(f"- {value}\n")
        handle.write("\n*** Sources\n")
        for value in args.source or ["TODO"]:
            handle.write(f"- Retrieved {date}: {value}\n")
        handle.write("\n*** Repositories Reviewed\n")
        for value in args.repository or ["None"]:
            handle.write(f"- {value}\n")
        handle.write("\n*** Commits Reviewed\n")
        for value in args.commit or ["None"]:
            handle.write(f"- {value}\n")
        handle.write("\n*** Affected Design Files\n")
        for value in args.design_file or ["None"]:
            handle.write(f"- {value}\n")
        handle.write(f"\n*** Next Action\n{args.next_action or 'TODO'}\n")

    print(path.relative_to(root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
