from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


PRIORITY = {"AP0": 0, "P0": 0, "P1": 1, "P2": 2, "P3": 3}
ADARD_PHASES = [
    "analyze",
    "design",
    "adversarial-review",
    "decision-gate",
    "realize-tdd-first",
    "verify-evaluate",
]


@dataclass(frozen=True)
class Candidate:
    repository: str
    number: int
    title: str
    labels: tuple[str, ...] = ()


def _priority(candidate: Candidate) -> int:
    values = [PRIORITY[label.upper()] for label in candidate.labels if label.upper() in PRIORITY]
    return min(values) if values else 99


def choose_candidate(candidates: Iterable[Candidate]) -> Candidate | None:
    values = list(candidates)
    if not values:
        return None
    return min(values, key=lambda item: (_priority(item), item.repository, item.number))


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:64] or "work"


def build_work_packet(
    candidate: Candidate,
    *,
    default_branch: str,
    head_sha: str,
    instructions: Sequence[str],
) -> dict[str, object]:
    if not default_branch or not head_sha:
        raise ValueError("Task Steward requires the exact remote default branch and head SHA")
    if not instructions:
        raise ValueError("Task Steward requires observed repository instructions before handoff")
    return {
        "repository": candidate.repository,
        "issue": candidate.number,
        "title": candidate.title,
        "base_branch": default_branch,
        "base_sha": head_sha,
        "branch": f"feature/issue-{candidate.number}-{slugify(candidate.title)}",
        "instructions": list(instructions),
        "adard": list(ADARD_PHASES),
        "rage_required": True,
        "tests_first": True,
        "merge_authorized": False,
    }


def _run_json(command: Sequence[str], runner: Callable[..., subprocess.CompletedProcess[str]]) -> object:
    result = runner(command, check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def discover_candidates(
    repositories: Sequence[str],
    *,
    label: str | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[Candidate]:
    discovered: list[Candidate] = []
    for repository in repositories:
        command = [
            "gh",
            "issue",
            "list",
            "--repo",
            repository,
            "--state",
            "open",
            "--limit",
            "100",
            "--json",
            "number,title,labels",
        ]
        if label:
            command.extend(["--label", label])
        rows = _run_json(command, runner)
        if not isinstance(rows, list):
            raise ValueError(f"unexpected GitHub issue payload for {repository}")
        for row in rows:
            labels = tuple(
                item.get("name", "")
                for item in row.get("labels", [])
                if isinstance(item, dict) and item.get("name")
            )
            discovered.append(
                Candidate(repository, int(row["number"]), str(row["title"]), labels)
            )
    return discovered


def repository_head(
    repository: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[str, str]:
    metadata = _run_json(
        ["gh", "repo", "view", repository, "--json", "defaultBranchRef"], runner
    )
    branch = metadata["defaultBranchRef"]["name"]
    ref = _run_json(
        ["gh", "api", f"repos/{repository}/git/ref/heads/{branch}"], runner
    )
    return branch, ref["object"]["sha"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Discover one bounded work item and emit a GitFlow RAGE/ADARD handoff packet."
    )
    parser.add_argument("--repo", action="append", required=True, dest="repositories")
    parser.add_argument("--label")
    parser.add_argument(
        "--instruction",
        action="append",
        default=[],
        help="Observed applicable repository instruction path; repeat for every applicable file.",
    )
    args = parser.parse_args()

    candidate = choose_candidate(
        discover_candidates(args.repositories, label=args.label)
    )
    if candidate is None:
        print(json.dumps({"status": "no-work"}, sort_keys=True))
        return 0

    branch, sha = repository_head(candidate.repository)
    packet = build_work_packet(
        candidate,
        default_branch=branch,
        head_sha=sha,
        instructions=tuple(args.instruction),
    )
    print(json.dumps(packet, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
