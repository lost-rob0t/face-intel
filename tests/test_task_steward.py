from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from task_steward import Candidate, build_work_packet, choose_candidate  # noqa: E402


class TaskStewardTests(unittest.TestCase):
    def test_selects_highest_priority_then_oldest_issue_number(self) -> None:
        candidates = [
            Candidate("lost-rob0t/quasar", 61, "filesystem persistence", ("P0",)),
            Candidate("lost-rob0t/quasar", 62, "other P0", ("P0",)),
            Candidate("lost-rob0t/quasar", 5, "lower priority", ("P1",)),
        ]
        selected = choose_candidate(candidates)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.number, 61)

    def test_returns_none_when_no_candidate_exists(self) -> None:
        self.assertIsNone(choose_candidate([]))

    def test_builds_gitflow_rage_adard_packet(self) -> None:
        candidate = Candidate(
            "lost-rob0t/quasar",
            61,
            "persist Auto-Dig runs to filesystem or Tek9 and add initfile configuration",
            ("P0",),
        )
        packet = build_work_packet(
            candidate,
            default_branch="main",
            head_sha="c1e5439b4ae84af2ba069c4b3bd363e78aef702d",
            instructions=("AGENTS.md",),
        )
        self.assertEqual(packet["repository"], "lost-rob0t/quasar")
        self.assertEqual(packet["issue"], 61)
        self.assertEqual(packet["base_branch"], "main")
        self.assertEqual(
            packet["base_sha"], "c1e5439b4ae84af2ba069c4b3bd363e78aef702d"
        )
        self.assertTrue(packet["branch"].startswith("feature/issue-61-"))
        self.assertEqual(packet["instructions"], ["AGENTS.md"])
        self.assertEqual(
            packet["adard"],
            [
                "analyze",
                "design",
                "adversarial-review",
                "decision-gate",
                "realize-tdd-first",
                "verify-evaluate",
            ],
        )
        self.assertTrue(packet["rage_required"])
        self.assertTrue(packet["tests_first"])
        self.assertFalse(packet["merge_authorized"])

    def test_refuses_handoff_without_repo_instructions_or_exact_head(self) -> None:
        candidate = Candidate("lost-rob0t/quasar", 61, "persist", ("P0",))
        with self.assertRaises(ValueError):
            build_work_packet(candidate, default_branch="main", head_sha="", instructions=())
        with self.assertRaises(ValueError):
            build_work_packet(
                candidate,
                default_branch="main",
                head_sha="abc123",
                instructions=(),
            )


if __name__ == "__main__":
    unittest.main()
