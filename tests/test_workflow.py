from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _roamlib import implementation_slot_problems, slugify  # noqa: E402


class RoamLibTests(unittest.TestCase):
    def test_slugify(self) -> None:
        self.assertEqual(slugify("Hello, RAGE World"), "hello-rage-world")

    def test_rejects_empty_slug(self) -> None:
        with self.assertRaises(SystemExit):
            slugify("---")

    def test_implementation_slot_allows_one_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            roam = Path(tmp) / "roam"
            project = roam / "implement" / "demo"
            project.mkdir(parents=True)
            (project / "one.org").write_text("one", encoding="utf-8")
            self.assertEqual(implementation_slot_problems(roam), [])

    def test_implementation_slot_rejects_two_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            roam = Path(tmp) / "roam"
            project = roam / "implement" / "demo"
            project.mkdir(parents=True)
            (project / "one.org").write_text("one", encoding="utf-8")
            (project / "two.org").write_text("two", encoding="utf-8")
            problems = implementation_slot_problems(roam)
            self.assertEqual(len(problems), 1)
            self.assertIn("contains 2 Org files", problems[0])


if __name__ == "__main__":
    unittest.main()
