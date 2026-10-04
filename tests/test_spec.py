import hashlib
import json
import tempfile
import unittest
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

from support import TOKEN, person

from face_intel.config import Settings
from face_intel.errors import InvalidDocument
from face_intel.spec import validate, verify_pin


class SpecTests(unittest.TestCase):
    def test_pin_matches_upstream_release_hashes(self):
        pin = verify_pin()
        self.assertEqual(pin["commit"], "5837e9924cf4ead99ccf085521121c7f6ad4703e")
        self.assertEqual(pin["schemaVersion"], "0.10.1")
        root = files("face_intel.contracts")
        for name, expected in pin["files"].items():
            self.assertEqual(hashlib.sha256(root.joinpath(name).read_bytes()).hexdigest(), expected)

    def test_modified_generated_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = files("face_intel.contracts")
            pin = json.loads(source.joinpath("pin.json").read_text())
            for name in ["pin.json", *pin["files"]]:
                root.joinpath(name).write_bytes(source.joinpath(name).read_bytes())
            root.joinpath("schema.json").write_text("{}")
            with patch("face_intel.spec.files", return_value=root):
                with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                    verify_pin()

    def test_reference_is_typed_and_dates_are_checked(self):
        invalid = person(images=["picture:untyped"])
        with self.assertRaises(InvalidDocument):
            validate(invalid, "person", "face-intel")
        invalid = person(dob="2026-99-99")
        with self.assertRaises(InvalidDocument):
            validate(invalid, "person", "face-intel")
        validate(
            person(images=[{"schema": "org.starintel/core@1/picture", "id": "picture:1"}]),
            "person",
            "face-intel",
        )

    def test_secrets_are_environment_only_and_not_in_repr(self):
        values = {
            "FACE_INTEL_API_TOKEN": TOKEN,
            "COUCHDB_USERNAME": "fixture-user",
            "COUCHDB_PASSWORD": "fixture-password",
        }
        with patch.dict("os.environ", values, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.couch_username, "fixture-user")
        self.assertNotIn(TOKEN, repr(settings))
        self.assertNotIn("fixture-password", repr(settings))
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                Settings.from_env()

    def test_configuration_rejects_embedded_credentials_and_invalid_bounds(self):
        for values in [
            {"couch_url": "http://user:password@localhost:5984"},
            {"couch_url": "http://localhost:5984/db"},
            {"couch_database": "../other"},
            {"max_photo_bytes": 0},
            {"couch_timeout": 40},
            {"couch_username": "user"},
        ]:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    Settings(api_token=TOKEN, **values)


if __name__ == "__main__":
    unittest.main()
