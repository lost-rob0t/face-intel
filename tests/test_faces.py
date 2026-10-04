"""Execute supplied face/candidate workflows through the real actor boundary."""

import json
import tempfile
import unittest
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from support import TOKEN, MemoryStore, candidate_person, face_observation, person, photo_bytes

from face_intel.api import create_app
from face_intel.config import Settings
from face_intel.spec import reference, verify_extension_pin, verify_pin


class FaceTests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore()
        self.client = TestClient(create_app(Settings(api_token=TOKEN), self.store))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.headers = {"Authorization": f"Bearer {TOKEN}"}

    def post(self, path, document):
        return self.client.post(path, json=document, headers=self.headers)

    def seed(self):
        picture = self.client.post("/v1/photos", content=photo_bytes(), headers=self.headers).json()
        response = self.post("/v1/face-observations", face_observation(picture))
        self.assertEqual(response.status_code, 200, response.text)
        face = response.json()
        response = self.post("/v1/candidate-persons", candidate_person(aliases=["Fixture Alias"]))
        self.assertEqual(response.status_code, 200, response.text)
        return picture, face, response.json()

    def target(self, operation, identifier, **options):
        return self.post(
            "/v1/targets",
            {
                "id": "target:face-fixture",
                "dataset": "face-intel",
                "dtype": "target",
                "schemaVersion": "0.10.1",
                "actor": "face-intel",
                "target": identifier,
                "options": {"operation": operation, **options},
            },
        )

    def test_annotation_is_idempotent_and_returns_distinct_records(self):
        picture, face, candidate = self.seed()
        annotation = {
            "faceId": face["id"],
            "personId": candidate["id"],
            "personType": "candidate-person",
            "basis": "Supplied explicit annotation",
        }
        response = self.post("/v1/links/face-person", annotation)
        self.assertEqual(response.status_code, 200, response.text)
        relation = response.json()
        self.assertEqual(relation["candidateStatus"], "candidate")
        self.assertEqual(relation["verificationStatus"], "candidate")
        self.assertEqual(relation["source"], reference(face))
        self.assertEqual(relation["destination"], reference(candidate))
        writes = self.store.writes
        self.assertEqual(self.post("/v1/links/face-person", annotation).json(), relation)
        self.assertEqual(self.store.writes, writes)
        result = self.target("get-face", face["id"])
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["documents"], [face, picture, candidate, relation])
        self.assertEqual(
            result.json()["target"]["extensions"]["faceIntel"]["resultRefs"],
            [reference(value) for value in [face, picture, candidate, relation]],
        )
        found = self.client.get("/v1/search/name?q=fixture%20alias", headers=self.headers)
        self.assertEqual(found.json()["documents"], [candidate])
        self.assertEqual(self.store.documents[candidate["id"]], candidate)
        self.assertTrue(candidate["extensions"]["faceIntel"]["candidate"])
        self.assertEqual(candidate["extensions"]["faceIntel"]["status"], "candidate")

    def test_target_ingestion_and_annotation_share_http_records(self):
        picture = self.client.post("/v1/photos", content=photo_bytes(), headers=self.headers).json()
        face = face_observation(picture)
        candidate = candidate_person()
        for operation, document in [
            ("ingest-face-observation", face),
            ("ingest-candidate-person", candidate),
        ]:
            response = self.target(operation, document["id"], document=document)
            self.assertEqual(response.status_code, 200, response.text)
        response = self.target(
            "link-face-person",
            face["id"],
            personId=candidate["id"],
            personType="candidate-person",
            basis="Supplied target annotation",
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            [d["dtype"] for d in response.json()["documents"]],
            ["face-observation", "picture", "candidate-person", "face-person-candidate"],
        )
        found = self.client.get("/v1/face-observations/" + face["id"], headers=self.headers)
        self.assertEqual(found.status_code, 200)
        self.assertEqual(len(found.json()["candidates"]), 1)
        self.assertEqual(self.target("get-candidate-person", candidate["id"]).status_code, 200)
        self.assertEqual(
            self.target("ingest-face-observation", "face:wrong", document=face).status_code, 422
        )

    def test_annotation_to_regular_person_does_not_rewrite_identity(self):
        _, face, _ = self.seed()
        record = self.post("/v1/persons", person(fullName="Supplied Name")).json()
        response = self.post(
            "/v1/links/face-person",
            {
                "faceId": face["id"],
                "personId": record["id"],
                "basis": "Supplied claim",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.store.documents[record["id"]], record)
        self.assertEqual(response.json()["destination"], reference(record))

    def test_invalid_status_reference_and_image_bounds_are_rejected_before_write(self):
        picture, _, _ = self.seed()
        invalid_faces = [
            face_observation(picture, x=-1),
            face_observation(picture, width=0),
            face_observation(picture, width=4),
            face_observation(picture, y=3),
            face_observation(
                picture, picture={"schema": "org.starintel/core@1/person", "id": picture["id"]}
            ),
        ]
        for document in invalid_faces:
            with self.subTest(document=document):
                writes = self.store.writes
                self.assertEqual(self.post("/v1/face-observations", document).status_code, 422)
                self.assertEqual(self.store.writes, writes)
        for fields in [
            {"candidateStatus": "confirmed"},
            {"verificationStatus": "verified"},
            {"annotationBasis": " "},
            {"annotationBasis": "x" * 2001},
        ]:
            self.assertEqual(
                self.post("/v1/candidate-persons", candidate_person(**fields)).status_code, 422
            )
        self.assertEqual(self.post("/v1/persons", candidate_person()).status_code, 422)

    def test_missing_record_and_wrong_destination_type_fail(self):
        _, face, candidate = self.seed()
        for destination, dtype in [("person:missing", "person"), (candidate["id"], "person")]:
            response = self.post(
                "/v1/links/face-person",
                {
                    "faceId": face["id"],
                    "personId": destination,
                    "personType": dtype,
                    "basis": "Supplied claim",
                },
            )
            self.assertEqual(response.status_code, 404, response.text)

    def test_face_revision_conflicts_are_preserved(self):
        _, face, _ = self.seed()
        updated = {**face, "annotationBasis": "Updated supplied annotation"}
        self.assertEqual(self.post("/v1/face-observations", updated).status_code, 200)
        self.assertEqual(self.post("/v1/face-observations", updated).status_code, 409)

    def test_face_routes_require_authentication(self):
        for path in ["/v1/face-observations", "/v1/candidate-persons", "/v1/links/face-person"]:
            self.assertEqual(self.client.post(path, json={}).status_code, 401)
        for path in ["/v1/face-observations/face:missing", "/v1/candidate-persons/person:missing"]:
            self.assertEqual(self.client.get(path).status_code, 401)

    def test_candidate_pages_and_invalid_target_options(self):
        _, face, first = self.seed()
        second = self.post(
            "/v1/candidate-persons", candidate_person("candidate-person:second")
        ).json()
        links = []
        for candidate in [first, second]:
            response = self.post(
                "/v1/links/face-person",
                {
                    "faceId": face["id"],
                    "personId": candidate["id"],
                    "personType": "candidate-person",
                    "basis": "Supplied paginated fixture",
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            links.append(response.json())
        result = self.client.get(
            "/v1/face-observations/" + face["id"], params={"limit": 1}, headers=self.headers
        ).json()
        self.assertEqual(len(result["candidates"]), 1)
        self.assertIsNotNone(result["nextCursor"])
        next_page = self.client.get(
            "/v1/face-observations/" + face["id"],
            params={"limit": 1, "after": result["nextCursor"]},
            headers=self.headers,
        ).json()
        self.assertIsNone(next_page["nextCursor"])
        self.assertEqual(
            {value["id"] for value in result["candidates"] + next_page["candidates"]},
            {value["id"] for value in links},
        )
        response = self.target(
            "link-face-person",
            face["id"],
            personId=first["id"],
            personType=[],
            basis="Invalid target fixture",
        )
        self.assertEqual(response.status_code, 422, response.text)

    def test_extension_pin_tampering_is_rejected(self):
        core_pin = verify_pin()
        original = files("face_intel.contracts.face_extension")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pin = json.loads(original.joinpath("pin.json").read_text())
            for name in ["pin.json", *pin["files"]]:
                root.joinpath(name).write_bytes(original.joinpath(name).read_bytes())
            root.joinpath("schema.json").write_text("{}")
            with patch("face_intel.spec.files", return_value=root):
                with self.assertRaisesRegex(RuntimeError, "extension hash mismatch"):
                    verify_extension_pin(core_pin)


if __name__ == "__main__":
    unittest.main()
