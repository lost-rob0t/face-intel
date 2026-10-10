"""Core-only image association lifecycle through HTTP and actor dispatch."""

import unittest
from importlib.resources import files

from fastapi.testclient import TestClient
from support import TOKEN, MemoryStore, candidate_person, person, photo_bytes

from face_intel.api import create_app
from face_intel.config import Settings
from face_intel.documents import LINK_PREDICATE
from face_intel.spec import reference, validate


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
        response = self.post("/v1/persons", candidate_person(aliases=["Fixture Alias"]))
        self.assertEqual(response.status_code, 200, response.text)
        record = response.json()
        response = self.post(
            "/v1/links/photo-person",
            {
                "photoId": picture["id"],
                "personId": record["id"],
                "basis": "Supplied claim",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        return picture, record, response.json()

    def target(self, operation, identifier, **options):
        return self.post(
            "/v1/targets",
            {
                "id": "target:fixture",
                "dataset": "face-intel",
                "dtype": "target",
                "schemaVersion": "0.10.1",
                "actor": "face-intel",
                "target": identifier,
                "options": {"operation": operation, **options},
            },
        )

    def test_core_records_candidate_metadata_and_idempotency(self):
        picture, record, relation = self.seed()
        self.assertEqual(record["dtype"], "person")
        self.assertEqual(relation["dtype"], "relation")
        self.assertEqual(relation["predicate"], LINK_PREDICATE)
        self.assertEqual(relation["source"], reference(picture))
        self.assertEqual(relation["destination"], reference(record))
        self.assertEqual(relation["verificationStatus"], "candidate")
        self.assertTrue(relation["extensions"]["faceIntel"]["candidate"])
        for document in [picture, record, relation]:
            validate(document, document["dtype"], "face-intel")
        writes = self.store.writes
        self.assertEqual(
            self.post(
                "/v1/links/photo-person",
                {
                    "photoId": picture["id"],
                    "personId": record["id"],
                    "basis": "Supplied claim",
                },
            ).json(),
            relation,
        )
        self.assertEqual(self.store.writes, writes)
        found = self.client.get("/v1/search/name?q=fixture%20alias", headers=self.headers)
        self.assertEqual(found.json()["documents"], [record])

    def test_relation_confirmation_preserves_ids_and_does_not_confirm_person(self):
        picture, record, relation = self.seed()
        updated = {
            **relation,
            "verificationStatus": "confirmed",
            "provenance": {
                "method": "review",
                "basis": "Supplied confirmation evidence",
                "reviewer": "fixture",
            },
        }
        response = self.post("/v1/relations", updated)
        self.assertEqual(response.status_code, 200, response.text)
        confirmed = response.json()
        self.assertEqual(confirmed["id"], relation["id"])
        self.assertNotEqual(confirmed["rev"], relation["rev"])
        self.assertFalse(confirmed["extensions"]["faceIntel"]["candidate"])
        self.assertEqual(confirmed["extensions"]["faceIntel"]["status"], "confirmed")
        self.assertEqual(self.store.documents[record["id"]], record)
        self.assertEqual(self.post("/v1/relations", updated).status_code, 409)
        self.assertEqual(
            self.post(
                "/v1/relations",
                {
                    **{k: v for k, v in updated.items() if k != "rev"},
                    "verificationStatus": "rejected",
                },
            ).status_code,
            409,
        )
        bundle = self.client.get("/v1/persons/" + record["id"], headers=self.headers).json()
        self.assertEqual(bundle["photos"], [picture])
        self.assertEqual(bundle["relations"], [confirmed])
        rejected = self.post("/v1/relations", {**confirmed, "verificationStatus": "rejected"})
        self.assertEqual(rejected.status_code, 200, rejected.text)
        self.assertFalse(rejected.json()["extensions"]["faceIntel"]["candidate"])

    def test_person_confirmation_does_not_confirm_image_association(self):
        _, record, relation = self.seed()
        response = self.post("/v1/persons", {**record, "verificationStatus": "confirmed"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["id"], record["id"])
        self.assertFalse(response.json()["extensions"]["faceIntel"]["candidate"])
        self.assertEqual(response.json()["extensions"]["faceIntel"]["status"], "confirmed")
        self.assertEqual(self.store.documents[relation["id"]], relation)

    def test_relation_http_target_parity_and_auth(self):
        _, _, relation = self.seed()
        response = self.target("ingest-relation", relation["id"], document=relation)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["documents"], [relation])
        response = self.target("get-relation", relation["id"])
        self.assertEqual(response.json()["documents"], [relation])
        self.assertEqual(
            response.json()["target"]["extensions"]["faceIntel"]["resultRefs"],
            [reference(relation)],
        )
        response = self.client.get("/v1/relations/" + relation["id"], headers=self.headers)
        self.assertEqual(response.json(), relation)
        self.assertEqual(self.client.post("/v1/relations", json=relation).status_code, 401)
        self.assertEqual(self.client.get("/v1/relations/" + relation["id"]).status_code, 401)
        self.assertEqual(
            self.target("ingest-relation", "relation:wrong", document=relation).status_code, 422
        )

    def test_image_region_bounds_and_claim_validation(self):
        _, _, relation = self.seed()
        region = {"x": 0, "y": 0, "width": 2, "height": 3}
        valid = {**relation, "extensions": {"faceIntel": {"region": region}}}
        response = self.post("/v1/relations", valid)
        self.assertEqual(response.status_code, 200, response.text)
        current = response.json()
        for patch in [{"x": -1}, {"width": 0}, {"width": 4}, {"y": 3}, {"x": True}, {"extra": 1}]:
            document = {**current, "extensions": {"faceIntel": {"region": {**region, **patch}}}}
            writes = self.store.writes
            with self.subTest(patch=patch):
                self.assertEqual(self.post("/v1/relations", document).status_code, 422)
                self.assertEqual(self.store.writes, writes)
        for patch in [
            {"verificationStatus": "unknown"},
            {"provenance": {}},
            {"provenance": {"basis": " "}},
            {"extensions": {"faceIntel": []}},
        ]:
            self.assertEqual(self.post("/v1/relations", {**current, **patch}).status_code, 422)

    def test_typed_endpoints_must_exist_and_use_core(self):
        _, _, relation = self.seed()
        for ref, status in [
            ({"schema": "org.starintel/core@1/person", "id": "person:missing"}, 404),
            ({"schema": "org.starintel/face-intel@1/candidate-person", "id": "x"}, 422),
            ({"schema": "org.starintel/core@1/file", "id": "x"}, 422),
        ]:
            self.assertEqual(
                self.post("/v1/relations", {**relation, "destination": ref}).status_code, status
            )

    def test_core_derived_from_relation(self):
        original, _, _ = self.seed()
        crop = self.client.post(
            "/v1/photos", content=photo_bytes(2, 2), headers=self.headers
        ).json()
        document = {
            "id": "relation:crop-origin",
            "dataset": "face-intel",
            "dtype": "relation",
            "schemaVersion": "0.10.1",
            "source": reference(crop),
            "destination": reference(original),
            "predicate": "org.starintel/core@1/derived-from",
            "provenance": {"method": "supplied-crop"},
        }
        response = self.post("/v1/relations", document)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["source"], reference(crop))
        self.assertEqual(response.json()["destination"], reference(original))

    def test_extension_types_are_removed_not_advertised(self):
        manifest = self.client.get("/v1/manifest", headers=self.headers).json()
        self.assertEqual(set(manifest["accepts"]), {"person", "picture", "relation", "target"})
        self.assertFalse(files("face_intel.contracts").joinpath("face_extension").is_dir())
        for dtype in ["candidate-person", "face-observation", "face-person-candidate"]:
            document = {**person(), "dtype": dtype}
            self.assertEqual(self.post("/v1/persons", document).status_code, 422)
            self.assertEqual(self.post("/v1/relations", document).status_code, 422)
        self.assertEqual(
            self.post("/v1/persons", {**person(), "candidateStatus": "candidate"}).status_code, 422
        )
        self.assertEqual(
            self.target("ingest-candidate-person", "person:x", document=person()).status_code, 422
        )
        self.assertEqual(self.post("/v1/candidate-persons", person()).status_code, 404)
        self.assertEqual(self.post("/v1/face-observations", {}).status_code, 404)
