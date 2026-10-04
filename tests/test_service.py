import base64
import hashlib
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from queue import Full

import pykka
from fastapi.testclient import TestClient
from support import TOKEN, MemoryStore, person, photo_bytes

from face_intel.actors import BoundedInbox, Command, FaceIntelSystem
from face_intel.api import create_app
from face_intel.config import Settings
from face_intel.errors import InvalidDocument, StorageUnavailable


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(api_token=TOKEN)
        self.store = MemoryStore()
        self.client = TestClient(create_app(self.settings, self.store))
        self.client.__enter__()
        self.headers = {"Authorization": f"Bearer {TOKEN}"}

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.assertTrue(self.store.closed)

    def put_person(self, document=None):
        response = self.client.post("/v1/persons", json=document or person(), headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def put_photo(self):
        response = self.client.post("/v1/photos", content=photo_bytes(), headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def annotate(self, picture, record):
        response = self.client.post(
            "/v1/links/photo-person",
            json={
                "photoId": picture["id"],
                "personId": record["id"],
                "basis": "Explicit fixture annotation supplied by the record owner",
            },
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_full_http_and_target_bundle(self):
        record = self.put_person(person(aliases=["ＦＩＸＴＵＲＥ   ALIAS"]))
        picture = self.put_photo()
        relation = self.annotate(picture, record)
        self.assertTrue(relation["extensions"]["faceIntel"]["candidate"])
        self.assertEqual(relation["verificationStatus"], "candidate")
        result = self.client.get(
            "/v1/search/name", params={"q": "fixture alias"}, headers=self.headers
        ).json()
        self.assertEqual([d["id"] for d in result["documents"]], [record["id"]])
        bundle = self.client.get("/v1/persons/" + record["id"], headers=self.headers).json()
        self.assertEqual(bundle["person"], record)
        self.assertEqual(bundle["photos"], [picture])
        self.assertEqual(bundle["relations"], [relation])
        target = {
            "id": "target:fixture",
            "dataset": "face-intel",
            "dtype": "target",
            "schemaVersion": "0.10.1",
            "actor": "face-intel",
            "target": record["id"],
            "options": {"operation": "get-person"},
        }
        response = self.client.post("/v1/targets", json=target, headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text)
        output = response.json()
        self.assertEqual(output["documents"], [record, picture, relation])
        self.assertEqual(output["target"]["state"], "completed")
        self.assertEqual(len(output["target"]["extensions"]["faceIntel"]["resultRefs"]), 3)
        self.assertNotIn("state", target)
        original = self.client.get("/v1/photo-bytes/" + picture["id"], headers=self.headers)
        self.assertEqual(original.content, photo_bytes())
        self.assertEqual(original.headers["content-type"], "image/png")

    def test_authentication_on_every_data_route(self):
        for route in [
            "/v1/manifest",
            "/v1/search/name?q=fixture",
            "/v1/persons/person:fixture",
            "/v1/photos/picture:fixture",
            "/v1/photo-bytes/picture:fixture",
        ]:
            with self.subTest(route=route):
                self.assertEqual(self.client.get(route).status_code, 401)
        for route in ["/v1/persons", "/v1/photos", "/v1/targets", "/v1/links/photo-person"]:
            with self.subTest(route=route):
                self.assertEqual(self.client.post(route, json={}).status_code, 401)
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(
            self.client.get(
                "/v1/manifest",
                headers={
                    "Authorization": "Bearer wrong-fixture-token",
                },
            ).status_code,
            401,
        )

    def test_manifest_is_canonical(self):
        response = self.client.get("/v1/manifest", headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["schemaVersion"], "0.10.1")
        self.assertIn("person.search-by-name", response.json()["capabilities"])

    def test_revision_checks_and_idempotent_writes(self):
        first = self.put_person()
        self.assertEqual(self.put_person(), first)
        picture = self.put_photo()
        self.assertEqual(self.put_photo(), picture)
        link = self.annotate(picture, first)
        self.assertEqual(self.annotate(picture, first), link)
        update = {**first, "bio": "Supplied biography"}
        second = self.put_person(update)
        self.assertNotEqual(second["rev"], first["rev"])
        stale = self.client.post("/v1/persons", json=update, headers=self.headers)
        self.assertEqual(stale.status_code, 409)
        without_rev = self.client.post(
            "/v1/persons", json=person(bio="Changed"), headers=self.headers
        )
        self.assertEqual(without_rev.status_code, 409)

    def test_same_names_remain_separate_people_with_pagination(self):
        records = [self.put_person(person(identifier=f"person:fixture-{i}")) for i in range(3)]
        response = self.client.get(
            "/v1/search/name", params={"q": "FIXTURE PERSON", "limit": 2}, headers=self.headers
        )
        page = response.json()
        self.assertEqual(page["documents"], records[:2])
        last = self.client.get(
            "/v1/search/name",
            params={
                "q": "FIXTURE PERSON",
                "limit": 2,
                "after": page["nextCursor"],
            },
            headers=self.headers,
        ).json()
        self.assertEqual(last["documents"], records[2:])
        self.assertIsNone(last["nextCursor"])

    def test_contract_rejects_old_keys_unknown_fields_and_cross_dataset(self):
        invalid = [
            person(schemaVersion="0.9.0"),
            person(dataset="other"),
            person(schema_version="0.10.1"),
            person(_id="person:wrong"),
            person(extensions={"faceIntel": "wrong"}),
        ]
        for value in invalid:
            with self.subTest(value=value):
                response = self.client.post("/v1/persons", json=value, headers=self.headers)
                self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(self.store.writes, 0)
        self.put_person()  # Errors do not kill the actor.

    def test_unnamed_person_records_are_supported(self):
        supplied = person()
        del supplied["fullName"]
        record = self.put_person(supplied)
        self.assertEqual(record["extensions"]["faceIntel"]["nameKeys"], [])

    def test_request_and_image_limits(self):
        response = self.client.post("/v1/photos", content=b"not-an-image", headers=self.headers)
        self.assertEqual(response.status_code, 422)
        response = self.client.post(
            "/v1/photos", content=b"x" * (self.settings.max_photo_bytes + 1), headers=self.headers
        )
        self.assertEqual(response.status_code, 413)
        for body in [b"[]", b"{", b"x" * (1024 * 1024 + 1)]:
            response = self.client.post("/v1/persons", content=body, headers=self.headers)
            self.assertIn(response.status_code, {413, 422})

    def test_link_does_not_accept_inferred_identity_or_missing_records(self):
        record = self.put_person()
        picture = self.put_photo()
        payload = {"photoId": picture["id"], "personId": record["id"], "basis": "fixture"}
        for extra in [{"status": "confirmed"}, {"similarity": 0.99}, {"basis": "   "}]:
            response = self.client.post(
                "/v1/links/photo-person", json={**payload, **extra}, headers=self.headers
            )
            self.assertEqual(response.status_code, 422)
        missing = self.client.post(
            "/v1/links/photo-person",
            json={
                **payload,
                "personId": "person:missing",
            },
            headers=self.headers,
        )
        self.assertEqual(missing.status_code, 404)

    def test_targets_require_canonical_actor_and_operation(self):
        target = {
            "id": "target:fixture",
            "dataset": "face-intel",
            "dtype": "target",
            "schemaVersion": "0.10.1",
            "actor": "face-intel",
            "target": "fixture",
            "options": {"operation": "search-name"},
        }
        for value in [
            {**target, "actor": "other"},
            {**target, "schemaVersion": "0.9.0"},
            {**target, "options": {"operation": "unknown"}},
            {**target, "options": {"operation": "search-name", "limit": True}},
            {**target, "extensions": {"faceIntel": "wrong"}},
        ]:
            response = self.client.post("/v1/targets", json=value, headers=self.headers)
            self.assertEqual(response.status_code, 422, response.text)
        self.put_person()
        output = self.client.post(
            "/v1/targets", json={**target, "target": "Fixture Person"}, headers=self.headers
        ).json()
        self.assertEqual(len(output["documents"]), 1)

    def test_target_ingest_and_annotation_share_http_actors(self):
        def execute(operation, identifier, **options):
            target = {
                "id": f"target:{operation}",
                "dataset": "face-intel",
                "dtype": "target",
                "schemaVersion": "0.10.1",
                "actor": "face-intel",
                "target": identifier,
                "options": {"operation": operation, **options},
            }
            return self.client.post("/v1/targets", json=target, headers=self.headers)

        response = execute("ingest-person", "person:fixture", document=person())
        self.assertEqual(response.status_code, 200, response.text)
        record = response.json()["documents"][0]
        data = photo_bytes()
        picture_id = "picture:sha256:" + hashlib.sha256(data).hexdigest()
        response = execute("ingest-photo", picture_id, photoBase64=base64.b64encode(data).decode())
        self.assertEqual(response.status_code, 200, response.text)
        picture = response.json()["documents"][0]
        response = execute("link-photo-person", picture_id, personId=record["id"], basis="fixture")
        self.assertEqual(response.status_code, 200, response.text)
        documents = response.json()["documents"]
        self.assertEqual(documents[:2], [record, picture])
        self.assertTrue(documents[2]["extensions"]["faceIntel"]["candidate"])
        for operation, identifier, options in [
            ("ingest-person", "person:wrong", {"document": person()}),
            ("ingest-photo", picture_id, {"photoBase64": "invalid="}),
            ("ingest-photo", "picture:wrong", {"photoBase64": base64.b64encode(data).decode()}),
            ("link-photo-person", picture_id, {"personId": None, "basis": "fixture"}),
        ]:
            with self.subTest(operation=operation, options=options):
                self.assertEqual(execute(operation, identifier, **options).status_code, 422)

    def test_concurrent_ingest_keeps_store_actor_ownership(self):
        system = self.client.app.state.system
        with ThreadPoolExecutor(max_workers=8) as executor:
            records = list(
                executor.map(
                    lambda i: system.request(
                        "ingest-person", person(identifier=f"person:thread-{i}")
                    ),
                    range(24),
                )
            )
        self.assertEqual(len({record["id"] for record in records}), 24)
        self.assertNotEqual(self.store.owner, __import__("threading").get_ident())


class ActorLifecycleTests(unittest.TestCase):
    def test_failed_startup_cleans_up_only_owned_actors(self):
        class BrokenStore(MemoryStore):
            def initialize(self):
                super().initialize()
                raise StorageUnavailable("fixture outage")

        before = {ref.actor_urn for ref in pykka.ActorRegistry.get_all()}
        store = BrokenStore()
        with self.assertRaises(StorageUnavailable):
            FaceIntelSystem(Settings(api_token=TOKEN), store)
        self.assertTrue(store.closed)
        self.assertEqual({ref.actor_urn for ref in pykka.ActorRegistry.get_all()}, before)

    def test_expired_command_cannot_write(self):
        store = MemoryStore()
        system = FaceIntelSystem(Settings(api_token=TOKEN), store)
        try:
            with self.assertRaises(pykka.Timeout):
                system.persons.ask(
                    Command("ingest-person", person(), time.monotonic() - 1), timeout=1
                )
            self.assertEqual(store.writes, 0)
        finally:
            system.close()

    def test_mailbox_capacity_is_enforced(self):
        inbox = BoundedInbox(maxsize=1)
        inbox.put("first")
        with self.assertRaises(Full):
            inbox.put("second", block=False)

    def test_pixel_budget_is_enforced(self):
        from face_intel.documents import picture_from_bytes

        settings = replace(Settings(api_token=TOKEN), max_photo_pixels=1)
        with self.assertRaises(InvalidDocument):
            picture_from_bytes(photo_bytes(), settings)


if __name__ == "__main__":
    unittest.main()
