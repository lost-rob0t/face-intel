import json
import unittest
from copy import deepcopy
from urllib.parse import unquote

import httpx
from support import TOKEN, person, photo_bytes

from face_intel.config import Settings
from face_intel.couch import CouchStore
from face_intel.documents import picture_from_bytes, prepare_person
from face_intel.errors import Conflict, StorageUnavailable


class FixtureCouch:
    def __init__(self):
        self.documents = {}
        self.calls = []
        self.counter = 0
        self.concurrent_create = False

    def handle(self, request):
        self.calls.append(request)
        identifier = unquote(request.url.path.split("/", 2)[-1])
        if request.url.path == "/face_intel" and request.method == "PUT":
            return httpx.Response(201, json={"ok": True})
        if request.method == "GET":
            if identifier not in self.documents:
                return httpx.Response(404, json={"error": "not_found"})
            return httpx.Response(200, json=deepcopy(self.documents[identifier]))
        if request.method == "PUT":
            value = json.loads(request.content)
            existing = self.documents.get(identifier)
            if existing and value.get("_rev") != existing["_rev"]:
                return httpx.Response(409, json={"error": "conflict"})
            self.counter += 1
            value["_rev"] = f"{self.counter}-fixture"
            self.documents[identifier] = deepcopy(value)
            if self.concurrent_create and not existing:
                self.concurrent_create = False
                return httpx.Response(409, json={"error": "conflict"})
            return httpx.Response(201, json={"id": identifier, "rev": value["_rev"]})
        raise AssertionError((request.method, request.url))


class CouchTests(unittest.TestCase):
    def setUp(self):
        self.fixture = FixtureCouch()
        self.settings = Settings(api_token=TOKEN)
        self.store = CouchStore(self.settings, httpx.MockTransport(self.fixture.handle))

    def tearDown(self):
        self.store.close()

    def test_attachment_and_metadata_are_one_atomic_write(self):
        data = photo_bytes()
        picture, media_type = picture_from_bytes(data, self.settings)
        saved = self.store.put(picture, (data, media_type))
        puts = [r for r in self.fixture.calls if r.method == "PUT"]
        self.assertEqual(len(puts), 1)
        wire = json.loads(puts[0].content)
        self.assertEqual(wire["_id"], picture["id"])
        self.assertEqual(wire["_attachments"]["original"]["content_type"], "image/png")
        self.assertIn("data", wire["_attachments"]["original"])
        self.assertNotIn("rev", wire)
        fetched = self.store.get(picture["id"], "picture")
        self.assertEqual(saved, fetched)
        self.assertNotIn("_id", fetched)
        self.assertNotIn("_attachments", fetched)
        self.assertEqual(self.store.put(picture, (data, media_type)), saved)
        self.assertEqual(len([r for r in self.fixture.calls if r.method == "PUT"]), 1)

    def test_ids_with_slashes_are_encoded_as_one_document_id(self):
        supplied = prepare_person(person(identifier="person:fixture/one"), "face-intel")
        saved = self.store.put(supplied)
        put = next(r for r in self.fixture.calls if r.method == "PUT")
        self.assertEqual(put.url.raw_path, b"/face_intel/person%3Afixture%2Fone")
        self.assertEqual(saved["id"], "person:fixture/one")

    def test_current_revision_required_and_stale_retries_rejected(self):
        initial = self.store.put(prepare_person(person(), "face-intel"))
        updated = {**initial, "bio": "Supplied change"}
        saved = self.store.put(updated)
        self.assertNotEqual(saved["rev"], initial["rev"])
        with self.assertRaises(Conflict):
            self.store.put(updated)
        with self.assertRaises(Conflict):
            self.store.put(prepare_person(person(bio="Other change"), "face-intel"))

    def test_identical_create_race_is_idempotent(self):
        self.fixture.concurrent_create = True
        initial = self.store.put(prepare_person(person(), "face-intel"))
        self.assertEqual(initial["id"], "person:fixture")
        self.assertEqual(len(self.fixture.documents), 1)

    def test_view_install_is_idempotent_and_preserves_unrelated_views(self):
        self.fixture.documents["_design/face-intel"] = {
            "_id": "_design/face-intel",
            "_rev": "0-fixture",
            "views": {
                "unrelated": {"map": "function(doc) { emit(doc.id, null); }"},
            },
        }
        self.store.initialize()
        design = self.fixture.documents["_design/face-intel"]
        self.assertIn("unrelated", design["views"])
        count = len([r for r in self.fixture.calls if r.method == "PUT"])
        self.store.initialize()
        # The database create is retried, but the identical design document is untouched.
        self.assertEqual(len([r for r in self.fixture.calls if r.method == "PUT"]), count + 1)

    def test_deleted_cursor_does_not_skip_the_following_record(self):
        rows = [
            {
                "id": f"person:{i}",
                "doc": {
                    **person(identifier=f"person:{i}"),
                    "_id": f"person:{i}",
                    "_rev": "1-fixture",
                },
            }
            for i in [2, 3]
        ]
        requests = []

        def view(request):
            requests.append(request)
            return httpx.Response(200, json={"rows": rows})

        store = CouchStore(self.settings, httpx.MockTransport(view))
        try:
            page = store.page("by_name", "fixture person", 1, after="person:1")
            self.assertEqual(page["documents"][0]["id"], "person:2")
            self.assertEqual(page["nextCursor"], "person:2")
            self.assertNotIn("skip", requests[0].url.params)
        finally:
            store.close()

    def test_network_error_is_sanitized(self):
        def failure(request):
            raise httpx.ConnectError("fixture-only-sensitive-detail", request=request)

        store = CouchStore(self.settings, httpx.MockTransport(failure))
        try:
            with self.assertRaises(StorageUnavailable) as caught:
                store.get("person:fixture", "person")
            self.assertNotIn("sensitive-detail", str(caught.exception))
            with self.assertRaises(StorageUnavailable):
                store.initialize()
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
