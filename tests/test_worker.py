"""Broker delivery uses verified server bytes and does not overwrite later reviews."""

import base64
import json
import unittest
from copy import deepcopy
from urllib.parse import unquote

import httpx
from support import TOKEN, MemoryStore
from test_pipeline import FixtureProcessor, colored_image

from face_intel.actors import FaceIntelSystem
from face_intel.config import Settings
from face_intel.documents import picture_from_bytes
from face_intel.errors import InvalidDocument, StorageUnavailable
from face_intel.worker import ImageWorker, StarServerClient, WorkerSettings, run_worker


class RemoteFixture:
    def __init__(self, settings):
        self.settings = settings
        self.bytes = colored_image((255, 0, 0))
        self.original, _ = picture_from_bytes(self.bytes, settings)
        self.original["rev"] = "1-remote"
        self.documents = {self.original["id"]: deepcopy(self.original)}
        self.attachments = {self.original["id"]: self.bytes}
        self.calls = []
        self.fail_publication = False
        self.pending = {}
        self.delayed_visibility = False

    def handle(self, request):
        self.calls.append(request)
        path = unquote(request.url.path)
        if request.method == "GET" and path.startswith("/api/v1/files/"):
            identifier = path.removeprefix("/api/v1/files/").removesuffix("/content")
            if identifier not in self.attachments:
                return httpx.Response(404)
            return httpx.Response(200, content=self.attachments[identifier])
        if request.method == "GET" and path.startswith("/api/v1/documents/"):
            identifier = path.removeprefix("/api/v1/documents/")
            if identifier in self.pending:
                self.documents[identifier] = self.pending.pop(identifier)
                return httpx.Response(404)
            if identifier not in self.documents:
                return httpx.Response(404)
            return httpx.Response(200, json=deepcopy(self.documents[identifier]))
        if request.method == "POST":
            body = json.loads(request.content)
            document = body["document"] if path == "/api/v1/files" else body
            if self.fail_publication and document["dtype"] == "person":
                self.fail_publication = False
                return httpx.Response(503)
            if document["id"] not in self.documents:
                assert "rev" not in document, "Local revision leaked to remote create"
            elif path == "/api/v1/files":
                assert document["rev"] == self.documents[document["id"]]["rev"]
            document["rev"] = "1-remote"
            if path == "/api/v1/files":
                binary = base64.b64decode(body["contentBase64"], validate=True)
                self.attachments[document["id"]] = binary
                self.documents[document["id"]] = deepcopy(document)
            elif self.delayed_visibility:
                self.pending[document["id"]] = deepcopy(document)
            else:
                self.documents[document["id"]] = deepcopy(document)
            return httpx.Response(202, json=document)
        raise AssertionError((request.method, path))


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(api_token=TOKEN)
        self.store = MemoryStore()
        self.system = FaceIntelSystem(self.settings, self.store, processor_factory=FixtureProcessor)
        self.addCleanup(self.system.close)
        self.remote = RemoteFixture(self.settings)
        self.worker_settings = WorkerSettings(
            "http://fixture.test", TOKEN, "amqp://guest:guest@localhost/%2F"
        )
        self.server = StarServerClient(
            self.worker_settings, self.settings, httpx.MockTransport(self.remote.handle)
        )
        self.addCleanup(self.server.close)
        self.worker = ImageWorker(self.system, self.server)

    def delivery(self, document=None):
        return self.worker.handle(json.dumps(document or self.remote.original).encode())

    def test_verified_incoming_picture_exports_graph_and_replays(self):
        result = self.delivery()
        self.assertEqual(result["faceCount"], 1)
        for doc in result["documents"]:
            self.assertIn(doc["id"], self.remote.documents)
            if doc["dtype"] == "picture":
                self.assertIn(doc["id"], self.remote.attachments)
        posts = len([r for r in self.remote.calls if r.method == "POST"])
        writes = self.store.writes
        self.assertEqual(self.delivery(), result)
        self.assertEqual(len([r for r in self.remote.calls if r.method == "POST"]), posts)
        self.assertEqual(self.store.writes, writes)

    def test_pending_document_visibility_is_checked_before_completion(self):
        self.remote.delayed_visibility = True
        result = self.delivery()
        self.assertFalse(self.remote.pending)
        for doc in result["documents"]:
            self.assertIn(doc["id"], self.remote.documents)

    def test_interrupted_publication_recovers_from_durable_local_graph(self):
        self.remote.fail_publication = True
        with self.assertRaises(StorageUnavailable):
            self.delivery()
        self.assertTrue(any(d["dtype"] == "target" for d in self.store.documents.values()))
        count = self.store.writes
        result = self.delivery()
        self.assertEqual(self.store.writes, count)
        self.assertEqual(result["faceCount"], 1)
        self.assertTrue(any(d["dtype"] == "person" for d in self.remote.documents.values()))

    def test_review_updates_fetch_latest_state_and_replay_does_not_downgrade(self):
        result = self.delivery()
        claim = next(
            d
            for d in result["documents"]
            if d["dtype"] == "relation" and d.get("verificationStatus") == "candidate"
        )
        old = deepcopy(self.remote.documents[claim["id"]])
        reviewed = {
            **old,
            "verificationStatus": "confirmed",
            "rev": "2-remote",
            "verifiedBy": "actor:review",
            "verifiedAt": 1791648000,
            "provenance": {"method": "review", "basis": "Supplied independent evidence"},
        }
        self.remote.documents[claim["id"]] = reviewed
        self.delivery(old)  # Stale notification still fetches current confirmed record.
        self.assertEqual(self.store.documents[claim["id"]]["verificationStatus"], "confirmed")
        self.delivery()
        self.assertEqual(self.remote.documents[claim["id"]]["verificationStatus"], "confirmed")

    def test_redelivery_recovers_remote_review_without_a_review_notification(self):
        result = self.delivery()
        claim = next(
            d
            for d in result["documents"]
            if d.get("verificationStatus") == "candidate" and d["dtype"] == "relation"
        )
        remote = self.remote.documents[claim["id"]]
        remote.update(
            verificationStatus="confirmed",
            verifiedBy="actor:review",
            verifiedAt=1791648000,
            rev="2-remote",
        )
        self.delivery()  # No documents.updated.relation notification was delivered.
        self.assertEqual(self.store.documents[claim["id"]]["verificationStatus"], "confirmed")
        self.assertEqual(self.remote.documents[claim["id"]]["verificationStatus"], "confirmed")

    def test_generic_file_image_is_processed_using_core_picture_derivative(self):
        source = deepcopy(self.remote.original)
        source["id"] = "file:incoming"
        source["dtype"] = "file"
        source["mediaType"] = "application/octet-stream"
        for key in ("width", "height", "pictureKind"):
            source.pop(key)
        self.remote.documents[source["id"]] = source
        self.remote.attachments[source["id"]] = self.remote.bytes
        result = self.delivery(source)
        original = result["documents"][0]
        self.assertEqual(original["dtype"], "picture")
        self.assertEqual(
            original["parentFile"], {"schema": "org.starintel/core@1/file", "id": source["id"]}
        )
        self.assertEqual(self.store.documents[source["id"]]["dtype"], "file")

    def test_invalid_bytes_dataset_and_generated_crop_handling(self):
        self.remote.attachments[self.remote.original["id"]] = b"wrong"
        with self.assertRaises(InvalidDocument):
            self.delivery()
        self.assertEqual(self.store.writes, 0)
        self.assertIsNone(self.delivery({**self.remote.original, "dataset": "other"}))
        self.remote.documents[self.remote.original["id"]]["extensions"] = {
            "faceIntel": {"faceCrop": True}
        }
        self.assertIsNone(self.delivery())
        with self.assertRaises(InvalidDocument):
            self.worker.handle(b"not json")

    def test_metadata_first_file_delivery_repairs_missing_bytes_preserving_review(self):
        self.delivery()
        crop = next(
            d
            for d in self.remote.documents.values()
            if d.get("extensions", {}).get("faceIntel", {}).get("faceCrop")
        )
        crop["provenance"] = {"basis": "Later metadata supplied by another actor"}
        self.remote.attachments.pop(crop["id"])
        local = self.store.documents[crop["id"]]
        binary = self.system.request("photo-bytes", {"id": crop["id"]})[0]
        repaired = self.server.publish(local, binary)
        self.assertEqual(repaired["provenance"], crop["provenance"])
        self.assertEqual(self.remote.attachments[crop["id"]], binary)

    def test_worker_secrets_not_in_repr(self):
        self.assertNotIn(TOKEN, repr(self.worker_settings))
        self.assertNotIn("guest:guest", repr(self.worker_settings))

    def test_broker_ack_after_work_and_dead_letter_invalid_input(self):
        events = []
        fixture = self.remote

        class Channel:
            def exchange_declare(self, **kw):
                pass

            def queue_declare(self, **kw):
                pass

            def queue_bind(self, **kw):
                events.append(("bind", kw["routing_key"]) if "routing_key" in kw else ("dead",))

            def basic_qos(self, **kw):
                pass

            def basic_consume(self, **kw):
                self.callback = kw["on_message_callback"]

            def basic_ack(self, **kw):
                events.append(("ack", kw["delivery_tag"]))

            def basic_nack(self, **kw):
                events.append(("nack", kw["requeue"]))

            def start_consuming(self):
                method = type("Method", (), {"delivery_tag": 1})()
                fixture.fail_publication = True
                self.callback(self, method, None, json.dumps(fixture.original).encode())
                assert ("nack", True) in events
                assert ("ack", 1) not in events
                self.callback(self, method, None, json.dumps(fixture.original).encode())
                assert any(d["dtype"] == "person" for d in fixture.documents.values())
                self.callback(self, method, None, b"bad")

        class Connection:
            def channel(self):
                return Channel()

            def close(self):
                events.append(("close",))

        run_worker(self.system, self.server, self.worker_settings, lambda p: Connection())
        self.assertIn(("bind", "documents.new.file"), events)
        self.assertIn(("bind", "documents.updated.picture"), events)
        self.assertIn(("bind", "documents.updated.relation"), events)
        self.assertIn(("ack", 1), events)
        self.assertIn(("nack", False), events)
        self.assertIn(("nack", True), events)
