"""Opt-in worker roundtrip over actual StarIntel HTTP, RabbitMQ and local CouchDB."""

import json
import os
import unittest
import uuid
from pathlib import Path

import httpx
from support import TOKEN
from test_pipeline import FixtureProcessor, colored_image

from face_intel.actors import FaceIntelSystem
from face_intel.config import Settings
from face_intel.documents import picture_from_bytes
from face_intel.worker import StarServerClient, WorkerSettings, run_worker


@unittest.skipUnless(
    os.environ.get("FACE_INTEL_RUN_LIVE_WORKER") == "1",
    "Configure actual server HTTP, Rabbit and CouchDB fixtures to test live worker",
)
class LiveWorkerTests(unittest.TestCase):
    def test_verified_file_to_candidate_graph_over_real_transports(self):
        import pika

        real_models = bool(os.environ.get("FACE_INTEL_TEST_YUNET_MODEL"))
        service = Settings(
            api_token=TOKEN,
            couch_url=os.environ["COUCHDB_URL"],
            couch_username=os.environ["COUCHDB_USERNAME"],
            couch_password=os.environ["COUCHDB_PASSWORD"],
            couch_database="face_intel_test_" + uuid.uuid4().hex,
            sface_model_path=os.environ.get("FACE_INTEL_TEST_SFACE_MODEL", "")
            if real_models
            else "",
            sface_model_sha256=os.environ.get("FACE_INTEL_TEST_SFACE_SHA256", "")
            if real_models
            else "",
            yunet_model_path=os.environ.get("FACE_INTEL_TEST_YUNET_MODEL", "")
            if real_models
            else "",
            yunet_model_sha256=os.environ.get("FACE_INTEL_TEST_YUNET_SHA256", "")
            if real_models
            else "",
        )
        settings = WorkerSettings.from_env()
        settings = WorkerSettings(
            settings.server_url,
            settings.server_token,
            settings.rabbit_url,
            "face-intel-test-" + uuid.uuid4().hex,
        )
        server = StarServerClient(settings, service)
        system = FaceIntelSystem(
            service, processor_factory=None if real_models else FixtureProcessor
        )
        connection_holder = []
        data = (
            Path(os.environ["FACE_INTEL_TEST_FACE_IMAGE"]).read_bytes()
            if real_models
            else colored_image((193, 27, 11))
        )
        original, _ = picture_from_bytes(data, service)
        # An independent fixture ID keeps this run isolated in the remote dataset.
        original["id"] = "picture:worker-fixture:" + uuid.uuid4().hex
        incoming = None
        completed = []
        acknowledged = []

        class Channel:
            def __init__(self, channel, connection):
                self.actual, self.connection = channel, connection

            def __getattr__(self, key):
                return getattr(self.actual, key)

            def basic_ack(self, **kwargs):
                self.actual.basic_ack(**kwargs)
                acknowledged.append(kwargs["delivery_tag"])

            def basic_consume(self, **kwargs):
                callback = kwargs.pop("on_message_callback")

                def delivery(channel, method, properties, body):
                    callback(self, method, properties, body)
                    if (
                        json.loads(body).get("id") == incoming["id"]
                        and method.delivery_tag in acknowledged
                    ):
                        completed.append(True)
                        channel.stop_consuming()

                self.actual.basic_consume(on_message_callback=delivery, **kwargs)

            def start_consuming(self):
                # Replay an already committed canonical event through the real broker.
                self.actual.basic_publish(
                    exchange="documents",
                    routing_key="documents.new.picture",
                    body=json.dumps(incoming).encode(),
                )
                self.connection.call_later(30, self.actual.stop_consuming)
                self.actual.start_consuming()

        class Connection:
            def __init__(self, parameters):
                self.actual = pika.BlockingConnection(parameters)
                connection_holder.append(self.actual)

            def channel(self):
                return Channel(self.actual.channel(), self.actual)

            def close(self):
                # Clean up only this test's broker resources before closing.
                channel = self.actual.channel()
                channel.queue_delete(queue=settings.queue)
                channel.queue_delete(queue=settings.queue + ".dead")
                channel.exchange_delete(exchange=settings.queue + ".dead")
                self.actual.close()

        try:
            incoming = server.publish(original, data)
            run_worker(system, server, settings, Connection)
            self.assertTrue(completed, "Worker did not consume the committed image")
            result = system.request("process-image", {"photoId": incoming["id"]})
            self.assertEqual(result["faceCount"], 1)
            for document in result["documents"]:
                persisted = server.document(document["id"])
                self.assertEqual(persisted["dtype"], document["dtype"])
                if document["dtype"] == "picture":
                    self.assertTrue(server.image_bytes(persisted))
            claim = next(
                d
                for d in result["documents"]
                if d.get("verificationStatus") == "candidate" and d["dtype"] == "relation"
            )
            self.assertEqual(server.document(claim["id"])["verificationStatus"], "candidate")
        finally:
            for connection in connection_holder:
                if connection.is_open:
                    connection.close()
            system.close()
            server.close()
            with httpx.Client(
                base_url=service.couch_url,
                auth=(service.couch_username, service.couch_password),
                timeout=5,
            ) as cleanup:
                self.assertIn(
                    cleanup.delete("/" + service.couch_database).status_code, {200, 202, 404}
                )
