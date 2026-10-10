"""Rabbit committed-image worker using StarIntel Server's verified file API."""

import base64
import hashlib
import json
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from urllib.parse import quote, urlsplit

import httpx

from .actors import FaceIntelSystem
from .config import Settings
from .errors import Conflict, InvalidDocument, NotFound, StorageUnavailable
from .spec import validate


@dataclass(frozen=True)
class WorkerSettings:
    server_url: str
    server_token: str = field(repr=False)
    rabbit_url: str = field(repr=False)
    queue: str = "face-intel.images"
    timeout: float = 30

    def __post_init__(self):
        origin = urlsplit(self.server_url)
        if (
            origin.scheme not in {"http", "https"}
            or not origin.hostname
            or origin.username
            or origin.password
            or origin.query
            or origin.fragment
        ):
            raise ValueError(
                "STARINTEL_SERVER_URL must be an HTTP URL without embedded credentials"
            )
        if not self.server_token or not self.rabbit_url or not self.queue:
            raise ValueError("Configure server token, Rabbit URL and owned queue")

    @classmethod
    def from_env(cls):
        return cls(
            os.environ.get("STARINTEL_SERVER_URL", ""),
            os.environ.get("STARINTEL_SERVER_TOKEN", ""),
            os.environ.get("FACE_INTEL_RABBIT_URL", ""),
            os.environ.get("FACE_INTEL_RABBIT_QUEUE", "face-intel.images"),
        )


class StarServerClient:
    def __init__(self, settings: WorkerSettings, service: Settings, transport=None):
        self.settings, self.service = settings, service
        self.client = httpx.Client(
            base_url=settings.server_url.rstrip("/") + "/",
            headers={"Authorization": "Bearer " + settings.server_token},
            timeout=settings.timeout,
            transport=transport,
            follow_redirects=False,
        )

    def close(self):
        self.client.close()

    def request(self, method, path, limit=1024 * 1024, **kwargs):
        try:
            with self.client.stream(method, path, **kwargs) as response:
                if response.status_code == 404:
                    raise NotFound("Server record not found")
                if response.status_code == 409:
                    raise Conflict("Server revision conflict")
                if not response.is_success:
                    raise StorageUnavailable("StarIntel Server rejected request")
                body = bytearray()
                for chunk in response.iter_bytes():
                    if len(body) + len(chunk) > limit:
                        raise InvalidDocument("Server response exceeds byte limit")
                    body.extend(chunk)
                return bytes(body)
        except httpx.HTTPError as exc:
            raise StorageUnavailable("StarIntel Server request failed") from exc

    def document(self, identifier):
        body = self.request("GET", "api/v1/documents/" + quote(identifier, safe=""))
        try:
            document = json.loads(body)
        except (ValueError, UnicodeDecodeError) as exc:
            raise InvalidDocument("Invalid server document response") from exc
        if not isinstance(document, dict) or document.get("id") != identifier:
            raise InvalidDocument("Server response has an unexpected document ID")
        validate(document, document.get("dtype"), self.service.dataset)
        return document

    def image_bytes(self, document):
        data = self.request(
            "GET",
            "api/v1/files/" + quote(document["id"], safe="") + "/content",
            limit=self.service.max_photo_bytes,
        )
        if document.get("bytesHashAlgorithm") != "sha256" or hashlib.sha256(
            data
        ).hexdigest() != document.get("bytesHash"):
            raise InvalidDocument("Incoming image bytes do not match their core document")
        if "sizeBytes" in document and len(data) != document["sizeBytes"]:
            raise InvalidDocument("Incoming image byte size differs from metadata")
        return data

    def publish(self, document, image_data=None):
        # Read latest state before any delivery. Replays must not downgrade another actor's review.
        try:
            existing = self.document(document["id"])
            if existing["dtype"] != document["dtype"]:
                raise InvalidDocument("Remote ID exists with a different type")
            if (
                document["dtype"] in {"file", "image", "picture"}
                and existing["bytesHash"] != document["bytesHash"]
            ):
                raise InvalidDocument("Remote image ID has different content")
            if image_data is not None:
                try:
                    self.image_bytes(existing)
                except NotFound:
                    # Attach bytes to metadata-first records without replacing latest reviews.
                    self.request(
                        "POST",
                        "api/v1/files",
                        json={
                            "document": existing,
                            "contentBase64": base64.b64encode(image_data).decode("ascii"),
                        },
                    )
                    existing = self.document(document["id"])
                    self.image_bytes(existing)
            return existing
        except NotFound:
            pass
        outgoing = deepcopy(document)
        outgoing.pop("rev", None)  # Local CouchDB revisions are not remote preconditions.
        if image_data is not None:
            self.request(
                "POST",
                "api/v1/files",
                json={
                    "document": outgoing,
                    "contentBase64": base64.b64encode(image_data).decode("ascii"),
                },
            )
        else:
            self.request("POST", "api/v1/documents", json=outgoing)
        # Generic metadata creation can be accepted before durable persistence.
        deadline = time.monotonic() + self.settings.timeout
        while True:
            try:
                persisted = self.document(outgoing["id"])
                if image_data is not None:
                    self.image_bytes(persisted)
                return persisted
            except NotFound:
                if time.monotonic() >= deadline:
                    raise StorageUnavailable("Published document has not become durable") from None
                time.sleep(0.1)


class ImageWorker:
    def __init__(self, system, server):
        self.system, self.server = system, server

    def handle(self, data):
        if len(data) > 1024 * 1024:
            raise InvalidDocument("Broker event exceeds byte limit")
        try:
            notification = json.loads(data)
        except (ValueError, UnicodeDecodeError, RecursionError) as exc:
            raise InvalidDocument("Expected a core document broker event") from exc
        if not isinstance(notification, dict):
            raise InvalidDocument("Expected a core document broker event")
        if notification.get("dataset") != self.system.settings.dataset:
            return None
        dtype = notification.get("dtype")
        if dtype not in {"file", "image", "picture", "person", "relation"}:
            return None
        validate(notification, dtype, self.system.settings.dataset)
        # Fetch current state so out-of-order notifications cannot roll back a review.
        document = self.server.document(notification["id"])
        if document["dtype"] != dtype:
            raise InvalidDocument("Broker and persisted document types differ")
        if dtype in {"person", "relation"}:
            return self.sync_review(document)
        if document.get("extensions", {}).get("faceIntel", {}).get("faceCrop"):
            return None  # Derived crops must not recurse through their own emitted events.
        media = document.get("mediaType", "")
        if (
            dtype == "file"
            and media
            and not media.startswith("image/")
            and media != "application/octet-stream"
        ):
            return None
        data = self.server.image_bytes(document)
        try:
            original = self.system.request(
                "ingest-image-file", {"document": document, "data": data}
            )
        except InvalidDocument:
            if dtype == "file" and not media.startswith("image/"):
                return None  # A generic non-image file is outside this worker's capability.
            raise
        result = self.system.request("process-image", {"photoId": original["id"]})
        for output in result["documents"]:
            binary = None
            if output["dtype"] == "picture":
                binary, _ = self.system.request("photo-bytes", {"id": output["id"]})
            self.server.publish(output, binary)
        return result

    def sync_review(self, document):
        dtype, identifier = document["dtype"], document["id"]
        try:
            if dtype == "person":
                existing = self.system.request("get-person", {"id": identifier})["person"]
            else:
                existing = self.system.request("get-relation", {"id": identifier})
        except NotFound:
            return None  # This worker mirrors reviews only for its own graph.
        document = deepcopy(document)
        document["rev"] = existing["rev"]
        return self.system.request(
            "ingest-person" if dtype == "person" else "ingest-relation", document
        )


def run_worker(system, server, settings, connection_factory=None):
    import pika

    parameters = pika.URLParameters(settings.rabbit_url)
    parameters.heartbeat = 120
    parameters.blocked_connection_timeout = 30
    connection = (connection_factory or pika.BlockingConnection)(parameters)
    try:
        channel = connection.channel()
        channel.exchange_declare(exchange="documents", exchange_type="topic", durable=True)
        dead = settings.queue + ".dead"
        channel.exchange_declare(exchange=dead, exchange_type="fanout", durable=True)
        channel.queue_declare(queue=dead, durable=True)
        channel.queue_bind(queue=dead, exchange=dead)
        channel.queue_declare(
            queue=settings.queue, durable=True, arguments={"x-dead-letter-exchange": dead}
        )
        for key in [
            "documents.new.picture",
            "documents.new.image",
            "documents.new.file",
            "documents.updated.picture",
            "documents.updated.image",
            "documents.updated.file",
            "documents.updated.person",
            "documents.updated.relation",
        ]:
            channel.queue_bind(queue=settings.queue, exchange="documents", routing_key=key)
        channel.basic_qos(prefetch_count=1)
        worker = ImageWorker(system, server)

        def delivery(channel, method, properties, body):
            try:
                worker.handle(body)
            except (InvalidDocument, NotFound):
                channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            except Exception:
                # A native/model/store failure is not proof of successful durable delivery.
                channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                time.sleep(1)
            else:
                channel.basic_ack(delivery_tag=method.delivery_tag)

        channel.basic_consume(queue=settings.queue, on_message_callback=delivery, auto_ack=False)
        channel.start_consuming()
    finally:
        connection.close()


def main():
    service = Settings.from_env()
    settings = WorkerSettings.from_env()
    system = FaceIntelSystem(service)
    server = StarServerClient(settings, service)
    try:
        if not system.pipeline_profile:
            raise ValueError("Configure pinned YuNet and SFace models for the image worker")
        run_worker(system, server, settings)
    finally:
        server.close()
        system.close()
