"""Bounded Pykka actors; one store actor owns all CouchDB state and its HTTP client."""

import base64
import binascii
import hashlib
import queue
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import pykka

from .config import Settings
from .couch import CouchStore
from .documents import annotated_link, normalize_name, picture_from_bytes, prepare_person
from .errors import InvalidDocument, SimilarityUnavailable
from .face_documents import check_face_bounds, face_candidate
from .similarity import SFaceEmbedder, aligned_crop, cosine, similarity_options, unit_vector
from .spec import VERSION, reference, validate, verify_pin


@dataclass(frozen=True, slots=True)
class Command:
    operation: str
    payload: Any
    deadline: float | None = None


def remaining(deadline: float | None, fallback: float) -> float:
    seconds = fallback if deadline is None else deadline - time.monotonic()
    if seconds <= 0:
        raise pykka.Timeout("Command deadline elapsed")
    return seconds


class BoundedInbox(queue.Queue):
    def put(self, item, block=True, timeout=None):
        # Pykka's default queue has no capacity or admission timeout.
        super().put(item, block=block, timeout=0.25 if block and timeout is None else timeout)


class BoundedActor(pykka.ThreadingActor):
    @staticmethod
    def _create_actor_inbox():
        return BoundedInbox(maxsize=64)


class StoreActor(BoundedActor):
    def __init__(self, store):
        super().__init__()
        self.store = store

    def on_receive(self, message: Command):
        remaining(message.deadline, 30)
        if message.operation == "initialize":
            return self.store.initialize()
        if message.operation == "get":
            return self.store.get(**message.payload)
        if message.operation == "put":
            return self.store.put(**message.payload)
        if message.operation == "page":
            return self.store.page(**message.payload)
        if message.operation == "photo-bytes":
            return self.store.photo_bytes(**message.payload)
        raise InvalidDocument("Unsupported store operation")

    def on_stop(self):
        self.store.close()


class PersonActor(BoundedActor):
    def __init__(self, store, settings: Settings):
        super().__init__()
        self.store = store
        self.settings = settings

    def ask_store(self, operation: str, **payload):
        return self.store.ask(
            Command(operation, payload, self.deadline),
            timeout=remaining(self.deadline, self.settings.request_timeout),
        )

    def on_receive(self, message: Command):
        remaining(message.deadline, self.settings.request_timeout)
        self.deadline = message.deadline
        if message.operation == "ingest-person":
            validate(message.payload, "person", self.settings.dataset)
            document = prepare_person(message.payload, self.settings.dataset)
            return self.ask_store("put", document=document)
        if message.operation == "get-person-record":
            return self.ask_store("get", identifier=message.payload["id"], dtype="person")
        if message.operation == "search-name":
            name = message.payload["name"]
            if not isinstance(name, str) or not 1 <= len(name.strip()) <= 256:
                raise InvalidDocument("Name query must contain 1 to 256 characters")
            return self.ask_store(
                "page",
                view="by_name",
                value=normalize_name(name),
                limit=message.payload["limit"],
                after=message.payload.get("after"),
            )
        if message.operation == "get-person":
            identifier = message.payload["id"]
            person = self.ask_store("get", identifier=identifier, dtype="person")
            page = self.ask_store(
                "page",
                view="photo_links",
                value=identifier,
                limit=message.payload["limit"],
                after=message.payload.get("after"),
            )
            pictures = {}
            for relation in page["documents"]:
                picture = self.ask_store(
                    "get", identifier=relation["source"]["id"], dtype="picture"
                )
                pictures[picture["id"]] = picture
            return {
                "person": person,
                "photos": list(pictures.values()),
                "relations": page["documents"],
                "nextCursor": page["nextCursor"],
            }
        raise InvalidDocument("Unsupported person operation")


class PhotoActor(BoundedActor):
    def __init__(self, store, settings: Settings):
        super().__init__()
        self.store = store
        self.settings = settings

    def ask_store(self, operation: str, **payload):
        return self.store.ask(
            Command(operation, payload, self.deadline),
            timeout=remaining(self.deadline, self.settings.request_timeout),
        )

    def on_receive(self, message: Command):
        remaining(message.deadline, self.settings.request_timeout)
        self.deadline = message.deadline
        if message.operation == "ingest-photo":
            document, media_type = picture_from_bytes(message.payload, self.settings)
            return self.ask_store(
                "put", document=document, attachment=(message.payload, media_type)
            )
        if message.operation == "get-photo":
            return self.ask_store("get", identifier=message.payload["id"], dtype="picture")
        if message.operation == "photo-bytes":
            return self.ask_store("photo-bytes", identifier=message.payload["id"])
        if message.operation == "link-photo-person":
            payload = message.payload
            if not all(
                isinstance(payload.get(key), str) and payload[key].strip()
                for key in ("photoId", "personId", "basis")
            ):
                raise InvalidDocument("Expected photoId, personId and annotation basis")
            picture = self.ask_store("get", identifier=payload["photoId"], dtype="picture")
            person = self.ask_store("get", identifier=payload["personId"], dtype="person")
            document = annotated_link(picture, person, payload["basis"], self.settings.dataset)
            return self.ask_store("put", document=document)
        raise InvalidDocument("Unsupported photo operation")


class FaceActor(BoundedActor):
    def __init__(self, store, settings: Settings):
        super().__init__()
        self.store = store
        self.settings = settings

    def ask_store(self, operation: str, **payload):
        return self.store.ask(
            Command(operation, payload, self.deadline),
            timeout=remaining(self.deadline, self.settings.request_timeout),
        )

    def on_receive(self, message: Command):
        remaining(message.deadline, self.settings.request_timeout)
        self.deadline = message.deadline
        operation, payload = message.operation, message.payload
        if operation == "ingest-face-observation":
            validate(payload, "face-observation", self.settings.dataset)
            picture = self.ask_store("get", identifier=payload["picture"]["id"], dtype="picture")
            check_face_bounds(payload, picture)
            return self.ask_store("put", document=payload)
        if operation == "ingest-candidate-person":
            validate(payload, "candidate-person", self.settings.dataset)
            return self.ask_store("put", document=prepare_person(payload, self.settings.dataset))
        if operation == "get-candidate-person":
            return self.ask_store("get", identifier=payload["id"], dtype="candidate-person")
        if operation == "get-face-record":
            return self.ask_store("get", identifier=payload["id"], dtype="face-observation")
        if operation == "link-face-person":
            if not all(
                isinstance(payload.get(key), str) and payload[key].strip()
                for key in ("faceId", "personId", "basis")
            ):
                raise InvalidDocument("Expected faceId, personId and annotation basis")
            dtype = payload.get("personType", "person")
            if not isinstance(dtype, str) or dtype not in {"person", "candidate-person"}:
                raise InvalidDocument("personType must be person or candidate-person")
            face = self.ask_store("get", identifier=payload["faceId"], dtype="face-observation")
            person = self.ask_store("get", identifier=payload["personId"], dtype=dtype)
            relation = face_candidate(face, person, payload["basis"], self.settings.dataset)
            return self.ask_store("put", document=relation)
        if operation == "get-face":
            face = self.ask_store("get", identifier=payload["id"], dtype="face-observation")
            picture = self.ask_store("get", identifier=face["picture"]["id"], dtype="picture")
            page = self.ask_store(
                "page",
                view="face_candidates",
                value=face["id"],
                limit=payload["limit"],
                after=payload.get("after"),
            )
            persons = {}
            for relation in page["documents"]:
                destination = relation["destination"]
                dtype = destination["schema"].rsplit("/", 1)[1]
                person = self.ask_store("get", identifier=destination["id"], dtype=dtype)
                persons[person["id"]] = person
            return {
                "face": face,
                "picture": picture,
                "persons": list(persons.values()),
                "candidates": page["documents"],
                "nextCursor": page["nextCursor"],
            }
        raise InvalidDocument("Unsupported face operation")


def page_options(payload: dict) -> dict:
    limit = payload.get("limit", 20)
    if type(limit) is not int or not 1 <= limit <= 100:
        raise InvalidDocument("limit must be an integer between 1 and 100")
    after = payload.get("after")
    if after is not None and (not isinstance(after, str) or not 1 <= len(after) <= 512):
        raise InvalidDocument("after must be a document ID of at most 512 characters")
    return {**payload, "limit": limit}


class SimilarityActor(BoundedActor):
    def __init__(self, store, settings: Settings, embedder_factory=None):
        super().__init__()
        self.store, self.settings = store, settings
        self.factory = embedder_factory
        self.embedder = None

    def on_receive(self, message: Command):
        remaining(message.deadline, self.settings.request_timeout)
        if message.operation == "initialize":
            if self.factory is not None:
                self.embedder = self.factory()
            elif self.settings.sface_model_path:
                self.embedder = SFaceEmbedder(
                    self.settings.sface_model_path, self.settings.sface_model_sha256
                )
            if self.embedder is not None and (
                not isinstance(self.embedder.model_id, str)
                or not 1 <= len(self.embedder.model_id) <= 256
            ):
                raise SimilarityUnavailable("Invalid embedding model identity")
            return (
                {"modelId": self.embedder.model_id, "metric": "cosine", "cropSize": [112, 112]}
                if self.embedder is not None
                else None
            )
        if message.operation != "search-similar-faces":
            raise InvalidDocument("Unsupported similarity operation")
        payload = similarity_options(message.payload)
        if self.embedder is None:
            raise SimilarityUnavailable("Facial similarity is not configured")

        def ask_store(operation, **fields):
            return self.store.ask(
                Command(operation, fields, message.deadline),
                timeout=remaining(message.deadline, self.settings.request_timeout),
            )

        embeddings, pictures = {}, {}
        for identifier in [payload["queryPhotoId"], *payload["candidatePhotoIds"]]:
            if identifier in embeddings:
                continue
            remaining(message.deadline, self.settings.request_timeout)
            pictures[identifier] = ask_store("get", identifier=identifier, dtype="picture")
            data, _ = ask_store("photo-bytes", identifier=identifier)
            with aligned_crop(data, self.settings) as image:
                remaining(message.deadline, self.settings.request_timeout)
                embeddings[identifier] = unit_vector(self.embedder.embed(image))
            remaining(message.deadline, self.settings.request_timeout)
        query = embeddings[payload["queryPhotoId"]]
        matches = [
            {
                "picture": reference(pictures[identifier]),
                "score": cosine(query, embeddings[identifier]),
            }
            for identifier in payload["candidatePhotoIds"]
        ]
        matches.sort(key=lambda item: (-item["score"], item["picture"]["id"]))
        return {
            "query": reference(pictures[payload["queryPhotoId"]]),
            "modelId": self.embedder.model_id,
            "metric": "cosine",
            "matches": matches[: payload["limit"]],
            "candidateCount": len(matches),
        }


class TargetActor(BoundedActor):
    def __init__(self, persons, photos, faces, similarity, settings: Settings):
        super().__init__()
        self.persons = persons
        self.photos = photos
        self.faces = faces
        self.similarity = similarity
        self.settings = settings

    def forward(self, actor, operation: str, payload, deadline):
        return actor.ask(
            Command(operation, payload, deadline),
            timeout=remaining(deadline, self.settings.request_timeout),
        )

    def on_receive(self, message: Command):
        remaining(message.deadline, self.settings.request_timeout)
        target = message.payload
        validate(target, "target", self.settings.dataset)
        if target["actor"] != "face-intel":
            raise InvalidDocument("Target must address face-intel")
        options = target.get("options", {})
        extension = target.get("extensions", {}).get("faceIntel", {})
        if not isinstance(extension, dict):
            raise InvalidDocument("extensions.faceIntel must be an object")
        operation = options.get("operation")
        if operation == "search-similar-faces":
            if set(options) - {"operation", "candidatePhotoIds", "limit"}:
                raise InvalidDocument("Unexpected similarity Target options")
            result = self.forward(
                self.similarity,
                operation,
                {
                    "queryPhotoId": target["target"],
                    "candidatePhotoIds": options.get("candidatePhotoIds"),
                    "limit": options.get("limit", 20),
                },
                message.deadline,
            )
            documents = []
        elif operation in {"ingest-face-observation", "ingest-candidate-person"}:
            document = options.get("document")
            if not isinstance(document, dict) or document.get("id") != target["target"]:
                raise InvalidDocument("options.document.id must match target")
            result = self.forward(self.faces, operation, document, message.deadline)
            documents = [result]
        elif operation == "get-candidate-person":
            result = self.forward(self.faces, operation, {"id": target["target"]}, message.deadline)
            documents = [result]
        elif operation == "get-face":
            result = self.forward(
                self.faces,
                operation,
                page_options({**options, "id": target["target"]}),
                message.deadline,
            )
            documents = [
                result["face"],
                result["picture"],
                *result["persons"],
                *result["candidates"],
            ]
        elif operation == "link-face-person":
            result = self.forward(
                self.faces,
                operation,
                {
                    "faceId": target["target"],
                    "personId": options.get("personId"),
                    "personType": options.get("personType", "person"),
                    "basis": options.get("basis"),
                },
                message.deadline,
            )
            face = self.forward(
                self.faces,
                "get-face-record",
                {"id": target["target"]},
                message.deadline,
            )
            picture = self.forward(
                self.photos,
                "get-photo",
                {"id": face["picture"]["id"]},
                message.deadline,
            )
            destination_type = result["destination"]["schema"].rsplit("/", 1)[1]
            person = self.forward(
                self.faces if destination_type == "candidate-person" else self.persons,
                "get-candidate-person"
                if destination_type == "candidate-person"
                else "get-person-record",
                {"id": result["destination"]["id"]},
                message.deadline,
            )
            documents = [face, picture, person, result]
        elif operation == "search-name":
            result = self.persons.ask(
                Command(
                    operation,
                    page_options(
                        {
                            **options,
                            "name": target["target"],
                        }
                    ),
                    message.deadline,
                ),
                timeout=remaining(message.deadline, self.settings.request_timeout),
            )
            documents = result["documents"]
        elif operation == "get-person":
            result = self.persons.ask(
                Command(
                    operation,
                    page_options(
                        {
                            **options,
                            "id": target["target"],
                        }
                    ),
                    message.deadline,
                ),
                timeout=remaining(message.deadline, self.settings.request_timeout),
            )
            documents = [result["person"], *result["photos"], *result["relations"]]
        elif operation == "get-photo":
            result = self.photos.ask(
                Command(operation, {"id": target["target"]}, message.deadline),
                timeout=remaining(message.deadline, self.settings.request_timeout),
            )
            documents = [result]
        elif operation == "ingest-person":
            document = options.get("document")
            if not isinstance(document, dict) or document.get("id") != target["target"]:
                raise InvalidDocument("options.document.id must match target")
            result = self.forward(self.persons, operation, document, message.deadline)
            documents = [result]
        elif operation == "ingest-photo":
            encoded = options.get("photoBase64")
            if not isinstance(encoded, str) or len(encoded) > self.settings.max_photo_bytes * 2:
                raise InvalidDocument("options.photoBase64 must contain a bounded base64 image")
            try:
                data = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise InvalidDocument("Invalid base64 photo") from exc
            identifier = "picture:sha256:" + hashlib.sha256(data).hexdigest()
            if identifier != target["target"]:
                raise InvalidDocument("Photo content hash ID must match target")
            result = self.forward(self.photos, operation, data, message.deadline)
            documents = [result]
        elif operation == "link-photo-person":
            result = self.forward(
                self.photos,
                operation,
                {
                    "photoId": target["target"],
                    "personId": options.get("personId"),
                    "basis": options.get("basis"),
                },
                message.deadline,
            )
            person = self.forward(
                self.persons,
                "get-person-record",
                {"id": result["destination"]["id"]},
                message.deadline,
            )
            picture = self.forward(
                self.photos,
                "get-photo",
                {
                    "id": result["source"]["id"],
                },
                message.deadline,
            )
            documents = [person, picture, result]
        else:
            raise InvalidDocument("Unsupported Target operation")
        completed = deepcopy(target)
        completed["state"] = "completed"
        extension = completed.setdefault("extensions", {}).setdefault("faceIntel", {})
        if not isinstance(extension, dict):
            raise InvalidDocument("extensions.faceIntel must be an object")
        extension["resultRefs"] = [reference(document) for document in documents]
        if operation == "search-similar-faces":
            extension["similarity"] = result
        validate(completed, "target", self.settings.dataset)
        return {"target": completed, "documents": documents, "nextCursor": result.get("nextCursor")}


class FaceIntelSystem:
    def __init__(self, settings: Settings, store=None, embedder_factory=None):
        verify_pin()
        self.settings = settings
        self.refs = []
        self.store = StoreActor.start(store or CouchStore(settings))
        self.refs.append(self.store)
        try:
            self.store.ask(Command("initialize", {}), timeout=settings.request_timeout)
            self.persons = PersonActor.start(self.store, settings)
            self.refs.append(self.persons)
            self.photos = PhotoActor.start(self.store, settings)
            self.refs.append(self.photos)
            self.faces = FaceActor.start(self.store, settings)
            self.refs.append(self.faces)
            self.similarity = SimilarityActor.start(self.store, settings, embedder_factory)
            self.refs.append(self.similarity)
            self.similarity_profile = self.similarity.ask(
                Command("initialize", {}), timeout=settings.request_timeout
            )
            self.targets = TargetActor.start(
                self.persons, self.photos, self.faces, self.similarity, settings
            )
            self.refs.append(self.targets)
        except BaseException:
            self.close()
            raise

    def close(self):
        for actor in reversed(self.refs):
            actor.stop(timeout=self.settings.request_timeout)
        self.refs.clear()

    def request(self, operation: str, payload):
        # Copy caller-owned mappings before transferring ownership into a mailbox.
        payload = deepcopy(payload)
        if operation in {"search-name", "get-person", "get-face"}:
            payload = page_options(payload)
        if operation in {"ingest-person", "search-name", "get-person"}:
            actor = self.persons
        elif operation in {"ingest-photo", "get-photo", "photo-bytes", "link-photo-person"}:
            actor = self.photos
        elif operation in {
            "ingest-face-observation",
            "ingest-candidate-person",
            "get-face",
            "get-candidate-person",
            "link-face-person",
        }:
            actor = self.faces
        elif operation == "search-similar-faces":
            payload = similarity_options(payload)
            actor = self.similarity
        elif operation == "execute-target":
            actor = self.targets
        else:
            raise InvalidDocument("Unsupported public operation")
        deadline = time.monotonic() + self.settings.request_timeout
        return actor.ask(
            Command(operation, payload, deadline), timeout=self.settings.request_timeout
        )

    def manifest(self) -> dict:
        manifest = {
            "id": "actor:face-intel",
            "dataset": self.settings.dataset,
            "dtype": "actor-manifest",
            "schemaVersion": VERSION,
            "actor": "face-intel",
            "actorVersion": "0.1.0",
            "runtime": "python-pykka",
            "capabilities": [
                "person.ingest",
                "person.search-by-name",
                "person.get",
                "photo.ingest",
                "photo.get",
                "person.photo.annotate",
                "target.execute",
                "face.observation.ingest",
                "face.observation.get",
                "candidate-person.ingest",
                "candidate-person.get",
                "face.person.annotate",
            ],
            "accepts": ["person", "picture", "target", "face-observation", "candidate-person"],
            "produces": [
                "person",
                "picture",
                "relation",
                "target",
                "face-observation",
                "candidate-person",
                "face-person-candidate",
            ],
        }
        if self.similarity_profile:
            manifest["capabilities"].append("face.similarity.search")
            manifest["extensions"] = {
                "faceIntel": {"similarity": deepcopy(self.similarity_profile)}
            }
        validate(manifest, "actor-manifest", self.settings.dataset)
        return manifest
