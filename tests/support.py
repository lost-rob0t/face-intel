"""Explicit test-only store; production has no memory-store configuration."""

import io
import threading
from copy import deepcopy

from PIL import Image

from face_intel.couch import content
from face_intel.documents import LINK_PREDICATE
from face_intel.errors import Conflict, NotFound

TOKEN = "fixture-token-for-tests-only-00000000"


def person(identifier="person:fixture", name="Fixture Person", **fields):
    return {
        "id": identifier,
        "dataset": "face-intel",
        "dtype": "person",
        "schemaVersion": "0.10.1",
        "fullName": name,
        **fields,
    }


def photo_bytes(width=3, height=4):
    image = Image.new("RGB", (width, height), color=(50, 70, 90))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def candidate_person(identifier="person:candidate-fixture", **fields):
    return person(identifier=identifier, verificationStatus="candidate", **fields)


class MemoryStore:
    def __init__(self):
        self.documents = {}
        self.attachments = {}
        self.owner = None
        self.closed = False
        self.writes = 0

    def assert_owner(self):
        assert self.owner == threading.get_ident(), "Store escaped its owning actor"

    def initialize(self):
        self.owner = threading.get_ident()

    def close(self):
        self.assert_owner()
        self.closed = True

    def get(self, identifier, dtype):
        self.assert_owner()
        document = self.documents.get(identifier)
        if document is None or document["dtype"] != dtype:
            raise NotFound(identifier)
        return deepcopy(document)

    def put(self, document, attachment=None):
        self.assert_owner()
        existing = self.documents.get(document["id"])
        if existing:
            if document.get("rev") and document["rev"] != existing["rev"]:
                raise Conflict("Stale document revision")
            if content(existing) == content(document):
                return deepcopy(existing)
            if not document.get("rev"):
                raise Conflict("Changing an existing document requires its rev")
        elif document.get("rev"):
            raise Conflict("New document cannot have a rev")
        self.writes += 1
        value = {**content(document), "rev": f"{self.writes}-fixture"}
        self.documents[value["id"]] = deepcopy(value)
        if attachment:
            self.attachments[value["id"]] = attachment
        return deepcopy(value)

    def page(self, view, value, limit, after=None):
        self.assert_owner()
        if view == "by_name":
            rows = [
                d
                for d in self.documents.values()
                if d["dtype"] == "person"
                and value in d.get("extensions", {}).get("faceIntel", {}).get("nameKeys", [])
            ]
        elif view == "photo_links":
            rows = [
                d
                for d in self.documents.values()
                if d["dtype"] == "relation"
                and d["predicate"] == LINK_PREDICATE
                and d["source"]["schema"] == "org.starintel/core@1/picture"
                and d["destination"]["schema"] == "org.starintel/core@1/person"
                and d["destination"]["id"] == value
            ]
        elif view == "picture_people":
            rows = [
                d
                for d in self.documents.values()
                if d["dtype"] == "relation"
                and d["predicate"] == LINK_PREDICATE
                and d["source"]["schema"] == "org.starintel/core@1/picture"
                and d["destination"]["schema"] == "org.starintel/core@1/person"
                and d["source"]["id"] == value
            ]
        elif view == "face_gallery":
            rows = [
                d
                for d in self.documents.values()
                if d["dtype"] == "picture"
                and d.get("extensions", {}).get("faceIntel", {}).get("faceCrop")
                and value in d["extensions"]["faceIntel"].get("embeddings", {})
            ]
        else:
            raise AssertionError("Unknown test view")
        rows = sorted((d for d in rows if not d.get("deleted")), key=lambda d: d["id"])
        if after:
            rows = [d for d in rows if d["id"] > after]
        selected = rows[:limit]
        return {
            "documents": deepcopy(selected),
            "nextCursor": selected[-1]["id"] if len(rows) > limit else None,
        }

    def photo_bytes(self, identifier):
        self.assert_owner()
        self.get(identifier, "picture")
        return self.attachments[identifier]
