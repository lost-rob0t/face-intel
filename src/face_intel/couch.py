"""CouchDB persistence and derived exact-name/annotation views."""

import base64
import json
from copy import deepcopy
from urllib.parse import quote

import httpx

from .config import Settings
from .documents import LINK_PREDICATE
from .errors import Conflict, NotFound, StorageUnavailable
from .spec import validate

VIEW_NAMES = """function(doc) {
  if (doc.dtype !== 'person') return;
  if (doc.deleted || !doc.extensions) return;
  var state = doc.extensions.faceIntel;
  if (!state || !Array.isArray(state.nameKeys)) return;
  state.nameKeys.forEach(function(name) { emit([doc.dataset, name], null); });
}"""
VIEW_LINKS = """function(doc) {
  if (doc.dtype !== 'relation' || doc.deleted || !doc.destination) return;
  if (doc.predicate !== 'LINK_PREDICATE' || !doc.source) return;
  if (doc.source.schema !== 'org.starintel/core@1/picture' ||
      doc.destination.schema !== 'org.starintel/core@1/person') return;
  emit([doc.dataset, doc.destination.id], null);
}""".replace("LINK_PREDICATE", LINK_PREDICATE)

VIEW_PICTURE_PEOPLE = VIEW_LINKS.replace("doc.destination.id", "doc.source.id")
VIEW_FACE_GALLERY = """function(doc) {
  if (doc.dtype !== 'picture' || doc.deleted || !doc.extensions) return;
  var face = doc.extensions.faceIntel;
  if (!face || !face.faceCrop || !face.embeddings) return;
  Object.keys(face.embeddings).forEach(function(model) { emit([doc.dataset, model], null); });
}"""


def public_document(raw: dict) -> dict:
    result = {key: deepcopy(value) for key, value in raw.items() if not key.startswith("_")}
    if "_rev" in raw:
        result["rev"] = raw["_rev"]
    return result


def content(document: dict) -> dict:
    return {key: value for key, value in document.items() if key != "rev"}


class CouchStore:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self.settings = settings
        auth = (
            (settings.couch_username, settings.couch_password) if settings.couch_username else None
        )
        self.client = httpx.Client(
            base_url=settings.couch_url.rstrip("/") + "/",
            auth=auth,
            timeout=settings.couch_timeout,
            transport=transport,
            follow_redirects=False,
        )
        self.database = quote(settings.couch_database, safe="")

    def close(self) -> None:
        self.client.close()

    def request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            response = self.client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise StorageUnavailable("CouchDB request failed") from exc
        if response.status_code == 404:
            raise NotFound("Document or database not found")
        if response.status_code == 409:
            raise Conflict("CouchDB revision conflict")
        if not response.is_success:
            raise StorageUnavailable("CouchDB rejected the request")
        return response

    def initialize(self) -> None:
        try:
            response = self.client.put(self.database)
        except httpx.HTTPError as exc:
            raise StorageUnavailable("Unable to initialize CouchDB database") from exc
        if response.status_code not in {201, 202, 412}:
            raise StorageUnavailable("Unable to initialize CouchDB database")
        path = f"{self.database}/_design/face-intel"
        views = {
            "by_name": {"map": VIEW_NAMES},
            "photo_links": {"map": VIEW_LINKS},
            "picture_people": {"map": VIEW_PICTURE_PEOPLE},
            "face_gallery": {"map": VIEW_FACE_GALLERY},
        }
        for _ in range(3):
            try:
                current = self.request("GET", path).json()
            except NotFound:
                current = {"_id": "_design/face-intel"}
            if "face_candidates" not in current.get("views", {}) and all(
                current.get("views", {}).get(name) == value for name, value in views.items()
            ):
                return
            desired = {
                **current,
                "language": "javascript",
                "views": {
                    **{k: v for k, v in current.get("views", {}).items() if k != "face_candidates"},
                    **views,
                },
            }
            try:
                self.request("PUT", path, json=desired)
                return
            except Conflict:
                continue
        raise Conflict("Unable to install views after three concurrent updates")

    def path(self, identifier: str) -> str:
        # Encode the entire ID: namespaced IDs may themselves contain slash characters.
        return f"{self.database}/{quote(identifier, safe='')}"

    def raw_get(self, identifier: str) -> dict:
        return self.request("GET", self.path(identifier)).json()

    def get(self, identifier: str, dtype: str) -> dict:
        document = public_document(self.raw_get(identifier))
        if document.get("dtype") != dtype:
            raise NotFound("Record has a different document type")
        validate(document, dtype, self.settings.dataset)
        return document

    def put(self, document: dict, attachment: tuple[bytes, str] | None = None) -> dict:
        validate(document, document["dtype"], self.settings.dataset)
        desired = content(document)
        try:
            existing_raw = self.raw_get(document["id"])
            existing = public_document(existing_raw)
        except NotFound:
            existing_raw = None
            existing = None
        if existing is not None:
            validate(existing, document["dtype"], self.settings.dataset)
            expected = document.get("rev")
            if expected is not None and expected != existing["rev"]:
                raise Conflict("Stale document revision")
            attachment_present = attachment is None or "original" in existing_raw.get(
                "_attachments", {}
            )
            if content(existing) == desired and attachment_present:
                return existing
            if expected is None:
                raise Conflict("Changing an existing document requires its rev")
        elif document.get("rev"):
            raise Conflict("A new document cannot have a revision precondition")

        raw = {**deepcopy(desired), "_id": document["id"]}
        if existing is not None:
            raw["_rev"] = existing["rev"]
            if "_attachments" in existing_raw:
                raw["_attachments"] = existing_raw["_attachments"]
        if attachment is not None:
            data, media_type = attachment
            raw["_attachments"] = {
                "original": {
                    "content_type": media_type,
                    "data": base64.b64encode(data).decode("ascii"),
                }
            }
        try:
            result = self.request("PUT", self.path(document["id"]), json=raw).json()
        except Conflict:
            # Retry only a byte-for-byte identical create; never overwrite a competing edit.
            if document.get("rev") is None:
                latest_raw = self.raw_get(document["id"])
                latest = public_document(latest_raw)
                present = attachment is None or "original" in latest_raw.get("_attachments", {})
                if content(latest) == desired and present:
                    return latest
            raise
        return {**desired, "rev": result["rev"]}

    def page(self, view: str, value: str, limit: int, after: str | None = None) -> dict:
        key = json.dumps([self.settings.dataset, value], ensure_ascii=False)
        params = {"key": key, "include_docs": "true", "limit": str(limit + 2)}
        if after:
            params.update(startkey=key, endkey=key, startkey_docid=after)
            del params["key"]
        response = self.request(
            "GET", f"{self.database}/_design/face-intel/_view/{view}", params=params
        ).json()
        rows = response["rows"]
        if after and rows and rows[0]["id"] == after:
            rows = rows[1:]
        page = rows[:limit]
        documents = [public_document(row["doc"]) for row in page]
        for document in documents:
            dtype = (
                "person"
                if view == "by_name"
                else "picture"
                if view == "face_gallery"
                else "relation"
            )
            validate(document, dtype, self.settings.dataset)
        return {"documents": documents, "nextCursor": page[-1]["id"] if len(rows) > limit else None}

    def photo_bytes(self, identifier: str) -> tuple[bytes, str]:
        picture = self.get(identifier, "picture")
        response = self.request("GET", self.path(identifier) + "/original")
        return response.content, picture["mediaType"]
