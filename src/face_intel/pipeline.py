"""Core-document incoming image graph; no automatic identity confirmation."""

import hashlib
import io
import json
import math
from copy import deepcopy

from PIL import Image

from .documents import LINK_PREDICATE, picture_from_bytes, prepare_person, prepare_relation
from .errors import InvalidDocument, NotFound, SimilarityUnavailable
from .similarity import aligned_crop, cosine, unit_vector
from .spec import AUTHORITY, VERSION, reference, validate


def pipeline_options(payload: dict) -> dict:
    if not isinstance(payload, dict) or set(payload) - {"photoId", "minScore", "limit"}:
        raise InvalidDocument("Expected photoId, minScore and limit")
    identifier = payload.get("photoId")
    if not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 512:
        raise InvalidDocument("photoId must be a bounded Picture ID")
    score = payload.get("minScore", 0.5)
    if type(score) not in {int, float} or not math.isfinite(score) or not -1 <= score <= 1:
        raise InvalidDocument("minScore must be a finite cosine score between -1 and 1")
    limit = payload.get("limit", 5)
    if type(limit) is not int or not 1 <= limit <= 20:
        raise InvalidDocument("limit must be an integer between 1 and 20")
    return {"photoId": identifier, "minScore": float(score), "limit": limit}


def stable_id(kind, values):
    digest = hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"{kind}:sha256:{digest}"


class ImagePipeline:
    """Owned by PipelineActor. Every storage operation uses its propagated deadline."""

    def __init__(self, processor, settings):
        self.processor, self.settings = processor, settings

    def run(self, payload, store, check):
        options = pipeline_options(payload)
        if self.processor is None:
            raise SimilarityUnavailable("Incoming image processing is not configured")
        model, detector = self.processor.model_id, self.processor.detector_id
        run_id = stable_id("target", [options, model, detector])
        try:
            completed = store("get", identifier=run_id, dtype="target")
            if completed["state"] == "completed":
                return self.result(completed, store)
        except NotFound:
            pass
        picture = store("get", identifier=options["photoId"], dtype="picture")
        data, _ = store("photo-bytes", identifier=picture["id"])
        picture_from_bytes(data, self.settings)
        with Image.open(io.BytesIO(data)) as image:
            if image.getexif().get(274, 1) != 1 or getattr(image, "n_frames", 1) != 1:
                raise InvalidDocument("Incoming images require normalized EXIF and a single frame")
            check()
            faces = self.processor.detect(image.convert("RGB"))
        check()
        if len(faces) > self.settings.max_detected_faces:
            for face in faces:
                face.image.close()
            raise InvalidDocument("Image exceeds the configured face count limit")
        gallery, truncated = self.gallery(model, store, check)
        output = {picture["id"]: picture}
        for face in faces:
            check()
            try:
                vector = unit_vector(self.processor.embed(face.image))
                buffer = io.BytesIO()
                face.image.save(buffer, "PNG")
                crop_data = buffer.getvalue()
            finally:
                face.image.close()
            check()
            crop, media_type = picture_from_bytes(crop_data, self.settings)
            if (crop["width"], crop["height"]) != (112, 112):
                raise SimilarityUnavailable("Face processor must return aligned 112x112 crops")
            crop = self.embed_picture(
                crop, vector, model, store, (crop_data, media_type), picture["id"]
            )
            output[crop["id"]] = crop
            derivation = {
                "id": stable_id("relation", [picture["id"], crop["id"], detector, face.region]),
                "dataset": self.settings.dataset,
                "dtype": "relation",
                "schemaVersion": VERSION,
                "source": reference(crop),
                "destination": reference(picture),
                "predicate": AUTHORITY + "/derived-from",
                "provenance": {
                    "method": "face-detection-alignment",
                    "detectorId": detector,
                    "region": face.region,
                    "landmarks": face.landmarks,
                    "detectionScore": face.score,
                },
            }
            derivation = self.put_once(derivation, store)
            output[derivation["id"]] = derivation
            choices = {}
            ranked = []
            for known in gallery:
                if (
                    known.get("extensions", {}).get("faceIntel", {}).get("sourcePhotoId")
                    == picture["id"]
                ):
                    continue
                score = cosine(vector, known["extensions"]["faceIntel"]["embeddings"][model])
                if score >= options["minScore"]:
                    ranked.append((score, known))
            ranked.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
            for score, known in ranked[: options["limit"]]:
                check()
                links = store("page", view="picture_people", value=known["id"], limit=100)
                if links["nextCursor"] is not None:
                    truncated = True
                for link in links["documents"]:
                    if link.get("verificationStatus") not in {"candidate", "confirmed"}:
                        continue
                    person = store("get", identifier=link["destination"]["id"], dtype="person")
                    if person.get("verificationStatus") == "rejected":
                        continue
                    choices.setdefault(person["id"], (person, score, known, link))
            if not choices:
                person = prepare_person(
                    {
                        "id": stable_id("person", [crop["id"]]),
                        "dataset": self.settings.dataset,
                        "dtype": "person",
                        "schemaVersion": VERSION,
                        "verificationStatus": "candidate",
                        "sources": [reference(picture), reference(crop)],
                        "provenance": {
                            "method": "face-detection-candidate",
                            "detectorId": detector,
                        },
                    },
                    self.settings.dataset,
                )
                person = self.put_once(person, store)
                choices[person["id"]] = (person, None, None, None)
            for person, score, known, link in choices.values():
                check()
                output[person["id"]] = person
                relation = self.candidate(crop, person, model, score, known, link)
                relation = prepare_relation(
                    relation, {"source": crop, "destination": person}, self.settings.dataset
                )
                relation = self.put_once(relation, store)
                output[relation["id"]] = relation
        target = {
            "id": run_id,
            "dataset": self.settings.dataset,
            "dtype": "target",
            "schemaVersion": VERSION,
            "actor": "face-intel",
            "target": picture["id"],
            "state": "completed",
            "options": {
                "operation": "process-image",
                "minScore": options["minScore"],
                "limit": options["limit"],
            },
            "extensions": {
                "faceIntel": {
                    "resultRefs": [reference(d) for d in output.values()],
                    "faceCount": len(faces),
                    "galleryTruncated": truncated,
                    "modelId": model,
                    "detectorId": detector,
                }
            },
        }
        check()
        completed = self.put_once(target, store)
        return self.result(completed, store)

    def result(self, target, store):
        state = target["extensions"]["faceIntel"]
        documents = [
            store("get", identifier=ref["id"], dtype=ref["schema"].rsplit("/", 1)[1])
            for ref in state["resultRefs"]
        ]
        return {
            "target": target,
            "documents": documents,
            "faceCount": state["faceCount"],
            "galleryTruncated": state["galleryTruncated"],
            "nextCursor": None,
        }

    def gallery(self, model, store, check):
        documents, after = [], None
        while True:
            check()
            size = min(100, self.settings.max_gallery_candidates - len(documents))
            page = store("page", view="face_gallery", value=model, limit=size, after=after)
            documents.extend(page["documents"])
            after = page["nextCursor"]
            if after is None or len(documents) >= self.settings.max_gallery_candidates:
                return documents, after is not None

    def embed_picture(self, document, vector, model, store, attachment=None, source=None):
        try:
            existing = store("get", identifier=document["id"], dtype="picture")
            document = deepcopy(existing)
        except NotFound:
            pass
        state = document.setdefault("extensions", {}).setdefault("faceIntel", {})
        if not isinstance(state, dict):
            raise InvalidDocument("extensions.faceIntel must be an object")
        embeddings = state.setdefault("embeddings", {})
        if not isinstance(embeddings, dict):
            raise InvalidDocument("extensions.faceIntel.embeddings must be an object")
        embeddings[model] = list(vector)
        state["faceCrop"] = True
        if source is not None:
            state.setdefault("sourcePhotoId", source)
        validate(document, "picture", self.settings.dataset)
        return store("put", document=document, attachment=attachment)

    def put_once(self, document, store):
        try:
            return store("get", identifier=document["id"], dtype=document["dtype"])
        except NotFound:
            return store("put", document=document)

    def candidate(self, crop, person, model, score, known, link):
        evidence = [reference(crop)]
        if known is not None:
            evidence += [reference(known), reference(link)]
        return {
            "id": stable_id("relation", [crop["id"], person["id"], model]),
            "dataset": self.settings.dataset,
            "dtype": "relation",
            "schemaVersion": VERSION,
            "source": reference(crop),
            "destination": reference(person),
            "predicate": LINK_PREDICATE,
            "verificationStatus": "candidate",
            "evidence": evidence,
            "provenance": {
                "method": "face-similarity-candidate" if known else "face-detection-candidate",
                "basis": "Model suggestion; requires independent identity review",
                "modelId": model,
            },
            "extensions": {
                "faceIntel": {"similarity": {"metric": "cosine", "score": score, "modelId": model}}
            },
        }

    def register(self, payload, store, check):
        if not isinstance(payload, dict) or set(payload) != {"photoId", "personId", "basis"}:
            raise InvalidDocument("Expected photoId, personId and basis")
        if self.processor is None:
            raise SimilarityUnavailable("Incoming image processing is not configured")
        for key in ("photoId", "personId", "basis"):
            if (
                not isinstance(payload[key], str)
                or not payload[key].strip()
                or len(payload[key]) > 2000
            ):
                raise InvalidDocument("Expected bounded nonempty registration fields")
        picture = store("get", identifier=payload["photoId"], dtype="picture")
        person = store("get", identifier=payload["personId"], dtype="person")
        data, _ = store("photo-bytes", identifier=picture["id"])
        with aligned_crop(data, self.settings) as image:
            check()
            vector = unit_vector(self.processor.embed(image))
        check()
        picture = self.embed_picture(picture, vector, self.processor.model_id, store)
        relation = self.candidate(picture, person, self.processor.model_id, None, None, None)
        relation["provenance"] = {
            "method": "supplied-gallery-registration",
            "basis": payload["basis"],
            "modelId": self.processor.model_id,
        }
        relation = prepare_relation(
            relation, {"source": picture, "destination": person}, self.settings.dataset
        )
        relation = self.put_once(relation, store)
        return {"documents": [person, picture, relation], "nextCursor": None}
