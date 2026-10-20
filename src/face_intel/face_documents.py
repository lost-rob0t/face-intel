"""Supplied observations and explicit candidate annotations using generated types."""

import hashlib
import json

from .contracts.face_extension.face_intel_types import FaceObservation, FacePersonCandidate
from .errors import InvalidDocument
from .spec import EXTENSION_AUTHORITY, VERSION, reference, validate

FACE_CANDIDATE_PREDICATE = f"{EXTENSION_AUTHORITY}/candidate-person-for-face"


def check_face_bounds(document: FaceObservation, picture: dict) -> None:
    if any(type(picture.get(key)) is not int or picture[key] <= 0 for key in ("width", "height")):
        raise InvalidDocument("The stored Picture requires positive pixel dimensions")
    if (
        document["x"] + document["width"] > picture["width"]
        or document["y"] + document["height"] > picture["height"]
    ):
        raise InvalidDocument("Face rectangle exceeds the Picture dimensions")


def face_candidate(face: dict, person: dict, basis: str, dataset: str) -> FacePersonCandidate:
    if not isinstance(basis, str) or not 1 <= len(basis.strip()) <= 2000:
        raise InvalidDocument("An annotation basis of 1 to 2000 characters is required")
    source, destination = reference(face), reference(person)
    identity = json.dumps([dataset, source, destination, basis.strip()], sort_keys=True)
    digest = hashlib.sha256(identity.encode()).hexdigest()
    document: FacePersonCandidate = {
        "id": f"face-person-candidate:sha256:{digest}",
        "dataset": dataset,
        "dtype": "face-person-candidate",
        "schemaVersion": VERSION,
        "source": source,
        "destination": destination,
        "predicate": FACE_CANDIDATE_PREDICATE,
        "candidateStatus": "candidate",
        "verificationStatus": "candidate",
        "annotationBasis": basis.strip(),
        "provenance": {"method": "explicit-annotation", "basis": basis.strip()},
        "extensions": {"faceIntel": {"candidate": True}},
    }
    validate(document, "face-person-candidate", dataset)
    return document
