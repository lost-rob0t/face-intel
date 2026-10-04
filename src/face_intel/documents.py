import hashlib
import io
import json
import unicodedata
import warnings
from copy import deepcopy

from PIL import Image, UnidentifiedImageError

from .config import Settings
from .contracts.starintel_types import Person, Picture, Relation
from .errors import InvalidDocument
from .spec import VERSION, reference, validate

LINK_PREDICATE = "org.starintel.face-intel/annotated-person@1"


def normalize_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def prepare_person(document: dict, dataset: str) -> Person:
    validate(document, "person", dataset)
    result = deepcopy(document)
    values = [result.get("fullName", ""), result.get("displayName", "")]
    values.append(" ".join(result.get(k, "") for k in ("fname", "mname", "lname")))
    values.extend(result.get("aliases", []))
    names = sorted({normalize_name(value) for value in values if value.strip()})
    extensions = result.setdefault("extensions", {})
    internal = extensions.setdefault("faceIntel", {})
    if not isinstance(internal, dict):
        raise InvalidDocument("extensions.faceIntel must be an object")
    internal["nameKeys"] = names
    validate(result, "person", dataset)
    return result


def picture_from_bytes(data: bytes, settings: Settings) -> tuple[Picture, str]:
    if not data or len(data) > settings.max_photo_bytes:
        raise InvalidDocument("Photo is empty or exceeds the byte limit")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {"PNG", "JPEG", "WEBP"}:
                    raise InvalidDocument("Supported photo formats: PNG, JPEG, WebP")
                width, height = image.size
                if width * height > settings.max_photo_pixels:
                    raise InvalidDocument("Photo exceeds the pixel limit")
                media_type = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}[
                    image.format
                ]
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                image.load()
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise InvalidDocument("Invalid or oversized image") from exc
    digest = hashlib.sha256(data).hexdigest()
    document: Picture = {
        "id": f"picture:sha256:{digest}",
        "dataset": settings.dataset,
        "dtype": "picture",
        "schemaVersion": VERSION,
        "bytesHash": digest,
        "bytesHashAlgorithm": "sha256",
        "sizeBytes": len(data),
        "mediaType": media_type,
        "width": width,
        "height": height,
        "storageId": "original",
        "pictureKind": "photo",
    }
    validate(document, "picture", settings.dataset)
    return document, media_type


def annotated_link(picture: dict, person: dict, basis: str, dataset: str) -> Relation:
    if not isinstance(basis, str) or not 1 <= len(basis.strip()) <= 2000:
        raise InvalidDocument("An annotation basis of 1 to 2000 characters is required")
    identity = json.dumps([dataset, picture["id"], person["id"], basis.strip()], ensure_ascii=False)
    digest = hashlib.sha256(identity.encode()).hexdigest()
    document: Relation = {
        "id": f"relation:sha256:{digest}",
        "dataset": dataset,
        "dtype": "relation",
        "schemaVersion": VERSION,
        "source": reference(picture),
        "destination": reference(person),
        "predicate": LINK_PREDICATE,
        "verificationStatus": "candidate",
        "provenance": {"method": "explicit-annotation", "basis": basis.strip()},
        "extensions": {"faceIntel": {"status": "candidate", "candidate": True}},
    }
    validate(document, "relation", dataset)
    return document
