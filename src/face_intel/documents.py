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

LINK_PREDICATE = "org.starintel/core@1/related-to"


def normalize_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def prepare_person(document: dict, dataset: str) -> Person:
    dtype = document.get("dtype") if isinstance(document, dict) else None
    if dtype != "person":
        raise InvalidDocument("Expected core Person")
    validate(document, dtype, dataset)
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
    internal["candidate"] = result.get("verificationStatus") == "candidate"
    internal.pop("status", None)
    if "verificationStatus" in result:
        internal["status"] = result["verificationStatus"]
    validate(result, dtype, dataset)
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


def prepare_relation(document: dict, endpoints: dict, dataset: str) -> Relation:
    """Apply image association rules on top of the unchanged core Relation schema."""
    validate(document, "relation", dataset)
    result = deepcopy(document)
    source, destination = endpoints["source"], endpoints["destination"]
    if (
        result["predicate"] == LINK_PREDICATE
        and source["dtype"] == "picture"
        and destination["dtype"] == "person"
    ):
        status = result.get("verificationStatus")
        if status not in {"candidate", "confirmed", "rejected"}:
            raise InvalidDocument(
                "Image associations require candidate, confirmed or rejected status"
            )
        provenance = result.get("provenance", {})
        basis = provenance.get("basis")
        if not isinstance(basis, str) or not 1 <= len(basis.strip()) <= 2000:
            raise InvalidDocument(
                "Image associations require provenance.basis of 1 to 2000 characters"
            )
        internal = result.setdefault("extensions", {}).setdefault("faceIntel", {})
        if not isinstance(internal, dict):
            raise InvalidDocument("extensions.faceIntel must be an object")
        internal["candidate"] = status == "candidate"
        internal["status"] = status
        region = internal.get("region")
        if region is not None:
            if (
                not isinstance(region, dict)
                or set(region) != {"x", "y", "width", "height"}
                or any(type(region[key]) is not int for key in region)
                or region["x"] < 0
                or region["y"] < 0
                or region["width"] <= 0
                or region["height"] <= 0
            ):
                raise InvalidDocument("Expected a pixel rectangle with x, y, width and height")
            if (
                any(
                    type(source.get(key)) is not int or source[key] <= 0
                    for key in ("width", "height")
                )
                or region["x"] + region["width"] > source["width"]
                or region["y"] + region["height"] > source["height"]
            ):
                raise InvalidDocument("Image association rectangle exceeds the Picture dimensions")
    validate(result, "relation", dataset)
    return result
