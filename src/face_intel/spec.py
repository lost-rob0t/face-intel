"""Consume the pinned generated schema; do not define a competing StarIntel schema."""

import hashlib
import json
from functools import cache
from importlib.resources import files

from jsonschema import Draft202012Validator, FormatChecker

from .errors import InvalidDocument

VERSION = "0.10.1"
AUTHORITY = "org.starintel/core@1"
TYPES = {
    "person": "Person",
    "picture": "Picture",
    "image": "Image",
    "file": "File",
    "relation": "Relation",
    "target": "Target",
    "actor-manifest": "ActorManifest",
}


def verify_pin() -> dict:
    root = files("face_intel.contracts")
    pin = json.loads(root.joinpath("pin.json").read_text())
    if pin["schemaVersion"] != VERSION:
        raise RuntimeError("Unexpected pinned StarIntel version")
    for name, expected in pin["files"].items():
        if hashlib.sha256(root.joinpath(name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Pinned contract hash mismatch: {name}")
    release = json.loads(root.joinpath("release-lock.json").read_text())
    for name in ("schema.json", "starintel_types.py"):
        if pin["files"][name] != release["artifacts"][name]:
            raise RuntimeError(f"Upstream release lock mismatch: {name}")
    if pin["files"]["core.star"] != release["sources"]["core.star"]:
        raise RuntimeError("Upstream source lock mismatch")
    if release["schemaVersion"] != VERSION or release["canonicalKeyStyle"] != "lowerCamelCase":
        raise RuntimeError("Unexpected upstream release contract")
    return pin


@cache
def validator(dtype: str) -> Draft202012Validator:
    verify_pin()
    schema = json.loads(files("face_intel.contracts").joinpath("schema.json").read_text())
    return Draft202012Validator(
        {**schema, "$ref": f"#/$defs/{TYPES[dtype]}"}, format_checker=FormatChecker()
    )


def validate(document: dict, dtype: str, dataset: str) -> None:
    if dtype not in TYPES:
        raise InvalidDocument("Unsupported document type")
    if not isinstance(document, dict):
        raise InvalidDocument("A document must be an object")
    if document.get("dtype") != dtype or document.get("schemaVersion") != VERSION:
        raise InvalidDocument(f"Expected {dtype} with schemaVersion {VERSION}")
    if document.get("dataset") != dataset:
        raise InvalidDocument("Document dataset differs from this service dataset")
    errors = sorted(validator(dtype).iter_errors(document), key=lambda e: str(list(e.path)))
    if errors:
        path = ".".join(str(value) for value in errors[0].path) or "document"
        raise InvalidDocument(f"Invalid {path}: {errors[0].message}")


def reference(document: dict) -> dict[str, str]:
    return {"schema": f"{AUTHORITY}/{document['dtype']}", "id": document["id"]}
