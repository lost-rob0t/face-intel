"""Consume the pinned generated schema; do not define a competing StarIntel schema."""

import hashlib
import json
from functools import cache
from importlib.resources import files

from jsonschema import Draft202012Validator, FormatChecker

from .errors import InvalidDocument

VERSION = "0.10.1"
AUTHORITY = "org.starintel/core@1"
EXTENSION_AUTHORITY = "org.starintel/face-intel@1"
EXTENSION_TYPES = {
    "face-observation": "FaceObservation",
    "candidate-person": "CandidatePerson",
    "face-person-candidate": "FacePersonCandidate",
}
TYPES = {
    "person": "Person",
    "picture": "Picture",
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
    verify_extension_pin(pin)
    return pin


def verify_extension_pin(core_pin: dict) -> dict:
    root = files("face_intel.contracts.face_extension")
    pin = json.loads(root.joinpath("pin.json").read_text())
    if (
        pin["authorityLibrary"] != EXTENSION_AUTHORITY
        or pin["schemaVersion"] != VERSION
        or pin["releaseVersion"] != "0.1.0"
    ):
        raise RuntimeError("Unexpected face extension contract")
    for name, expected in pin["files"].items():
        if hashlib.sha256(root.joinpath(name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Face extension hash mismatch: {name}")
    release = json.loads(root.joinpath("release-lock.json").read_text())
    if (
        release["authorityLibrary"] != pin["authorityLibrary"]
        or release["schemaVersion"] != VERSION
        or release["releaseVersion"] != pin["releaseVersion"]
        or release["coreReleaseLockHash"] != core_pin["files"]["release-lock.json"]
        or release["imports"]
        != [
            {
                "kind": "import",
                "name": AUTHORITY,
                "version": VERSION,
                "digest": "sha256:" + core_pin["files"]["core.star"],
            }
        ]
    ):
        raise RuntimeError("Face extension core dependency mismatch")
    for name, expected in pin["files"].items():
        if name == "release-lock.json":
            continue
        locked = release["sources"].get(name, release["artifacts"].get(name))
        if expected != locked:
            raise RuntimeError(f"Face extension release lock mismatch: {name}")
    return pin


@cache
def validator(dtype: str) -> Draft202012Validator:
    verify_pin()
    package = (
        "face_intel.contracts.face_extension"
        if dtype in EXTENSION_TYPES
        else "face_intel.contracts"
    )
    types = EXTENSION_TYPES if dtype in EXTENSION_TYPES else TYPES
    schema = json.loads(files(package).joinpath("schema.json").read_text())
    return Draft202012Validator(
        {**schema, "$ref": f"#/$defs/{types[dtype]}"}, format_checker=FormatChecker()
    )


def validate(document: dict, dtype: str, dataset: str) -> None:
    if dtype not in TYPES and dtype not in EXTENSION_TYPES:
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
    if dtype in EXTENSION_TYPES:
        basis = document["annotationBasis"]
        if not 1 <= len(basis.strip()) <= 2000:
            raise InvalidDocument("An annotation basis of 1 to 2000 characters is required")
    if dtype in {"candidate-person", "face-person-candidate"}:
        if document.get("verificationStatus") != "candidate":
            raise InvalidDocument("Candidate documents require verificationStatus candidate")
    if dtype == "face-observation":
        if document["picture"]["schema"] != f"{AUTHORITY}/picture":
            raise InvalidDocument("A face observation requires a core Picture reference")
    if dtype == "face-person-candidate":
        if (
            document["source"]["schema"] != f"{EXTENSION_AUTHORITY}/face-observation"
            or document["destination"]["schema"]
            not in {
                f"{AUTHORITY}/person",
                f"{EXTENSION_AUTHORITY}/candidate-person",
            }
            or document["predicate"] != f"{EXTENSION_AUTHORITY}/candidate-person-for-face"
        ):
            raise InvalidDocument("Invalid face candidate relationship")


def reference(document: dict) -> dict[str, str]:
    authority = EXTENSION_AUTHORITY if document["dtype"] in EXTENSION_TYPES else AUTHORITY
    return {"schema": f"{authority}/{document['dtype']}", "id": document["id"]}
