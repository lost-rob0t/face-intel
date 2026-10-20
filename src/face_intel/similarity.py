"""Similarity of supplied aligned crops. Scores are not identity probabilities."""

import hashlib
import io
import math
from collections.abc import Sequence
from typing import Protocol

from PIL import Image

from .config import Settings
from .documents import picture_from_bytes
from .errors import InvalidDocument, SimilarityUnavailable


class Embedder(Protocol):
    model_id: str

    def embed(self, image: Image.Image) -> Sequence[float]: ...


def unit_vector(values: Sequence[float]) -> tuple[float, ...]:
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SimilarityUnavailable("Invalid model embedding") from exc
    if not 1 <= len(vector) <= 4096 or not all(math.isfinite(v) for v in vector):
        raise SimilarityUnavailable("Invalid model embedding")
    norm = math.hypot(*vector)
    if not math.isfinite(norm) or norm == 0:
        raise SimilarityUnavailable("Invalid model embedding")
    return tuple(value / norm for value in vector)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left, right = unit_vector(left), unit_vector(right)
    if len(left) != len(right):
        raise SimilarityUnavailable("Incompatible model embeddings")
    return max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(left, right, strict=True))))


def aligned_crop(data: bytes, settings: Settings) -> Image.Image:
    document, _ = picture_from_bytes(data, settings)
    if (document["width"], document["height"]) != (112, 112):
        raise InvalidDocument("Expected an already aligned 112x112 face crop")
    with Image.open(io.BytesIO(data)) as image:
        # Crop coordinates use stored pixels, so do not apply EXIF transforms here.
        if image.getexif().get(274, 1) != 1:
            raise InvalidDocument("Aligned crops must have normalized EXIF orientation")
        if getattr(image, "n_frames", 1) != 1:
            raise InvalidDocument("Aligned crops must contain a single frame")
        return image.convert("RGB")


class SFaceEmbedder:
    """CPU OpenCV SFace; constructed and used exclusively by the similarity actor."""

    def __init__(self, path: str, digest: str):
        try:
            import cv2
            import numpy as np

            # Load the verified bytes, rather than reopening a potentially changed file.
            with open(path, "rb") as stream:
                data = stream.read(100 * 1024 * 1024 + 1)
            if len(data) > 100 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != digest:
                raise SimilarityUnavailable("SFace model hash mismatch or size limit exceeded")
            self.net = cv2.dnn.readNetFromONNX(np.frombuffer(data, dtype=np.uint8))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except SimilarityUnavailable:
            raise
        except Exception as exc:
            raise SimilarityUnavailable("Unable to load configured SFace model") from exc
        self.cv2, self.np = cv2, np
        self.model_id = "opencv-sface:sha256:" + digest

    def embed(self, image: Image.Image) -> Sequence[float]:
        try:
            bgr = self.np.asarray(image)[:, :, ::-1].copy()
            # Match FaceRecognizerSF: BGR pixels, unit scale, zero mean, swap to RGB.
            blob = self.cv2.dnn.blobFromImage(bgr, 1.0, (112, 112), (0, 0, 0), True, False)
            self.net.setInput(blob)
            vector = self.net.forward().reshape(-1)
            if vector.size != 128:
                raise SimilarityUnavailable("Expected a 128-dimensional SFace embedding")
            return unit_vector(vector)
        except SimilarityUnavailable:
            raise
        except Exception as exc:
            raise SimilarityUnavailable("SFace inference failed") from exc


def similarity_options(payload: dict) -> dict:
    if not isinstance(payload, dict) or set(payload) - {
        "queryPhotoId",
        "candidatePhotoIds",
        "limit",
    }:
        raise InvalidDocument("Expected queryPhotoId, candidatePhotoIds and optional limit")
    query = payload.get("queryPhotoId")
    candidates = payload.get("candidatePhotoIds")

    def valid_id(value):
        return isinstance(value, str) and 1 <= len(value) <= 512 and bool(value.strip())

    if not valid_id(query):
        raise InvalidDocument(
            "queryPhotoId must be a nonempty Picture ID of at most 512 characters"
        )
    if (
        not isinstance(candidates, list)
        or not 1 <= len(candidates) <= 100
        or not all(valid_id(value) for value in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        raise InvalidDocument("candidatePhotoIds must contain 1 to 100 distinct Picture IDs")
    limit = payload.get("limit", 20)
    if type(limit) is not int or not 1 <= limit <= 100:
        raise InvalidDocument("limit must be an integer between 1 and 100")
    return {"queryPhotoId": query, "candidatePhotoIds": candidates, "limit": limit}
