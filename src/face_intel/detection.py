"""Verified CPU YuNet detection and five-landmark SFace alignment."""

import hashlib
import math
from dataclasses import dataclass

from PIL import Image

from .config import Settings
from .errors import InvalidDocument, SimilarityUnavailable
from .similarity import SFaceEmbedder


def model_bytes(path: str, digest: str) -> bytes:
    with open(path, "rb") as stream:
        data = stream.read(100 * 1024 * 1024 + 1)
    if len(data) > 100 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != digest:
        raise SimilarityUnavailable("Face model hash mismatch or size limit exceeded")
    return data


@dataclass
class DetectedFace:
    image: Image.Image
    region: dict
    landmarks: list
    score: float


class FaceProcessor:
    def __init__(self, settings: Settings):
        try:
            import cv2
            import numpy as np

            detector = np.frombuffer(
                model_bytes(settings.yunet_model_path, settings.yunet_model_sha256), dtype=np.uint8
            )
            recognizer = np.frombuffer(
                model_bytes(settings.sface_model_path, settings.sface_model_sha256), dtype=np.uint8
            )
            empty = np.array([], dtype=np.uint8)
            self.detector = cv2.FaceDetectorYN.create(
                "onnx", detector, empty, (320, 320), 0.9, 0.3, 5000
            )
            self.aligner = cv2.FaceRecognizerSF.create("onnx", recognizer, empty)
            self.embedder = SFaceEmbedder(settings.sface_model_path, settings.sface_model_sha256)
            self.cv2, self.np = cv2, np
            self.detector_id = "opencv-yunet:sha256:" + settings.yunet_model_sha256
            self.model_id = self.embedder.model_id
            self.max_faces = settings.max_detected_faces
        except SimilarityUnavailable:
            raise
        except Exception as exc:
            raise SimilarityUnavailable("Unable to initialize face extraction models") from exc

    def detect(self, image: Image.Image) -> list[DetectedFace]:
        try:
            # Bound detector memory independently of the original image pixel limit.
            scale = min(1.0, 1600 / max(image.size))
            rgb = self.np.asarray(image.convert("RGB"))
            bgr = rgb[:, :, ::-1].copy()
            small = self.cv2.resize(bgr, None, fx=scale, fy=scale) if scale < 1 else bgr
            self.detector.setInputSize((small.shape[1], small.shape[0]))
            _, rows = self.detector.detect(small)
            if rows is None:
                return []
            if len(rows) > self.max_faces:
                raise InvalidDocument("Image exceeds the configured face count limit")
            rows = sorted(rows, key=lambda row: (float(row[1]), float(row[0])))
            faces = []
            for detection in rows:
                row = detection.copy()
                row[:14] /= scale
                if not self.np.isfinite(row).all():
                    raise SimilarityUnavailable("Detector returned invalid coordinates")
                x = max(0, math.floor(float(row[0])))
                y = max(0, math.floor(float(row[1])))
                right = min(image.width, math.ceil(float(row[0] + row[2])))
                bottom = min(image.height, math.ceil(float(row[1] + row[3])))
                if right <= x or bottom <= y:
                    continue
                aligned = self.aligner.alignCrop(bgr, row)
                faces.append(
                    DetectedFace(
                        Image.fromarray(self.cv2.cvtColor(aligned, self.cv2.COLOR_BGR2RGB)),
                        {"x": x, "y": y, "width": right - x, "height": bottom - y},
                        row[4:14].reshape(5, 2).tolist(),
                        float(row[14]),
                    )
                )
            return faces
        except (InvalidDocument, SimilarityUnavailable):
            raise
        except Exception as exc:
            raise SimilarityUnavailable("Face detection or alignment failed") from exc

    def embed(self, image):
        return self.embedder.embed(image)
