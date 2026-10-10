import hashlib
import io
import math
import os
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path

import pykka
from fastapi.testclient import TestClient
from PIL import Image
from support import TOKEN, MemoryStore, photo_bytes

from face_intel.actors import Command, FaceIntelSystem
from face_intel.api import create_app
from face_intel.config import Settings
from face_intel.errors import InvalidDocument, SimilarityUnavailable
from face_intel.similarity import (
    SFaceEmbedder,
    aligned_crop,
    cosine,
    similarity_options,
    unit_vector,
)


def crop(color):
    data = io.BytesIO()
    Image.new("RGB", (112, 112), color).save(data, "PNG")
    return data.getvalue()


class FixtureEmbedder:
    model_id = "fixture:rgb-v1"

    def __init__(self):
        self.owner = threading.get_ident()
        self.calls = 0

    def embed(self, image):
        assert threading.get_ident() == self.owner
        self.calls += 1
        return image.getpixel((0, 0))


class SimilarityMathTests(unittest.TestCase):
    def test_cosine_scale_sign_and_orthogonality(self):
        self.assertAlmostEqual(cosine([3, 4], [6, 8]), 1)
        self.assertAlmostEqual(cosine([3, 4], [-6, -8]), -1)
        self.assertEqual(cosine([1, 0], [0, 1]), 0)
        self.assertTrue(math.isfinite(cosine([1e308, 1], [1e308, 1])))

    def test_invalid_embeddings(self):
        for vector in [[], [0, 0], [math.nan], [math.inf], ["bad"], [1] * 4097]:
            with self.subTest(vector=str(vector)[:30]), self.assertRaises(SimilarityUnavailable):
                unit_vector(vector)
        with self.assertRaises(SimilarityUnavailable):
            cosine([1, 0], [1])

    def test_bounded_options(self):
        base = {"queryPhotoId": "picture:q", "candidatePhotoIds": ["picture:c"]}
        self.assertEqual(similarity_options(base)["limit"], 20)
        for patch in [
            {"queryPhotoId": " "},
            {"queryPhotoId": "x" * 513},
            {"candidatePhotoIds": []},
            {"candidatePhotoIds": ["x"] * 101},
            {"candidatePhotoIds": ["x", "x"]},
            {"candidatePhotoIds": [None]},
            {"limit": True},
            {"limit": 0},
            {"limit": 101},
            {"extra": 1},
        ]:
            with self.subTest(patch=patch), self.assertRaises(InvalidDocument):
                similarity_options({**base, **patch})

    def test_crop_validation(self):
        settings = Settings(api_token=TOKEN)
        with aligned_crop(crop((1, 2, 3)), settings) as image:
            self.assertEqual(image.mode, "RGB")
        for data in [b"bad", photo_bytes(), crop((1, 2, 3))]:
            limit = replace(settings, max_photo_bytes=1) if data == crop((1, 2, 3)) else settings
            with self.assertRaises(InvalidDocument):
                aligned_crop(data, limit)
        data = io.BytesIO()
        image = Image.new("RGB", (112, 112))
        exif = Image.Exif()
        exif[274] = 6
        image.save(data, "JPEG", exif=exif)
        with self.assertRaises(InvalidDocument):
            aligned_crop(data.getvalue(), settings)

    def test_model_configuration_requires_digest(self):
        for fields in [
            {"sface_model_path": "model.onnx"},
            {"sface_model_sha256": "a" * 64},
            {"sface_model_path": "m", "sface_model_sha256": "bad"},
        ]:
            with self.assertRaises(ValueError):
                Settings(api_token=TOKEN, **fields)


class SimilarityServiceTests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore()
        self.engine = None

        def factory():
            self.engine = FixtureEmbedder()
            return self.engine

        self.client = TestClient(create_app(Settings(api_token=TOKEN), self.store, factory))
        self.client.__enter__()
        self.headers = {"Authorization": "Bearer " + TOKEN}
        self.ids = [self.upload(crop(c)) for c in [(255, 0, 0), (128, 0, 0), (0, 255, 0)]]
        self.payload = {"queryPhotoId": self.ids[0], "candidatePhotoIds": self.ids}

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.assertTrue(self.store.closed)

    def upload(self, data):
        response = self.client.post("/v1/photos", content=data, headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["id"]

    def search(self, payload=None):
        return self.client.post(
            "/v1/search/face-similarity", json=payload or self.payload, headers=self.headers
        )

    def test_ranking_ties_limits_and_read_only(self):
        writes = self.store.writes
        response = self.search({**self.payload, "limit": 2})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["modelId"], "fixture:rgb-v1")
        self.assertEqual(result["metric"], "cosine")
        self.assertEqual(result["candidateCount"], 3)
        self.assertEqual([m["picture"]["id"] for m in result["matches"]], sorted(self.ids[:2]))
        self.assertEqual([m["score"] for m in result["matches"]], [1, 1])
        self.assertEqual(self.engine.calls, 3)  # Query reused as candidate, one inference per ID.
        self.assertEqual(self.store.writes, writes)
        self.assertIn(
            "face.similarity.search",
            self.client.get("/v1/manifest", headers=self.headers).json()["capabilities"],
        )

    def test_http_target_parity(self):
        expected = self.search().json()
        target = {
            "id": "target:similarity",
            "dataset": "face-intel",
            "dtype": "target",
            "schemaVersion": "0.10.1",
            "actor": "face-intel",
            "target": self.ids[0],
            "options": {"operation": "search-similar-faces", "candidatePhotoIds": self.ids},
        }
        response = self.client.post("/v1/targets", json=target, headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["target"]["extensions"]["faceIntel"]["similarity"], expected)
        self.assertEqual(result["documents"], [])
        self.assertEqual(result["target"]["state"], "completed")
        target["options"]["extra"] = True
        self.assertEqual(
            self.client.post("/v1/targets", json=target, headers=self.headers).status_code, 422
        )

    def test_auth_errors_and_incompatible_images(self):
        self.assertEqual(
            self.client.post("/v1/search/face-similarity", json=self.payload).status_code, 401
        )
        self.assertEqual(self.search({**self.payload, "limit": True}).status_code, 422)
        self.assertEqual(
            self.search({**self.payload, "queryPhotoId": "picture:missing"}).status_code, 404
        )
        small = self.upload(photo_bytes())
        self.assertEqual(self.search({**self.payload, "queryPhotoId": small}).status_code, 422)
        zero = self.upload(crop((0, 0, 0)))
        response = self.search({**self.payload, "queryPhotoId": zero})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Facial similarity engine is unavailable")

    def test_expired_command_does_not_run_inference(self):
        system = self.client.app.state.system
        with self.assertRaises(pykka.Timeout):
            system.similarity.ask(
                Command("search-similar-faces", self.payload, time.monotonic() - 1)
            )
        self.assertEqual(self.engine.calls, 0)


class SimilarityLifecycleTests(unittest.TestCase):
    def test_disabled_engine(self):
        with TestClient(create_app(Settings(api_token=TOKEN), MemoryStore())) as client:
            headers = {"Authorization": "Bearer " + TOKEN}
            self.assertNotIn(
                "face.similarity.search",
                client.get("/v1/manifest", headers=headers).json()["capabilities"],
            )
            self.assertEqual(
                client.post(
                    "/v1/search/face-similarity",
                    headers=headers,
                    json={"queryPhotoId": "q", "candidatePhotoIds": ["c"]},
                ).status_code,
                503,
            )

    def test_initialization_failure_cleans_up_actors(self):
        def fail():
            raise SimilarityUnavailable("Fixture failure")

        store = MemoryStore()
        before = set(pykka.ActorRegistry.get_all())
        with self.assertRaises(SimilarityUnavailable):
            FaceIntelSystem(Settings(api_token=TOKEN), store, fail)
        self.assertTrue(store.closed)
        self.assertEqual(set(pykka.ActorRegistry.get_all()), before)

    def test_elapsed_inference_starts_no_further_candidate_work(self):
        class SlowEmbedder(FixtureEmbedder):
            def embed(self, image):
                vector = super().embed(image)
                time.sleep(0.03)
                return vector

        engine = []

        def factory():
            engine.append(SlowEmbedder())
            return engine[-1]

        system = FaceIntelSystem(Settings(api_token=TOKEN), MemoryStore(), factory)
        try:
            query = system.request("ingest-photo", crop((255, 0, 0)))
            other = system.request("ingest-photo", crop((0, 255, 0)))
            payload = {"queryPhotoId": query["id"], "candidatePhotoIds": [other["id"]]}
            with self.assertRaises(pykka.Timeout):
                system.similarity.ask(
                    Command("search-similar-faces", payload, time.monotonic() + 0.01), timeout=1
                )
            self.assertEqual(engine[0].calls, 1)
        finally:
            system.close()


@unittest.skipUnless(os.environ.get("FACE_INTEL_TEST_SFACE_MODEL"), "Opt-in real SFace model test")
class RealModelTests(unittest.TestCase):
    def test_real_inference_matches_opencv_reference(self):
        import cv2
        import numpy as np

        path = os.environ["FACE_INTEL_TEST_SFACE_MODEL"]
        digest = os.environ["FACE_INTEL_TEST_SFACE_SHA256"]
        model = SFaceEmbedder(path, digest)
        # Spatial and color variation detects channel/preprocessing differences.
        rgb = np.random.default_rng(73).integers(0, 256, (112, 112, 3), dtype=np.uint8)
        image = Image.fromarray(rgb)
        actual = model.embed(image)
        reference = cv2.FaceRecognizerSF.create(path, "").feature(rgb[:, :, ::-1].copy())
        self.assertEqual(len(actual), 128)
        np.testing.assert_allclose(actual, unit_vector(reference.reshape(-1)), atol=1e-6)
        self.assertAlmostEqual(cosine(actual, model.embed(image)), 1)
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "model.onnx"
            invalid.write_bytes(b"not a model")
            with self.assertRaises(SimilarityUnavailable):
                SFaceEmbedder(str(invalid), hashlib.sha256(b"not a model").hexdigest())
        with self.assertRaises(SimilarityUnavailable):
            SFaceEmbedder(path, "0" * 64)

        settings = Settings(api_token=TOKEN, sface_model_path=path, sface_model_sha256=digest)
        with TestClient(create_app(settings, MemoryStore())) as client:
            headers = {"Authorization": "Bearer " + TOKEN}
            data = io.BytesIO()
            image.save(data, "PNG")
            picture = client.post("/v1/photos", content=data.getvalue(), headers=headers).json()
            response = client.post(
                "/v1/search/face-similarity",
                json={"queryPhotoId": picture["id"], "candidatePhotoIds": [picture["id"]]},
                headers=headers,
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["modelId"], "opencv-sface:sha256:" + digest)
            self.assertAlmostEqual(response.json()["matches"][0]["score"], 1)
