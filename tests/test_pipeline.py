"""Incoming images -> core candidate graph, replay, gallery and later review."""

import hashlib
import io
import os
import threading
import unittest
from dataclasses import replace

from fastapi.testclient import TestClient
from PIL import Image
from support import TOKEN, MemoryStore, person

from face_intel.api import create_app
from face_intel.config import Settings
from face_intel.detection import DetectedFace
from face_intel.documents import LINK_PREDICATE
from face_intel.errors import StorageUnavailable
from face_intel.spec import reference, validate


class FixtureProcessor:
    model_id = "fixture:rgb-v1"
    detector_id = "fixture:one-region-v1"

    def __init__(self):
        self.owner = threading.get_ident()
        self.detections = 0

    def detect(self, image):
        assert threading.get_ident() == self.owner
        self.detections += 1
        pixel = image.getpixel((0, 0))
        if pixel == (0, 0, 0):
            return []
        return [
            DetectedFace(
                Image.new("RGB", (112, 112), pixel),
                {"x": 0, "y": 0, "width": 2, "height": 2},
                [[0, 0]] * 5,
                0.99,
            )
        ]

    def embed(self, image):
        assert threading.get_ident() == self.owner
        return image.getpixel((0, 0))


def colored_image(color, size=(200, 200)):
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, "PNG")
    return stream.getvalue()


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore()
        self.processor = None
        self.settings = Settings(api_token=TOKEN)
        self.open_client()

    def open_client(self):
        def factory():
            self.processor = FixtureProcessor()
            return self.processor

        self.client = TestClient(create_app(self.settings, self.store, processor_factory=factory))
        self.client.__enter__()
        self.headers = {"Authorization": "Bearer " + TOKEN}

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def post(self, path, payload):
        return self.client.post(path, json=payload, headers=self.headers)

    def upload(self, color=(255, 0, 0), size=(200, 200)):
        response = self.client.post(
            "/v1/photos", content=colored_image(color, size), headers=self.headers
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def process(self, photo, **options):
        return self.post("/v1/images/process", {"photoId": photo["id"], **options})

    def test_new_image_emits_core_graph_and_deduplicates(self):
        original = self.upload()
        response = self.process(original)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["faceCount"], 1)
        self.assertFalse(result["galleryTruncated"])
        docs = result["documents"]
        self.assertEqual(
            [d["dtype"] for d in docs], ["picture", "picture", "relation", "person", "relation"]
        )
        for document in [*docs, result["target"]]:
            validate(document, document["dtype"], "face-intel")
        crop, derivation, tentative, claim = docs[1:]
        self.assertEqual((crop["width"], crop["height"]), (112, 112))
        self.assertEqual(derivation["predicate"], "org.starintel/core@1/derived-from")
        self.assertEqual(derivation["destination"], reference(original))
        self.assertNotIn("fullName", tentative)
        self.assertEqual(tentative["verificationStatus"], "candidate")
        self.assertEqual(claim["destination"], reference(tentative))
        self.assertEqual(claim["verificationStatus"], "candidate")
        writes = self.store.writes
        self.assertEqual(self.process(original).json(), result)
        self.assertEqual(self.store.writes, writes)
        self.assertEqual(self.processor.detections, 1)
        self.assertEqual(
            self.client.get("/v1/photo-bytes/" + crop["id"], headers=self.headers).content,
            colored_image((255, 0, 0), (112, 112)),
        )

    def test_known_gallery_match_reuses_person_never_confirms(self):
        record = self.post("/v1/persons", person(verificationStatus="confirmed")).json()
        known = self.upload((128, 0, 0), (112, 112))
        response = self.post(
            "/v1/faces/register",
            {
                "photoId": known["id"],
                "personId": record["id"],
                "basis": "Supplied known image for review",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        before = dict(self.store.documents[record["id"]])
        result = self.process(self.upload()).json()
        people = [d for d in result["documents"] if d["dtype"] == "person"]
        self.assertEqual(people, [before])
        claims = [
            d
            for d in result["documents"]
            if d["dtype"] == "relation" and d["predicate"] == LINK_PREDICATE
        ]
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["verificationStatus"], "candidate")
        self.assertAlmostEqual(claims[0]["extensions"]["faceIntel"]["similarity"]["score"], 1)
        self.assertEqual(self.store.documents[record["id"]], before)

    def test_later_images_match_previous_candidates_and_replay_preserves_review(self):
        first = self.process(self.upload((255, 0, 0))).json()
        previous = next(d for d in first["documents"] if d["dtype"] == "person")
        second_image = self.upload((128, 0, 0))
        second = self.process(second_image).json()
        self.assertEqual(
            next(d for d in second["documents"] if d["dtype"] == "person")["id"], previous["id"]
        )
        claim = next(
            d
            for d in second["documents"]
            if d["dtype"] == "relation" and d["predicate"] == LINK_PREDICATE
        )
        confirmed = self.post(
            "/v1/relations",
            {
                **claim,
                "verificationStatus": "confirmed",
                "verifiedBy": "actor:review-fixture",
                "verifiedAt": 1791648000,
                "provenance": {
                    "method": "review",
                    "basis": "Independent supplied identity evidence",
                },
            },
        )
        self.assertEqual(confirmed.status_code, 200, confirmed.text)
        replay = self.process(second_image).json()
        replay_claim = next(d for d in replay["documents"] if d["id"] == claim["id"])
        self.assertEqual(replay_claim, confirmed.json())
        self.assertEqual(self.store.documents[previous["id"]]["verificationStatus"], "candidate")

    def test_no_faces_is_completed_and_replayable(self):
        original = self.upload((0, 0, 0))
        result = self.process(original).json()
        self.assertEqual(result["faceCount"], 0)
        self.assertEqual(result["documents"], [original])
        self.assertEqual(result["target"]["state"], "completed")
        self.assertEqual(self.process(original).json(), result)

    def test_partial_failure_retry_recovers_without_duplicate_people(self):
        class FailOnceStore(MemoryStore):
            failed = False

            def put(self, document, attachment=None):
                if (
                    document["dtype"] == "relation"
                    and document["predicate"] == LINK_PREDICATE
                    and not self.failed
                ):
                    self.failed = True
                    raise StorageUnavailable("Fixture interrupted graph write")
                return super().put(document, attachment)

        self.client.__exit__(None, None, None)
        self.store = FailOnceStore()
        self.open_client()
        original = self.upload()
        self.assertEqual(self.process(original).status_code, 503)
        self.assertFalse(any(d["dtype"] == "target" for d in self.store.documents.values()))
        retry = self.process(original)
        self.assertEqual(retry.status_code, 200, retry.text)
        self.assertEqual(
            len([d for d in self.store.documents.values() if d["dtype"] == "person"]), 1
        )

    def test_truncated_gallery_is_reported(self):
        self.client.__exit__(None, None, None)
        self.settings = replace(self.settings, max_gallery_candidates=1)
        self.open_client()
        for color in [(255, 0, 0), (0, 255, 0)]:
            self.process(self.upload(color))
        result = self.process(self.upload((0, 0, 255))).json()
        self.assertTrue(result["galleryTruncated"])

    def test_http_target_parity_and_validation(self):
        original = self.upload()
        target = {
            "id": "target:incoming",
            "dataset": "face-intel",
            "dtype": "target",
            "schemaVersion": "0.10.1",
            "actor": "face-intel",
            "target": original["id"],
            "options": {"operation": "process-image"},
        }
        response = self.post("/v1/targets", target)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["documents"], self.process(original).json()["documents"])
        self.assertEqual(
            self.client.post("/v1/images/process", json={"photoId": original["id"]}).status_code,
            401,
        )
        for options in [
            {"minScore": True},
            {"minScore": 2},
            {"limit": True},
            {"limit": 100},
            {"extra": 1},
        ]:
            self.assertEqual(self.process(original, **options).status_code, 422)
        self.assertEqual(
            self.post(
                "/v1/faces/register", {"photoId": original["id"], "personId": "p", "basis": "x"}
            ).status_code,
            404,
        )

    def test_restart_durable_run_and_gallery(self):
        original = self.upload()
        result = self.process(original).json()
        self.client.__exit__(None, None, None)
        self.store.closed = False
        self.open_client()
        self.assertEqual(self.process(original).json(), result)
        self.assertEqual(self.processor.detections, 0)
        later = self.process(self.upload((128, 0, 0))).json()
        self.assertEqual(
            next(d for d in later["documents"] if d["dtype"] == "person")["id"],
            next(d for d in result["documents"] if d["dtype"] == "person")["id"],
        )


@unittest.skipUnless(
    os.environ.get("FACE_INTEL_TEST_YUNET_MODEL"), "Opt-in YuNet/SFace real pipeline"
)
class RealPipelineTests(unittest.TestCase):
    def test_real_detection_alignment_and_candidate_graph(self):
        settings = Settings(
            api_token=TOKEN,
            sface_model_path=os.environ["FACE_INTEL_TEST_SFACE_MODEL"],
            sface_model_sha256=os.environ["FACE_INTEL_TEST_SFACE_SHA256"],
            yunet_model_path=os.environ["FACE_INTEL_TEST_YUNET_MODEL"],
            yunet_model_sha256=os.environ["FACE_INTEL_TEST_YUNET_SHA256"],
            auto_process_images=True,
        )
        with open(os.environ["FACE_INTEL_TEST_FACE_IMAGE"], "rb") as stream:
            data = stream.read()
        self.assertEqual(
            hashlib.sha256(data).hexdigest(),
            "7de7ed51a1594fff247f4cae2301eceacf5313d6011e37b4a4c8733f7bb72c07",
        )
        store = MemoryStore()
        headers = {"Authorization": "Bearer " + TOKEN}
        with TestClient(create_app(settings, store)) as client:
            response = client.post("/v1/photos", content=data, headers=headers)
            self.assertEqual(response.status_code, 200, response.text)
            result = client.post(
                "/v1/images/process", json={"photoId": response.json()["id"]}, headers=headers
            )
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual(result.json()["faceCount"], 1)
            crops = [
                d
                for d in result.json()["documents"]
                if d.get("extensions", {}).get("faceIntel", {}).get("faceCrop")
            ]
            self.assertEqual(len(crops), 1)
            self.assertEqual((crops[0]["width"], crops[0]["height"]), (112, 112))
            model = "opencv-sface:sha256:" + settings.sface_model_sha256
            self.assertEqual(len(crops[0]["extensions"]["faceIntel"]["embeddings"][model]), 128)
