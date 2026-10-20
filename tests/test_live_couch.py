"""Opt-in integration test against a real CouchDB, using a freshly created disposable DB."""

import os
import unittest
import uuid

import httpx
from fastapi.testclient import TestClient
from support import TOKEN, candidate_person, person, photo_bytes
from test_pipeline import FixtureProcessor, colored_image

from face_intel.api import create_app
from face_intel.config import Settings


@unittest.skipUnless(
    os.environ.get("FACE_INTEL_RUN_LIVE_COUCH") == "1",
    "Set FACE_INTEL_RUN_LIVE_COUCH=1 to test against real CouchDB",
)
class LiveCouchTests(unittest.TestCase):
    def test_pipeline_gallery_reviews_and_run_survive_restart(self):
        settings = Settings(
            api_token=TOKEN,
            couch_url=os.environ.get("COUCHDB_URL", "http://127.0.0.1:5984"),
            couch_username=os.environ["COUCHDB_USERNAME"],
            couch_password=os.environ["COUCHDB_PASSWORD"],
            couch_database="face_intel_test_" + uuid.uuid4().hex,
        )
        headers = {"Authorization": "Bearer " + TOKEN}
        try:
            with TestClient(create_app(settings, processor_factory=FixtureProcessor)) as client:
                original = client.post(
                    "/v1/photos", content=colored_image((255, 0, 0)), headers=headers
                ).json()
                response = client.post(
                    "/v1/images/process", json={"photoId": original["id"]}, headers=headers
                )
                self.assertEqual(response.status_code, 200, response.text)
                first = response.json()
                tentative = next(d for d in first["documents"] if d["dtype"] == "person")
                claim = next(
                    d
                    for d in first["documents"]
                    if d.get("verificationStatus") == "candidate" and d["dtype"] == "relation"
                )
                reviewed = client.post(
                    "/v1/relations",
                    json={
                        **claim,
                        "verificationStatus": "confirmed",
                        "verifiedBy": "actor:fixture-review",
                        "verifiedAt": 1791648000,
                    },
                    headers=headers,
                )
                self.assertEqual(reviewed.status_code, 200, reviewed.text)
            with TestClient(create_app(settings, processor_factory=FixtureProcessor)) as client:
                replay = client.post(
                    "/v1/images/process", json={"photoId": original["id"]}, headers=headers
                )
                self.assertEqual(replay.status_code, 200, replay.text)
                self.assertIn(reviewed.json(), replay.json()["documents"])
                later = client.post(
                    "/v1/photos", content=colored_image((128, 0, 0)), headers=headers
                ).json()
                response = client.post(
                    "/v1/images/process", json={"photoId": later["id"]}, headers=headers
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertIn(tentative, response.json()["documents"])
                crop = next(
                    d
                    for d in response.json()["documents"]
                    if d.get("extensions", {}).get("faceIntel", {}).get("faceCrop")
                )
                binary = client.get("/v1/photo-bytes/" + crop["id"], headers=headers)
                self.assertEqual(binary.content, colored_image((128, 0, 0), (112, 112)))
        finally:
            with httpx.Client(
                base_url=settings.couch_url,
                auth=(settings.couch_username, settings.couch_password),
                timeout=5,
            ) as cleanup:
                response = cleanup.delete("/" + settings.couch_database)
                self.assertIn(response.status_code, {200, 202, 404}, response.text)

    def test_persisted_photo_person_annotation_search_and_target(self):
        settings = Settings(
            api_token=TOKEN,
            couch_url=os.environ.get("COUCHDB_URL", "http://127.0.0.1:5984"),
            couch_username=os.environ["COUCHDB_USERNAME"],
            couch_password=os.environ["COUCHDB_PASSWORD"],
            couch_database="face_intel_test_" + uuid.uuid4().hex,
        )
        headers = {"Authorization": f"Bearer {TOKEN}"}
        try:
            with TestClient(create_app(settings)) as client:
                response = client.post(
                    "/v1/persons", json=person(aliases=["Fixture Alias"]), headers=headers
                )
                self.assertEqual(response.status_code, 200, response.text)
                record = response.json()
                response = client.post("/v1/photos", content=photo_bytes(), headers=headers)
                self.assertEqual(response.status_code, 200, response.text)
                picture = response.json()
                response = client.post(
                    "/v1/links/photo-person",
                    json={
                        "photoId": picture["id"],
                        "personId": record["id"],
                        "basis": "Fixture annotation",
                    },
                    headers=headers,
                )
                self.assertEqual(response.status_code, 200, response.text)
                link = response.json()
                response = client.get(
                    "/v1/search/name", params={"q": "FIXTURE ALIAS"}, headers=headers
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["documents"], [record])
                target = {
                    "id": "target:live-fixture",
                    "dataset": "face-intel",
                    "dtype": "target",
                    "schemaVersion": "0.10.1",
                    "actor": "face-intel",
                    "target": record["id"],
                    "options": {"operation": "get-person"},
                }
                response = client.post("/v1/targets", json=target, headers=headers)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["documents"], [record, picture, link])
                response = client.post(
                    "/v1/persons",
                    json=candidate_person(aliases=["Candidate Alias"]),
                    headers=headers,
                )
                self.assertEqual(response.status_code, 200, response.text)
                candidate = response.json()
                response = client.post(
                    "/v1/links/photo-person",
                    json={
                        "photoId": picture["id"],
                        "personId": candidate["id"],
                        "basis": "Supplied integration fixture",
                    },
                    headers=headers,
                )
                self.assertEqual(response.status_code, 200, response.text)
                candidate_link = response.json()
                response = client.post(
                    "/v1/relations",
                    json={
                        **candidate_link,
                        "verificationStatus": "confirmed",
                        "verifiedBy": "actor:review-fixture",
                        "verifiedAt": 1791648000,
                        "evidence": [
                            {"schema": "org.starintel/core@1/picture", "id": picture["id"]}
                        ],
                    },
                    headers=headers,
                )
                self.assertEqual(response.status_code, 200, response.text)
                confirmed_link = response.json()
            # Restart actors and the HTTP client: documents and original image survive.
            with TestClient(create_app(settings)) as client:
                response = client.get("/v1/photo-bytes/" + picture["id"], headers=headers)
                self.assertEqual(response.content, photo_bytes())
                bundle = client.get("/v1/persons/" + record["id"], headers=headers)
                self.assertEqual(bundle.json()["relations"], [link])
                response = client.get("/v1/persons/" + candidate["id"], headers=headers)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["person"], candidate)
                self.assertEqual(response.json()["relations"], [confirmed_link])
                self.assertFalse(confirmed_link["extensions"]["faceIntel"]["candidate"])
                response = client.get(
                    "/v1/search/name", params={"q": "CANDIDATE ALIAS"}, headers=headers
                )
                self.assertEqual(response.json()["documents"], [candidate])
                target["target"] = confirmed_link["id"]
                target["options"] = {"operation": "get-relation"}
                response = client.post("/v1/targets", json=target, headers=headers)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["documents"], [confirmed_link])

        finally:
            with httpx.Client(
                base_url=settings.couch_url,
                auth=(settings.couch_username, settings.couch_password),
                timeout=5,
            ) as cleanup:
                response = cleanup.delete("/" + settings.couch_database)
                self.assertIn(response.status_code, {200, 202, 404}, response.text)


if __name__ == "__main__":
    unittest.main()
