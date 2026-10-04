import os
import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit


@dataclass(frozen=True, slots=True)
class Settings:
    api_token: str = field(repr=False)
    couch_url: str = "http://127.0.0.1:5984"
    couch_database: str = "face_intel"
    couch_username: str = field(default="", repr=False)
    couch_password: str = field(default="", repr=False)
    dataset: str = "face-intel"
    max_photo_bytes: int = 10 * 1024 * 1024
    max_photo_pixels: int = 20_000_000
    request_timeout: float = 30.0
    couch_timeout: float = 5.0

    def __post_init__(self) -> None:
        url = urlsplit(self.couch_url)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path not in {"", "/"}
        ):
            raise ValueError("COUCHDB_URL must be an HTTP origin without credentials or a path")
        if not re.fullmatch(r"[a-z][a-z0-9_$()+-]*", self.couch_database):
            raise ValueError("Invalid CouchDB database name")
        if len(self.api_token) < 24:
            raise ValueError("FACE_INTEL_API_TOKEN must contain at least 24 characters")
        if not self.dataset.strip() or len(self.dataset) > 256:
            raise ValueError("Invalid service dataset")
        if not 1 <= self.max_photo_bytes <= 50 * 1024 * 1024:
            raise ValueError("Photo byte limit must be between 1 byte and 50 MiB")
        if not 1 <= self.max_photo_pixels <= 100_000_000:
            raise ValueError("Invalid photo pixel limit")
        if not 0 < self.couch_timeout < self.request_timeout <= 120:
            raise ValueError("Timeouts must satisfy 0 < couch < request <= 120 seconds")
        if bool(self.couch_username) != bool(self.couch_password):
            raise ValueError("Set both CouchDB username and password")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            api_token=os.environ.get("FACE_INTEL_API_TOKEN", ""),
            couch_url=os.environ.get("COUCHDB_URL", "http://127.0.0.1:5984"),
            couch_database=os.environ.get("COUCHDB_DATABASE", "face_intel"),
            couch_username=os.environ.get("COUCHDB_USERNAME", ""),
            couch_password=os.environ.get("COUCHDB_PASSWORD", ""),
            dataset=os.environ.get("FACE_INTEL_DATASET", "face-intel"),
        )
