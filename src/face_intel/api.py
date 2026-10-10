import hmac
import json
import queue
from contextlib import asynccontextmanager

import pykka
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from .actors import FaceIntelSystem
from .config import Settings
from .errors import Conflict, InvalidDocument, NotFound, SimilarityUnavailable, StorageUnavailable
from .spec import VERSION


class PhotoAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    photoId: str = Field(min_length=1, max_length=512)
    personId: str = Field(min_length=1, max_length=512)
    basis: str = Field(min_length=1, max_length=2000)


async def read_body(request: Request, limit: int) -> bytes:
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > limit:
            raise HTTPException(413, "Request body exceeds the byte limit")
        data.extend(chunk)
    return bytes(data)


async def read_object(request: Request) -> dict:
    data = await read_body(request, 1024 * 1024)
    try:
        result = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
        raise InvalidDocument("Expected a JSON object") from exc
    if not isinstance(result, dict):
        raise InvalidDocument("Expected a JSON object")
    return result


def create_app(settings: Settings | None = None, store=None, embedder_factory=None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        system = await run_in_threadpool(FaceIntelSystem, settings, store, embedder_factory)
        app.state.system = system
        try:
            yield
        finally:
            await run_in_threadpool(system.close)

    app = FastAPI(
        title="face-intel",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    async def authenticated(request: Request):
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(
            token.encode(), settings.api_token.encode()
        ):
            raise HTTPException(401, "Invalid bearer token", headers={"WWW-Authenticate": "Bearer"})

    async def error_handler(request: Request, error: Exception):
        if isinstance(error, InvalidDocument):
            status, detail = 422, str(error)
        elif isinstance(error, NotFound):
            status, detail = 404, "Record not found"
        elif isinstance(error, Conflict):
            status, detail = 409, str(error)
        elif isinstance(error, pykka.Timeout):
            status, detail = 503, "Request timed out; completion may still occur"
        elif isinstance(error, SimilarityUnavailable):
            status, detail = 503, "Facial similarity engine is unavailable"
        else:
            status, detail = 503, "Actor service or durable store is unavailable"
        return JSONResponse({"detail": detail}, status_code=status)

    for exception in (
        InvalidDocument,
        NotFound,
        Conflict,
        StorageUnavailable,
        SimilarityUnavailable,
        queue.Full,
        pykka.Timeout,
        pykka.ActorDeadError,
    ):
        app.add_exception_handler(exception, error_handler)

    async def dispatch(request: Request, operation: str, payload):
        return await run_in_threadpool(request.app.state.system.request, operation, payload)

    secured = [Depends(authenticated)]

    @app.get("/health")
    async def health(request: Request):
        refs = request.app.state.system.refs
        if not refs or not all(actor.is_alive() for actor in refs):
            raise HTTPException(503, "Actor service is unavailable")
        return {"status": "ok", "schemaVersion": VERSION}

    @app.get("/v1/manifest", dependencies=secured)
    async def manifest(request: Request):
        return request.app.state.system.manifest()

    @app.post("/v1/persons", dependencies=secured)
    async def ingest_person(request: Request):
        return await dispatch(request, "ingest-person", await read_object(request))

    @app.get("/v1/search/name", dependencies=secured)
    async def search_name(request: Request, q: str, limit: int = 20, after: str | None = None):
        return await dispatch(request, "search-name", {"name": q, "limit": limit, "after": after})

    @app.get("/v1/persons/{identifier:path}", dependencies=secured)
    async def get_person(
        request: Request, identifier: str, limit: int = 20, after: str | None = None
    ):
        return await dispatch(
            request, "get-person", {"id": identifier, "limit": limit, "after": after}
        )

    @app.post("/v1/photos", dependencies=secured)
    async def ingest_photo(request: Request):
        data = await read_body(request, settings.max_photo_bytes)
        return await dispatch(request, "ingest-photo", data)

    @app.get("/v1/photos/{identifier:path}", dependencies=secured)
    async def get_photo(request: Request, identifier: str):
        return await dispatch(request, "get-photo", {"id": identifier})

    @app.get("/v1/photo-bytes/{identifier:path}", dependencies=secured)
    async def photo_bytes(request: Request, identifier: str):
        data, media_type = await dispatch(request, "photo-bytes", {"id": identifier})
        return Response(
            data,
            media_type=media_type,
            headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"},
        )

    @app.post("/v1/links/photo-person", dependencies=secured)
    async def link_photo(request: Request):
        try:
            annotation = PhotoAnnotation.model_validate(await read_object(request))
        except ValidationError as exc:
            raise InvalidDocument("Expected photoId, personId and annotation basis") from exc
        return await dispatch(request, "link-photo-person", annotation.model_dump())

    @app.post("/v1/targets", dependencies=secured)
    async def target(request: Request):
        return await dispatch(request, "execute-target", await read_object(request))

    @app.post("/v1/search/face-similarity", dependencies=secured)
    async def search_similarity(request: Request):
        return await dispatch(request, "search-similar-faces", await read_object(request))

    @app.post("/v1/relations", dependencies=secured)
    async def ingest_relation(request: Request):
        return await dispatch(request, "ingest-relation", await read_object(request))

    @app.get("/v1/relations/{identifier:path}", dependencies=secured)
    async def get_relation(request: Request, identifier: str):
        return await dispatch(request, "get-relation", {"id": identifier})

    return app
