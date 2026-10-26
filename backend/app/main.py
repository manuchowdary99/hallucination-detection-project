"""FastAPI application entry point."""

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.errors import register_error_handlers
from app.api.routes import router
from app.core.config import get_settings
from app.models.database import init_db
from app.services.entity_recognizer import get_entity_recognizer
from app.services.knowledge_retriever import get_knowledge_retriever
from app.services.nli_verifier import NLIUnavailable, get_nli_verifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
settings = get_settings()


def _preload() -> None:
    get_entity_recognizer().load()
    try:
        get_nli_verifier().load()
    except NLIUnavailable:
        pass


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    if settings.nli_preload:
        threading.Thread(target=_preload, daemon=True).start()
    yield
    await get_knowledge_retriever().fetcher.aclose()


app = FastAPI(
    title="VeriAI API",
    description="Real-time LLM hallucination detection and verification.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.frontend_origin.split(",") if o.strip()],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
    expose_headers=["X-Total-Count"],
)


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > settings.max_request_bytes:
        return JSONResponse(status_code=413, content={"error": {"code": "payload_too_large",
                                                               "message": "Request body is too large."}})
    return await call_next(request)


register_error_handlers(app)
app.include_router(router)
