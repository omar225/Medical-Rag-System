"""FastAPI app: exact MTI contract at / (and /api for the web UI), one service per process."""

import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from threading import Lock

from fastapi import APIRouter, Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from medical_rag.api.schemas import ChatRequest, ChatResponse, ErrorResponse, HealthResponse
from medical_rag.generation.config import ChatSettings
from medical_rag.generation.ollama import OllamaChat
from medical_rag.retrieval.config import PROJECT_ROOT, RetrievalSettings
from medical_rag.services.chat import ChatService
from medical_rag.services.sessions import SessionStore, Turn


logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:  # uvicorn configures only its own loggers
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
FRONTEND_DIST = Path(os.getenv("FRONTEND_DIST", str(PROJECT_ROOT / "frontend" / "dist")))


def operational_error(error: Exception) -> JSONResponse:
    logger.error("Chat operational failure: %s", type(error).__name__)
    return JSONResponse(status_code=500, content=ErrorResponse(
        error="Chat service failed or timed out; please retry later.").model_dump())


def build_service(settings: ChatSettings) -> ChatService:
    """Load the knowledge base and model clients. Never downloads a model."""
    from medical_rag.knowledge.base import KnowledgeBase
    from medical_rag.retrieval.embeddings import OllamaEmbeddings

    retrieval = RetrievalSettings.from_env()
    provider = OllamaEmbeddings(retrieval.ollama_base_url, retrieval.model, retrieval.timeout_seconds)
    wait = float(os.getenv("INDEX_BUILD_WAIT_SECONDS", "0"))
    knowledge = KnowledgeBase.load(retrieval, provider, build_wait_seconds=wait)
    provider.timeout_seconds = min(retrieval.timeout_seconds, settings.request_timeout_seconds)
    return ChatService(knowledge, OllamaChat(retrieval.ollama_base_url, settings), settings)


class ServiceHolder:
    """Builds the service at startup, or lazily on the first request if startup could not."""

    def __init__(self, settings: ChatSettings):
        self.settings, self.service, self._lock = settings, None, Lock()

    def get(self) -> ChatService:
        with self._lock:
            if self.service is None:
                self.service = build_service(self.settings)
            return self.service


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.holder.get()


def create_app() -> FastAPI:
    settings = ChatSettings.from_env()
    health = HealthResponse(status="ok", llm=settings.model, embedding=os.getenv("EMBEDDING_MODEL", "bge-m3"))
    holder = ServiceHolder(settings)
    sessions = SessionStore()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if os.getenv("PRELOAD_ON_STARTUP", "1") == "1":
            try:
                await asyncio.to_thread(holder.get)
            except Exception as error:  # served lazily (and reported per request) instead
                logger.error("Startup preload failed: %s: %s", type(error).__name__, error)
        yield

    application = FastAPI(title="MTI Medical RAG API", lifespan=lifespan)
    application.state.holder = holder
    application.state.sessions = sessions

    @application.exception_handler(Exception)
    async def unexpected_failure(request: Request, error: Exception):
        return operational_error(error)

    router = APIRouter()

    @router.get("/health", response_model=HealthResponse)
    async def get_health() -> HealthResponse:
        """Liveness and configured model names (read by the grader)."""
        return health

    @router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK,
                 responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse}})
    async def chat(request: ChatRequest, service: ChatService = Depends(get_chat_service)):
        try:
            history = sessions.history(request.session_id)
            run = await asyncio.wait_for(asyncio.to_thread(service.run, request.message, history),
                                         timeout=settings.request_timeout_seconds)
        except Exception as error:
            # No patient input, source text, model traces or machine paths in errors/logs.
            return operational_error(error)
        sessions.append(request.session_id, Turn(request.message, run.response.answer))
        logger.info("Chat measurements: %s", run.measurements())
        return run.response

    application.include_router(router)
    application.include_router(router, prefix="/api", include_in_schema=False)
    if FRONTEND_DIST.is_dir():
        application.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
    return application


app = create_app()
