"""MTI API with a small grounded chat service and the unchanged public contract."""

import asyncio
import logging
import os

from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse

from medical_rag.api.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
)
from medical_rag.generation.config import ChatSettings
from medical_rag.services.chat import ChatService, open_chat_service


logger = logging.getLogger(__name__)


def operational_error(error: Exception) -> JSONResponse:
    logger.error("Chat operational failure: %s", type(error).__name__)
    return JSONResponse(status_code=500, content=ErrorResponse(
        error="Chat service failed or timed out; please retry later.").model_dump())


def get_chat_service(request: Request):
    with open_chat_service(request.app.state.chat_settings) as service:
        yield service


def create_app() -> FastAPI:
    """Read model identities once at startup; no model connections are made."""
    settings = ChatSettings.from_env()
    health = HealthResponse(
        status="ok",
        llm=settings.model,
        embedding=os.getenv("EMBEDDING_MODEL", "bge-m3"),
    )
    application = FastAPI(title="MTI Medical RAG API")
    application.state.chat_settings = settings

    @application.exception_handler(Exception)
    async def unexpected_failure(request: Request, error: Exception):
        # Covers operational failures while constructing request dependencies too.
        return operational_error(error)

    @application.get("/health", response_model=HealthResponse)
    async def get_health() -> HealthResponse:
        """Report API liveness and configured identities, not model readiness."""
        return health

    @application.post(
        "/chat",
        response_model=ChatResponse,
        status_code=status.HTTP_200_OK,
        responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse}},
    )
    async def chat(request: ChatRequest, service: ChatService = Depends(get_chat_service)):
        """Sessions are accepted for compatibility; no session state is stored."""
        try:
            run = await asyncio.wait_for(asyncio.to_thread(service.run, request),
                                         timeout=settings.request_timeout_seconds)
            logger.info("Chat measurements: %s", run.measurements())
            return run.response
        except Exception as error:
            # No patient input, source excerpts, model traces, or machine paths in errors/logs.
            return operational_error(error)

    return application


app = create_app()
