"""Stage 1 API: liveness, request validation, and the future chat contract."""

import os

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

from medical_rag.api.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
)


def create_app() -> FastAPI:
    """Read model identities once at startup; no model connections are made."""
    health = HealthResponse(
        status="ok",
        llm=os.getenv("LLM_MODEL", "qwen2.5:7b"),
        embedding=os.getenv("EMBEDDING_MODEL", "bge-m3"),
    )
    application = FastAPI(title="MTI Medical RAG API")

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
    async def chat(request: ChatRequest) -> JSONResponse:
        """Validate input; the pipeline and session persistence are pending."""
        error = ErrorResponse(error="Chat pipeline is not implemented yet")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error.model_dump(),
        )

    return application


app = create_app()
