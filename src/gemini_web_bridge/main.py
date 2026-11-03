from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .api import router
from .config import Settings
from .errors import BridgeError
from .gemini_web import GeminiWebDriver


def configure_logging(level: str) -> None:
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level.upper())
        return
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


def create_app(
    settings: Settings | None = None, driver: GeminiWebDriver | None = None
) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        await app.state.driver.close()

    app = FastAPI(
        title="Gemini Web Bridge",
        version="0.1.0",
        description=(
            "OpenAI-compatible API backed by the Gemini web interface. "
            "Uses a persistent Chrome profile and Playwright UI automation."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.driver = driver or GeminiWebDriver(settings)

    @app.exception_handler(BridgeError)
    async def bridge_error_handler(_request: Request, exc: BridgeError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.payload())

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = exc.errors()
        if errors:
            first = errors[0]
            location = ".".join(
                str(part) for part in first.get("loc", []) if part != "body"
            )
            message = (
                f"{location}: {first.get('msg', 'invalid value')}"
                if location
                else str(first.get("msg", "invalid request"))
            )
        else:
            message = "Invalid request"
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "validation_error",
                }
            },
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or "req_" + uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.monotonic()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        response.headers["x-gemini-bridge"] = "0.1.0"
        elapsed = time.monotonic() - started
        logging.getLogger("gemini_bridge.access").info(
            "%s %s -> %s (%.2fs) %s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
            request_id,
        )
        return response

    @app.get("/health")
    async def health(request: Request) -> dict[str, object]:
        return {
            "status": "ok",
            "service": "gemini-web-bridge",
            "browser": request.app.state.driver.status(),
        }

    @app.get("/")
    async def root() -> dict[str, object]:
        return {
            "service": "gemini-web-bridge",
            "docs": "/docs",
            "health": "/health",
            "openai_base_url": "/v1",
        }

    app.include_router(router)

    if not settings.api_key and settings.host not in {"127.0.0.1", "localhost", "::1"}:
        logging.getLogger(__name__).warning(
            "No GEMINI_BRIDGE_API_KEY is set while binding to %s. "
            "Use an API key or bind to 127.0.0.1.",
            settings.host,
        )

    return app


app = create_app()
