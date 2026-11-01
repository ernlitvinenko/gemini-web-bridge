from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from .auth import require_api_key
from .config import Settings
from .errors import BridgeError, ModelUnavailable
from .gemini_web import GeminiWebDriver, GenerationChunk
from .prompting import (
    build_prompt,
    build_prompt_from_responses,
    estimate_tokens,
    unwrap_json_fences,
)
from .schemas import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    ResponsesRequest,
)

router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _driver(request: Request) -> GeminiWebDriver:
    return request.app.state.driver


def _model_and_target(settings: Settings, requested: str) -> tuple[str, str | None]:
    model = requested.strip() or settings.default_model
    try:
        target = settings.model_target(model)
    except KeyError as exc:
        raise ModelUnavailable(
            f"Unknown model {str(exc)!r}. Available: {', '.join(settings.model_map)}"
        ) from exc
    return model, target


def _reject_tools(tools: list[dict[str, Any]] | None, label: str = "tools") -> None:
    if tools:
        raise BridgeError(
            f"`{label}` is not supported: this bridge talks to Gemini's web UI, "
            "which has no function-calling API. Remove tool definitions.",
            status_code=400,
            code="unsupported_tools",
            error_type="invalid_request_error",
            param=label,
        )


async def _collect(
    driver: GeminiWebDriver, prompt: str, model: str
) -> tuple[str, str | None]:
    text = ""
    actual_model: str | None = None
    async for chunk in driver.stream(prompt, model):
        if chunk.text:
            text = chunk.text
        if chunk.model:
            actual_model = chunk.model
    return text, actual_model


async def _first_chunk(
    driver: GeminiWebDriver, prompt: str, model: str
) -> tuple[AsyncIterator[GenerationChunk], GenerationChunk]:
    generator = driver.stream(prompt, model)
    try:
        first = await generator.__anext__()
    except StopAsyncIteration as exc:
        await generator.aclose()
        raise BridgeError(
            "Gemini returned an empty answer.",
            status_code=502,
            code="empty_response",
            error_type="api_error",
        ) from exc
    except BaseException:
        await generator.aclose()
        raise
    return generator, first


def _delta(previous: str, current: str) -> str:
    if not current:
        return ""
    if current.startswith(previous):
        return current[len(previous) :]
    prefix = os.path.commonprefix([previous, current])
    return current[len(prefix) :]


def _sse(data: dict[str, Any], event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _usage(prompt_text: str, completion_text: str) -> dict[str, int]:
    prompt_tokens = estimate_tokens(prompt_text)
    completion_tokens = estimate_tokens(completion_text)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _responses_usage(prompt_text: str, completion_text: str) -> dict[str, Any]:
    prompt_tokens = estimate_tokens(prompt_text)
    completion_tokens = estimate_tokens(completion_text)
    return {
        "input_tokens": prompt_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": completion_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _chat_chunk(
    *,
    completion_id: str,
    created: int,
    model: str,
    delta: dict[str, Any] | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": "gemini-web-bridge",
        "choices": [
            {
                "index": 0,
                "delta": delta or {},
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }


@router.get("/models")
async def list_models(request: Request) -> dict[str, Any]:
    settings = _settings(request)
    created = 1_700_000_000
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": created,
                "owned_by": "google-web",
            }
            for model_id in settings.model_map
        ],
    }


@router.get("/bridge/status")
async def bridge_status(request: Request) -> dict[str, Any]:
    settings = _settings(request)
    driver = _driver(request)
    return {
        "ok": True,
        "default_model": settings.default_model,
        "models": settings.model_map,
        "browser": driver.status(),
    }


@router.post("/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest) -> Any:
    settings = _settings(request)
    driver = _driver(request)
    _reject_tools(body.tools)
    _reject_tools(body.functions, "functions")
    if body.n != 1:
        raise BridgeError(
            "Only n=1 is supported.",
            status_code=400,
            code="unsupported_n",
            error_type="invalid_request_error",
            param="n",
        )

    model, _ = _model_and_target(settings, body.model)
    prompt = build_prompt(body.messages, body.response_format)
    completion_id = "chatcmpl-" + uuid.uuid4().hex
    created = int(time.time())
    include_usage = bool((body.stream_options or {}).get("include_usage"))

    if not body.stream:
        text, _actual = await _collect(driver, prompt, model)
        text = unwrap_json_fences(text, body.response_format)
        if not text:
            raise BridgeError(
                "Gemini returned an empty answer.",
                status_code=502,
                code="empty_response",
                error_type="api_error",
            )
        usage = _usage(prompt, text)
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "system_fingerprint": "gemini-web-bridge",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text,
                        "refusal": None,
                        "annotations": [],
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }

    generator, first = await _first_chunk(driver, prompt, model)

    async def event_stream() -> AsyncIterator[str]:
        prompt_text = prompt
        full_text = first.text
        previous = ""
        try:
            yield _sse(
                _chat_chunk(
                    completion_id=completion_id,
                    created=created,
                    model=model,
                    delta={"role": "assistant", "content": ""},
                )
            )
            current_delta = _delta(previous, full_text)
            if current_delta:
                yield _sse(
                    _chat_chunk(
                        completion_id=completion_id,
                        created=created,
                        model=model,
                        delta={"content": current_delta},
                    )
                )
            previous = full_text

            async for chunk in generator:
                if not chunk.text:
                    continue
                full_text = chunk.text
                current_delta = _delta(previous, full_text)
                previous = full_text
                if current_delta:
                    yield _sse(
                        _chat_chunk(
                            completion_id=completion_id,
                            created=created,
                            model=model,
                            delta={"content": current_delta},
                        )
                    )

            full_text = unwrap_json_fences(full_text, body.response_format)
            yield _sse(
                _chat_chunk(
                    completion_id=completion_id,
                    created=created,
                    model=model,
                    finish_reason="stop",
                )
            )
            if include_usage:
                usage = _usage(prompt_text, full_text)
                yield _sse(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [],
                        "usage": usage,
                    }
                )
            yield "data: [DONE]\n\n"
        except BridgeError as exc:
            yield _sse(exc.payload())
            yield "data: [DONE]\n\n"
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive
            yield _sse(
                BridgeError(
                    f"Streaming failed: {exc}",
                    status_code=502,
                    code="stream_error",
                ).payload()
            )
            yield "data: [DONE]\n\n"
        finally:
            await generator.aclose()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream; charset=utf-8",
        headers=SSE_HEADERS,
    )


@router.post("/completions")
async def completions(request: Request, body: CompletionRequest) -> Any:
    settings = _settings(request)
    driver = _driver(request)
    model, _ = _model_and_target(settings, body.model)

    if isinstance(body.prompt, list):
        prompt_text = "\n\n".join(body.prompt)
    else:
        prompt_text = body.prompt
    prompt = build_prompt([ChatMessage(role="user", content=prompt_text)])
    completion_id = "cmpl-" + uuid.uuid4().hex
    created = int(time.time())

    if not body.stream:
        text, _actual = await _collect(driver, prompt, model)
        if not text:
            raise BridgeError(
                "Gemini returned an empty answer.",
                status_code=502,
                code="empty_response",
                error_type="api_error",
            )
        return {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "text": text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": _usage(prompt_text, text),
        }

    generator, first = await _first_chunk(driver, prompt, model)

    async def event_stream() -> AsyncIterator[str]:
        previous = ""
        full_text = first.text
        try:
            current_delta = _delta(previous, full_text)
            previous = full_text
            yield _sse(
                {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "text": current_delta,
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
            )
            async for chunk in generator:
                if not chunk.text:
                    continue
                full_text = chunk.text
                current_delta = _delta(previous, full_text)
                previous = full_text
                if current_delta:
                    yield _sse(
                        {
                            "id": completion_id,
                            "object": "text_completion",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "text": current_delta,
                                    "index": 0,
                                    "logprobs": None,
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
            yield _sse(
                {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "text": "",
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": "stop",
                        }
                    ],
                }
            )
            yield "data: [DONE]\n\n"
        except BridgeError as exc:
            yield _sse(exc.payload())
            yield "data: [DONE]\n\n"
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive
            yield _sse(
                BridgeError(
                    f"Streaming failed: {exc}",
                    status_code=502,
                    code="stream_error",
                ).payload()
            )
            yield "data: [DONE]\n\n"
        finally:
            await generator.aclose()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream; charset=utf-8",
        headers=SSE_HEADERS,
    )


def _response_payload(
    *,
    response_id: str,
    model: str,
    created_at: int,
    status: str,
    output: list[dict[str, Any]],
    usage: dict[str, Any] | None,
    instructions: str | None,
    max_output_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": instructions,
        "max_output_tokens": max_output_tokens,
        "model": model,
        "output": output,
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "store": False,
        "temperature": temperature,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": top_p,
        "truncation": "disabled",
        "usage": usage,
        "metadata": {},
    }


@router.post("/responses")
async def responses(request: Request, body: ResponsesRequest) -> Any:
    settings = _settings(request)
    driver = _driver(request)
    _reject_tools(body.tools)
    if body.previous_response_id:
        raise BridgeError(
            "`previous_response_id` is not supported; send the full conversation in `input`.",
            status_code=400,
            code="unsupported_previous_response_id",
            error_type="invalid_request_error",
            param="previous_response_id",
        )

    model, _ = _model_and_target(settings, body.model)
    prompt = build_prompt_from_responses(body.input, body.instructions)
    response_id = "resp_" + uuid.uuid4().hex
    message_id = "msg_" + uuid.uuid4().hex
    created_at = int(time.time())
    temperature = getattr(body, "temperature", None)
    top_p = getattr(body, "top_p", None)

    if not body.stream:
        text, _actual = await _collect(driver, prompt, model)
        if not text:
            raise BridgeError(
                "Gemini returned an empty answer.",
                status_code=502,
                code="empty_response",
                error_type="api_error",
            )
        usage = _responses_usage(prompt, text)
        output = [
            {
                "id": message_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ]
        return _response_payload(
            response_id=response_id,
            model=model,
            created_at=created_at,
            status="completed",
            output=output,
            usage=usage,
            instructions=body.instructions,
            max_output_tokens=body.max_output_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    generator, first = await _first_chunk(driver, prompt, model)

    async def event_stream() -> AsyncIterator[str]:
        full_text = first.text
        previous = ""
        created_response = _response_payload(
            response_id=response_id,
            model=model,
            created_at=created_at,
            status="in_progress",
            output=[],
            usage=None,
            instructions=body.instructions,
            max_output_tokens=body.max_output_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        output_item = {
            "id": message_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        content_part = {"type": "output_text", "text": "", "annotations": []}

        try:
            yield _sse(
                {"type": "response.created", "response": created_response},
                "response.created",
            )
            yield _sse(
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": output_item,
                },
                "response.output_item.added",
            )
            yield _sse(
                {
                    "type": "response.content_part.added",
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": content_part,
                },
                "response.content_part.added",
            )

            current_delta = _delta(previous, full_text)
            previous = full_text
            if current_delta:
                yield _sse(
                    {
                        "type": "response.output_text.delta",
                        "item_id": message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": current_delta,
                    },
                    "response.output_text.delta",
                )

            async for chunk in generator:
                if not chunk.text:
                    continue
                full_text = chunk.text
                current_delta = _delta(previous, full_text)
                previous = full_text
                if current_delta:
                    yield _sse(
                        {
                            "type": "response.output_text.delta",
                            "item_id": message_id,
                            "output_index": 0,
                            "content_index": 0,
                            "delta": current_delta,
                        },
                        "response.output_text.delta",
                    )

            yield _sse(
                {
                    "type": "response.output_text.done",
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": full_text,
                },
                "response.output_text.done",
            )
            yield _sse(
                {
                    "type": "response.content_part.done",
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "text": full_text,
                        "annotations": [],
                    },
                },
                "response.content_part.done",
            )
            final_item = {
                "id": message_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": full_text, "annotations": []}
                ],
            }
            yield _sse(
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": final_item,
                },
                "response.output_item.done",
            )
            usage = _responses_usage(prompt, full_text)
            completed = _response_payload(
                response_id=response_id,
                model=model,
                created_at=created_at,
                status="completed",
                output=[final_item],
                usage=usage,
                instructions=body.instructions,
                max_output_tokens=body.max_output_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            yield _sse(
                {"type": "response.completed", "response": completed},
                "response.completed",
            )
        except BridgeError as exc:
            failed = _response_payload(
                response_id=response_id,
                model=model,
                created_at=created_at,
                status="failed",
                output=[],
                usage=None,
                instructions=body.instructions,
                max_output_tokens=body.max_output_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            failed["error"] = {
                "code": exc.code,
                "message": exc.message,
            }
            yield _sse(
                {"type": "response.failed", "response": failed},
                "response.failed",
            )
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive
            failed = _response_payload(
                response_id=response_id,
                model=model,
                created_at=created_at,
                status="failed",
                output=[],
                usage=None,
                instructions=body.instructions,
                max_output_tokens=body.max_output_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            failed["error"] = {"code": "stream_error", "message": str(exc)}
            yield _sse(
                {"type": "response.failed", "response": failed},
                "response.failed",
            )
        finally:
            await generator.aclose()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream; charset=utf-8",
        headers=SSE_HEADERS,
    )
