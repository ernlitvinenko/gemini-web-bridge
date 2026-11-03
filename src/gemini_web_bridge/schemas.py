from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ContentPart(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    text: str | None = None
    image_url: Any | None = None
    input_text: str | None = None
    output_text: str | None = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[ContentPart] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = ""
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    n: int = 1
    tools: list[dict[str, Any]] | None = None
    functions: list[dict[str, Any]] | None = None
    response_format: dict[str, Any] | None = None
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = ""
    prompt: str | list[str]
    stream: bool = False
    max_tokens: int | None = None
    stop: str | list[str] | None = None


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = ""
    input: str | list[dict[str, Any]]
    instructions: str | None = None
    stream: bool = False
    max_output_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    previous_response_id: str | None = None
    store: bool = False
