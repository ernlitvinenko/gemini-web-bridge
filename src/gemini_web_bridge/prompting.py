from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .errors import BridgeError
from .schemas import ChatMessage, ContentPart


@dataclass
class NormalizedMessage:
    role: str
    text: str
    name: str | None = None


SYSTEM_ROLES = {"system", "developer"}
TEXT_PART_TYPES = {"text", "input_text", "output_text"}


def _part_text(part: ContentPart) -> str:
    if part.type in TEXT_PART_TYPES:
        value = part.text
        if value is None:
            value = part.input_text if part.input_text is not None else part.output_text
        return value or ""
    if part.type in {"image_url", "input_image", "image"}:
        raise BridgeError(
            "Image inputs are not supported by this bridge yet. Send text only.",
            status_code=400,
            code="unsupported_content",
            error_type="invalid_request_error",
            param="messages",
        )
    if part.type in {"input_audio", "audio"}:
        raise BridgeError(
            "Audio inputs are not supported by this bridge. Send text only.",
            status_code=400,
            code="unsupported_content",
            error_type="invalid_request_error",
            param="messages",
        )
    raise BridgeError(
        f"Unsupported content part type: {part.type!r}",
        status_code=400,
        code="unsupported_content",
        error_type="invalid_request_error",
        param="messages",
    )


def content_to_text(content: str | list[ContentPart] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "\n".join(filter(None, (_part_text(part) for part in content)))


def normalize_messages(messages: Iterable[ChatMessage]) -> list[NormalizedMessage]:
    normalized: list[NormalizedMessage] = []
    for message in messages:
        role = message.role.strip().lower()
        if role in SYSTEM_ROLES:
            role = "system"
        elif role in {"user", "assistant"}:
            pass
        elif role in {"tool", "function"}:
            role = "tool"
        else:
            raise BridgeError(
                f"Unsupported message role: {message.role!r}",
                status_code=400,
                code="unsupported_role",
                error_type="invalid_request_error",
                param="messages",
            )

        text = content_to_text(message.content).strip()
        if message.tool_calls:
            raise BridgeError(
                "Tool calling is not supported by this bridge. Remove `tools`/"
                "`tool_calls` from the request.",
                status_code=400,
                code="unsupported_tools",
                error_type="invalid_request_error",
                param="messages",
            )
        if not text and role != "assistant":
            raise BridgeError(
                f"Message with role {role!r} has no text content.",
                status_code=400,
                code="empty_message",
                error_type="invalid_request_error",
                param="messages",
            )
        normalized.append(NormalizedMessage(role=role, text=text, name=message.name))

    if not normalized:
        raise BridgeError(
            "`messages` must not be empty.",
            status_code=400,
            code="empty_messages",
            error_type="invalid_request_error",
            param="messages",
        )
    return normalized


def _response_format_instruction(response_format: dict[str, Any] | None) -> str | None:
    if not response_format:
        return None
    fmt_type = str(response_format.get("type") or "").lower()
    if fmt_type == "json_object":
        return (
            "Return only a valid JSON object. Do not wrap it in Markdown code fences."
        )
    if fmt_type == "json_schema":
        schema_spec = response_format.get("json_schema") or {}
        schema = schema_spec.get("schema") if isinstance(schema_spec, dict) else None
        name = schema_spec.get("name") if isinstance(schema_spec, dict) else None
        pieces = ["Return only valid JSON that conforms to the requested schema."]
        if name:
            pieces.append(f"Schema name: {name}.")
        if schema:
            pieces.append("JSON Schema:\n" + json.dumps(schema, ensure_ascii=False))
        pieces.append("Do not wrap the JSON in Markdown code fences.")
        return "\n".join(pieces)
    return None


def build_prompt(
    messages: Iterable[ChatMessage],
    response_format: dict[str, Any] | None = None,
) -> str:
    """Convert an OpenAI message list into one prompt for Gemini's web UI.

    Gemini Web does not expose separate system/user channels through the UI, so
    the whole transcript is serialized into one message. Stateless prompts avoid
    depending on the web conversation's hidden state.
    """

    normalized = normalize_messages(messages)
    system = "\n\n".join(m.text for m in normalized if m.role == "system" and m.text)
    convo = [m for m in normalized if m.role != "system"]

    directive = _response_format_instruction(response_format)
    if not system and not directive and len(convo) == 1 and convo[0].role == "user":
        return convo[0].text

    blocks: list[str] = []
    if system:
        blocks.append("SYSTEM INSTRUCTIONS:\n" + system)
    if directive:
        blocks.append("RESPONSE FORMAT:\n" + directive)

    if convo:
        lines = ["CONVERSATION:"]
        for message in convo:
            label = message.role.upper()
            if message.name:
                label += f" ({message.name})"
            if message.role == "tool":
                label = "TOOL RESULT"
            lines.append(f"{label}:\n{message.text}")
        blocks.append("\n\n".join(lines))

        if convo[-1].role == "user":
            blocks.append(
                "Answer the final USER message. Use the earlier conversation as "
                "context and do not restate the transcript."
            )
        elif convo[-1].role == "assistant":
            blocks.append("Continue the conversation from the final ASSISTANT message.")

    return "\n\n".join(blocks).strip()


def build_prompt_from_responses(
    input_value: str | list[dict[str, Any]],
    instructions: str | None = None,
) -> str:
    messages: list[ChatMessage] = []
    if instructions:
        messages.append(ChatMessage(role="system", content=instructions))

    if isinstance(input_value, str):
        messages.append(ChatMessage(role="user", content=input_value))
    else:
        for item in input_value:
            if not isinstance(item, dict):
                raise BridgeError(
                    "Each item in `input` must be an object.",
                    status_code=400,
                    code="invalid_input",
                    error_type="invalid_request_error",
                    param="input",
                )
            item_type = str(item.get("type") or "message").lower()
            if item_type not in {"message", "input_message"}:
                raise BridgeError(
                    f"Unsupported Responses API input item type: {item_type!r}",
                    status_code=400,
                    code="unsupported_content",
                    error_type="invalid_request_error",
                    param="input",
                )
            role = str(item.get("role") or "user")
            raw_content = item.get("content")
            if isinstance(raw_content, str):
                messages.append(ChatMessage(role=role, content=raw_content))
            elif isinstance(raw_content, list):
                parts: list[ContentPart] = []
                for raw_part in raw_content:
                    if isinstance(raw_part, str):
                        parts.append(ContentPart(type="text", text=raw_part))
                    elif isinstance(raw_part, dict):
                        part_type = str(raw_part.get("type") or "input_text")
                        if part_type in {"input_text", "output_text", "text"}:
                            parts.append(
                                ContentPart(
                                    type="text",
                                    text=str(raw_part.get("text") or ""),
                                )
                            )
                        else:
                            parts.append(ContentPart(**raw_part))
                    else:
                        raise BridgeError(
                            "Invalid content part in `input`.",
                            status_code=400,
                            code="invalid_input",
                            error_type="invalid_request_error",
                            param="input",
                        )
                messages.append(ChatMessage(role=role, content=parts))
            else:
                raise BridgeError(
                    "Responses API `content` must be a string or a list of parts.",
                    status_code=400,
                    code="invalid_input",
                    error_type="invalid_request_error",
                    param="input",
                )

        if not any(message.role == "user" for message in messages):
            raise BridgeError(
                "`input` must contain at least one user message.",
                status_code=400,
                code="empty_input",
                error_type="invalid_request_error",
                param="input",
            )
    return build_prompt(messages)


def unwrap_json_fences(text: str, response_format: dict[str, Any] | None) -> str:
    if not response_format:
        return text
    fmt_type = str(response_format.get("type") or "").lower()
    if fmt_type not in {"json_object", "json_schema"}:
        return text
    stripped = text.strip()
    match = re.fullmatch(
        r"```(?:json)?\s*\n?(.*?)\n?```", stripped, flags=re.DOTALL | re.IGNORECASE
    )
    return match.group(1).strip() if match else text


def estimate_tokens(text: str) -> int:
    """A conservative, provider-agnostic estimate (not a real tokenizer)."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))
