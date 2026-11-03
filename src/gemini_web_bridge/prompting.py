from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .errors import BridgeError
from .schemas import ChatMessage, ContentPart
from .tools import ToolSpec, render_tools_block


@dataclass
class NormalizedMessage:
    role: str
    text: str
    name: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None


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
        if not text and role not in {"assistant", "tool"}:
            raise BridgeError(
                f"Message with role {role!r} has no text content.",
                status_code=400,
                code="empty_message",
                error_type="invalid_request_error",
                param="messages",
            )
        normalized.append(
            NormalizedMessage(
                role=role,
                text=text,
                name=message.name,
                tool_calls=list(message.tool_calls or []),
                tool_call_id=message.tool_call_id,
            )
        )

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


def _tool_call_for_prompt(call: dict[str, Any]) -> dict[str, Any]:
    nested = call.get("function")
    payload = nested if isinstance(nested, dict) else call
    name = payload.get("name")
    arguments = payload.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass
    return {"name": name, "arguments": arguments}


def _conversation_lines(convo: Sequence[NormalizedMessage]) -> list[str]:
    lines = ["CONVERSATION:"]
    for message in convo:
        if message.role == "tool":
            call_id = message.tool_call_id or "unknown"
            lines.append(f"TOOL RESULT ({call_id}):\n{message.text}")
            continue

        if message.role == "assistant" and message.tool_calls:
            if message.text:
                lines.append(f"ASSISTANT:\n{message.text}")
            rendered = [_tool_call_for_prompt(call) for call in message.tool_calls]
            lines.append(
                "ASSISTANT TOOL CALLS:\n" + json.dumps(rendered, ensure_ascii=False)
            )
            continue

        label = message.role.upper()
        if message.name:
            label += f" ({message.name})"
        lines.append(f"{label}:\n{message.text}")
    return lines


def build_prompt(
    messages: Iterable[ChatMessage],
    response_format: dict[str, Any] | None = None,
    *,
    tools: Sequence[ToolSpec] = (),
    tool_choice: Any = None,
    parallel_tool_calls: bool = True,
) -> str:
    """Convert an OpenAI message list into one prompt for Gemini's web UI.

    Gemini Web does not expose separate system/user channels through the UI, so
    the whole transcript is serialized into one message. Stateless prompts avoid
    depending on the web conversation's hidden state.

    Tool calling is emulated: tool definitions and a strict textual protocol are
    appended to the prompt, and the answer is parsed later by ``tools.py``.
    """

    normalized = normalize_messages(messages)
    system = "\n\n".join(m.text for m in normalized if m.role == "system" and m.text)
    convo = [m for m in normalized if m.role != "system"]

    directive = _response_format_instruction(response_format)
    tools_block = render_tools_block(
        tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
    )
    if (
        not system
        and not directive
        and not tools_block
        and len(convo) == 1
        and convo[0].role == "user"
    ):
        return convo[0].text

    blocks: list[str] = []
    if system:
        blocks.append("SYSTEM INSTRUCTIONS:\n" + system)
    if tools_block:
        blocks.append(tools_block)
    if directive:
        blocks.append("RESPONSE FORMAT:\n" + directive)

    if convo:
        blocks.append("\n\n".join(_conversation_lines(convo)))

        if convo[-1].role == "user":
            blocks.append(
                "Answer the final USER message. Use the earlier conversation as "
                "context and do not restate the transcript."
            )
        elif convo[-1].role == "tool":
            blocks.append(
                "Use the TOOL RESULT above to answer the user's request. Do not "
                "call the same tool again with identical arguments."
            )
        elif convo[-1].role == "assistant":
            blocks.append("Continue the conversation from the final ASSISTANT message.")

    return "\n\n".join(blocks).strip()


def _responses_output_text(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        pieces: list[str] = []
        for part in output:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict):
                if part.get("type") in TEXT_PART_TYPES:
                    pieces.append(str(part.get("text") or ""))
                else:
                    pieces.append(json.dumps(part, ensure_ascii=False))
            else:
                pieces.append(str(part))
        return "\n".join(piece for piece in pieces if piece)
    return json.dumps(output, ensure_ascii=False)


def _responses_message(item: dict[str, Any]) -> ChatMessage:
    role = str(item.get("role") or "user")
    raw_content = item.get("content")
    if isinstance(raw_content, str):
        return ChatMessage(role=role, content=raw_content)
    if isinstance(raw_content, list):
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
        return ChatMessage(role=role, content=parts)
    raise BridgeError(
        "Responses API `content` must be a string or a list of parts.",
        status_code=400,
        code="invalid_input",
        error_type="invalid_request_error",
        param="input",
    )


def build_prompt_from_responses(
    input_value: str | list[dict[str, Any]],
    instructions: str | None = None,
    *,
    tools: Sequence[ToolSpec] = (),
    tool_choice: Any = None,
    parallel_tool_calls: bool = True,
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
            if item_type in {"message", "input_message"}:
                messages.append(_responses_message(item))
                continue

            if item_type == "function_call":
                call_id = str(item.get("call_id") or item.get("id") or "unknown")
                name = str(item.get("name") or "").strip()
                if not name:
                    raise BridgeError(
                        "Responses API `function_call` item has no name.",
                        status_code=400,
                        code="invalid_input",
                        error_type="invalid_request_error",
                        param="input",
                    )
                arguments = item.get("arguments", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": arguments,
                                },
                            }
                        ],
                    )
                )
                continue

            if item_type == "function_call_output":
                call_id = str(item.get("call_id") or "unknown")
                content = _responses_output_text(item.get("output"))
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=content,
                        tool_call_id=call_id,
                    )
                )
                continue

            raise BridgeError(
                f"Unsupported Responses API input item type: {item_type!r}",
                status_code=400,
                code="unsupported_content",
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

    return build_prompt(
        messages,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
    )


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
