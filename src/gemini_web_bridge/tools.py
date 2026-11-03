from __future__ import annotations

import html
import json
import re
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .errors import BridgeError

TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
TOOL_TAG_RE = re.compile(
    r"(?:<tool_call>|&lt;tool_call&gt;)\s*(.*?)\s*(?:</tool_call>|&lt;/tool_call&gt;)",
    re.DOTALL,
)
FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    strict: bool = False


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


def _tool_payload(item: dict[str, Any]) -> dict[str, Any]:
    """Accept both Chat Completions and Responses API tool shapes."""

    nested = item.get("function")
    if isinstance(nested, dict):
        return nested
    return item


def parse_tool_specs(
    tools: Sequence[dict[str, Any]] | None,
    functions: Sequence[dict[str, Any]] | None = None,
    *,
    max_tools: int = 128,
) -> list[ToolSpec]:
    raw: list[dict[str, Any]] = []

    for item in tools or []:
        if not isinstance(item, dict):
            raise BridgeError(
                "Each item in `tools` must be an object.",
                status_code=400,
                code="invalid_tools",
                error_type="invalid_request_error",
                param="tools",
            )
        tool_type = str(item.get("type") or "function").lower()
        if tool_type != "function":
            raise BridgeError(
                f"Tool type {tool_type!r} is not supported. Only `function` tools "
                "are available through this bridge.",
                status_code=400,
                code="unsupported_tool_type",
                error_type="invalid_request_error",
                param="tools",
            )
        raw.append(_tool_payload(item))

    for item in functions or []:
        if not isinstance(item, dict):
            raise BridgeError(
                "Each item in `functions` must be an object.",
                status_code=400,
                code="invalid_tools",
                error_type="invalid_request_error",
                param="functions",
            )
        raw.append(item)

    if len(raw) > max_tools:
        raise BridgeError(
            f"Too many tools ({len(raw)}; limit is {max_tools}).",
            status_code=400,
            code="tools_too_large",
            error_type="invalid_request_error",
            param="tools",
        )

    specs: list[ToolSpec] = []
    seen: set[str] = set()
    for item in raw:
        name = str(item.get("name") or "").strip()
        if not TOOL_NAME_RE.match(name):
            raise BridgeError(
                f"Invalid tool name {name!r}. Use 1-64 characters: letters, digits, "
                "underscores, or hyphens.",
                status_code=400,
                code="invalid_tool_name",
                error_type="invalid_request_error",
                param="tools",
            )
        if name in seen:
            raise BridgeError(
                f"Duplicate tool name {name!r}.",
                status_code=400,
                code="duplicate_tool",
                error_type="invalid_request_error",
                param="tools",
            )
        seen.add(name)

        parameters = item.get("parameters")
        if parameters is None:
            parameters = {"type": "object", "properties": {}}
        if not isinstance(parameters, dict):
            raise BridgeError(
                f"`parameters` for tool {name!r} must be a JSON Schema object.",
                status_code=400,
                code="invalid_tool_parameters",
                error_type="invalid_request_error",
                param="tools",
            )

        specs.append(
            ToolSpec(
                name=name,
                description=str(item.get("description") or "").strip(),
                parameters=parameters,
                strict=bool(item.get("strict")),
            )
        )

    return specs


def normalize_tool_choice(
    tool_choice: Any,
    specs: Sequence[ToolSpec],
) -> tuple[str, str | None]:
    """Return (mode, forced_name) where mode is auto, none, or required."""

    names = {spec.name for spec in specs}
    if tool_choice is None or tool_choice == "" or tool_choice == "auto":
        return "auto", None
    if tool_choice == "none":
        return "none", None
    if isinstance(tool_choice, str) and tool_choice in {"required", "any"}:
        if not specs:
            raise BridgeError(
                "`tool_choice` requires at least one tool definition.",
                status_code=400,
                code="invalid_tool_choice",
                error_type="invalid_request_error",
                param="tool_choice",
            )
        return "required", None

    if isinstance(tool_choice, dict):
        choice_type = str(tool_choice.get("type") or "function").lower()
        if choice_type != "function":
            raise BridgeError(
                f"Unsupported `tool_choice` type {choice_type!r}.",
                status_code=400,
                code="invalid_tool_choice",
                error_type="invalid_request_error",
                param="tool_choice",
            )
        nested = tool_choice.get("function")
        name = (
            nested.get("name") if isinstance(nested, dict) else tool_choice.get("name")
        )
        name = str(name or "").strip()
        if not name:
            raise BridgeError(
                "`tool_choice` must name a function.",
                status_code=400,
                code="invalid_tool_choice",
                error_type="invalid_request_error",
                param="tool_choice",
            )
        if name not in names:
            raise BridgeError(
                f"`tool_choice` references unknown tool {name!r}.",
                status_code=400,
                code="invalid_tool_choice",
                error_type="invalid_request_error",
                param="tool_choice",
            )
        return "required", name

    raise BridgeError(
        "Unsupported `tool_choice` value.",
        status_code=400,
        code="invalid_tool_choice",
        error_type="invalid_request_error",
        param="tool_choice",
    )


def render_tools_block(
    specs: Sequence[ToolSpec],
    *,
    tool_choice: Any = None,
    parallel_tool_calls: bool = True,
) -> str | None:
    if not specs:
        return None
    mode, forced_name = normalize_tool_choice(tool_choice, specs)
    if mode == "none":
        return None

    definitions = [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }
        for spec in specs
    ]

    lines = [
        "TOOLS:",
        "You can call the following functions:",
        json.dumps(definitions, ensure_ascii=False),
        "",
        "TOOL CALL PROTOCOL:",
        "- To call a tool, output ONLY one block per call in this exact format:",
        '- <tool_call>{"name": "<tool_name>", "arguments": {"<arg>": "<value>"}}</tool_call>',
        "- Do not wrap <tool_call> in Markdown code fences or add commentary to it.",
        "- Never invent tool results. A result will arrive as a TOOL RESULT message.",
        "- If no tool is needed, answer the user normally without a <tool_call> block.",
    ]
    if parallel_tool_calls:
        lines.append(
            "- You may emit several <tool_call> blocks in one answer when the calls "
            "are independent."
        )
    else:
        lines.append("- Emit at most one <tool_call> block per answer.")

    if mode == "required" and forced_name is None:
        lines.append("- You MUST call at least one tool before answering.")
    elif forced_name is not None:
        lines.append(f"- You MUST call the function `{forced_name}` before answering.")

    return "\n".join(lines)


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    match = FENCE_RE.match(stripped)
    return match.group(1).strip() if match else stripped


def _looks_like_tool_payload(data: Any) -> bool:
    if isinstance(data, list):
        return bool(data) and all(_looks_like_tool_payload(item) for item in data)
    if not isinstance(data, dict):
        return False
    if isinstance(data.get("tool_calls"), list):
        return True
    return "name" in data and "arguments" in data


def _iter_call_items(data: Any) -> Iterable[dict[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get("tool_calls"), list):
        data = data["tool_calls"]
    items = data if isinstance(data, list) else [data]
    for item in items:
        if not isinstance(item, dict):
            raise BridgeError(
                "Gemini returned a tool call that is not a JSON object.",
                status_code=502,
                code="tool_call_invalid",
                error_type="api_error",
            )
        nested = item.get("function")
        if isinstance(nested, dict):
            item = nested
        yield item


def _parse_arguments(value: Any) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        try:
            json.loads(value)
        except json.JSONDecodeError as exc:
            raise BridgeError(
                f"Gemini returned invalid tool arguments: {exc}",
                status_code=502,
                code="tool_call_invalid",
                error_type="api_error",
            ) from exc
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    raise BridgeError(
        "Tool call arguments must be a JSON object or a JSON string.",
        status_code=502,
        code="tool_call_invalid",
        error_type="api_error",
    )


def parse_tool_calls(
    text: str,
    specs: Sequence[ToolSpec],
    *,
    max_calls: int = 8,
) -> list[ToolCall] | None:
    names = {spec.name for spec in specs}
    if not names:
        return None

    payloads = TOOL_TAG_RE.findall(text)
    if not payloads:
        if "<tool_call>" in text or "&lt;tool_call&gt;" in text:
            raise BridgeError(
                "Gemini started a tool call but did not close the <tool_call> block.",
                status_code=502,
                code="tool_call_invalid",
                error_type="api_error",
            )
        candidate = _strip_fence(text)
        if not candidate or candidate[0] not in "{[":
            return None
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        if not _looks_like_tool_payload(data):
            return None
        payloads = [json.dumps(data, ensure_ascii=False)]

    calls: list[ToolCall] = []
    for payload in payloads:
        try:
            data = json.loads(html.unescape(payload))
        except json.JSONDecodeError as exc:
            raise BridgeError(
                f"Gemini returned an invalid tool call: {exc}",
                status_code=502,
                code="tool_call_invalid",
                error_type="api_error",
            ) from exc

        for item in _iter_call_items(data):
            name = str(item.get("name") or "").strip()
            if name not in names:
                raise BridgeError(
                    f"Gemini called unknown tool {name!r}. Declared tools: "
                    f"{', '.join(sorted(names))}.",
                    status_code=502,
                    code="tool_call_invalid",
                    error_type="api_error",
                )
            calls.append(
                ToolCall(
                    id="call_" + uuid.uuid4().hex[:24],
                    name=name,
                    arguments=_parse_arguments(item.get("arguments")),
                )
            )
            if len(calls) > max_calls:
                raise BridgeError(
                    f"Gemini returned more than {max_calls} tool calls.",
                    status_code=502,
                    code="tool_call_limit",
                    error_type="api_error",
                )

    return calls


def validate_tool_calls(
    calls: list[ToolCall] | None,
    *,
    tool_choice: Any,
    specs: Sequence[ToolSpec],
    parallel_tool_calls: bool = True,
) -> list[ToolCall] | None:
    mode, forced_name = normalize_tool_choice(tool_choice, specs)
    if mode == "required" and not calls:
        raise BridgeError(
            "Gemini did not return a tool call although `tool_choice` requires one.",
            status_code=502,
            code="tool_call_required",
            error_type="api_error",
        )
    if calls and not parallel_tool_calls and len(calls) > 1:
        raise BridgeError(
            "Gemini returned parallel tool calls while `parallel_tool_calls` is false.",
            status_code=502,
            code="tool_call_invalid",
            error_type="api_error",
        )
    if (
        forced_name is not None
        and calls
        and any(call.name != forced_name for call in calls)
    ):
        raise BridgeError(
            f"Gemini called a tool other than the required {forced_name!r}.",
            status_code=502,
            code="tool_call_invalid",
            error_type="api_error",
        )
    return calls


def chat_tool_calls(calls: Sequence[ToolCall]) -> list[dict[str, Any]]:
    return [
        {
            "id": call.id,
            "type": "function",
            "function": {"name": call.name, "arguments": call.arguments},
        }
        for call in calls
    ]


def response_function_item(call: ToolCall) -> dict[str, Any]:
    return {
        "id": "fc_" + uuid.uuid4().hex,
        "type": "function_call",
        "status": "completed",
        "call_id": call.id,
        "name": call.name,
        "arguments": call.arguments,
    }
