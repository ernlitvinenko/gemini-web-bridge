from __future__ import annotations

import json

import pytest

from gemini_web_bridge.errors import BridgeError
from gemini_web_bridge.tools import (
    normalize_tool_choice,
    parse_tool_calls,
    parse_tool_specs,
    render_tools_block,
    validate_tool_calls,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {"type": "object"},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web",
            "parameters": {"type": "object"},
        },
    },
]


def test_parse_tool_specs_accepts_nested_and_flat_shapes() -> None:
    nested = parse_tool_specs(TOOLS[:1])
    flat = parse_tool_specs(
        [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {"type": "object"},
            }
        ]
    )
    assert [spec.name for spec in nested] == ["get_weather"]
    assert [spec.name for spec in flat] == ["get_weather"]


def test_parse_tool_specs_accepts_many_tools() -> None:
    # DeepSeek Harness sends ~38 tool definitions by default.
    tools = [
        {
            "type": "function",
            "function": {
                "name": f"tool_{index}",
                "description": f"Tool {index}",
                "parameters": {"type": "object"},
            },
        }
        for index in range(38)
    ]
    assert len(parse_tool_specs(tools)) == 38


def test_parse_tool_specs_rejects_duplicates() -> None:
    with pytest.raises(BridgeError) as excinfo:
        parse_tool_specs([TOOLS[0], TOOLS[0]])
    assert excinfo.value.code == "duplicate_tool"


def test_normalize_tool_choice() -> None:
    specs = parse_tool_specs(TOOLS)
    assert normalize_tool_choice("auto", specs) == ("auto", None)
    assert normalize_tool_choice("none", specs) == ("none", None)
    assert normalize_tool_choice("required", specs) == ("required", None)
    assert normalize_tool_choice(
        {"type": "function", "function": {"name": "search"}}, specs
    ) == ("required", "search")


def test_render_tools_block_respects_none() -> None:
    specs = parse_tool_specs(TOOLS)
    assert render_tools_block(specs, tool_choice="none") is None
    block = render_tools_block(specs, tool_choice="required")
    assert block is not None
    assert "get_weather" in block
    assert "You MUST call at least one tool" in block


def test_parse_tool_calls_from_tags() -> None:
    specs = parse_tool_specs(TOOLS)
    calls = parse_tool_calls(
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}'
        "</tool_call>",
        specs,
    )
    assert calls is not None
    assert len(calls) == 1
    assert calls[0].name == "get_weather"
    assert calls[0].id.startswith("call_")
    assert json.loads(calls[0].arguments) == {"city": "Paris"}


def test_parse_tool_calls_from_bare_json() -> None:
    specs = parse_tool_specs(TOOLS)
    calls = parse_tool_calls(
        '{"tool_calls": [{"name": "search", "arguments": "{\\"q\\": \\"x\\"}"}]}',
        specs,
    )
    assert calls is not None
    assert calls[0].name == "search"
    assert json.loads(calls[0].arguments) == {"q": "x"}


def test_parse_tool_calls_returns_none_for_normal_text() -> None:
    specs = parse_tool_specs(TOOLS)
    assert parse_tool_calls("Just an answer.", specs) is None


def test_parse_tool_calls_rejects_unknown_tool() -> None:
    specs = parse_tool_specs(TOOLS)
    with pytest.raises(BridgeError) as excinfo:
        parse_tool_calls(
            '<tool_call>{"name": "missing", "arguments": {}}</tool_call>',
            specs,
        )
    assert excinfo.value.code == "tool_call_invalid"


def test_parse_tool_calls_rejects_unclosed_tag() -> None:
    specs = parse_tool_specs(TOOLS)
    with pytest.raises(BridgeError) as excinfo:
        parse_tool_calls('<tool_call>{"name": "search"', specs)
    assert excinfo.value.code == "tool_call_invalid"


def test_validate_tool_calls_requires_call_when_forced() -> None:
    specs = parse_tool_specs(TOOLS)
    with pytest.raises(BridgeError) as excinfo:
        validate_tool_calls(None, tool_choice="required", specs=specs)
    assert excinfo.value.code == "tool_call_required"


def test_validate_tool_calls_rejects_parallel_when_disabled() -> None:
    specs = parse_tool_specs(TOOLS)
    calls = parse_tool_calls(
        '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'
        '<tool_call>{"name": "search", "arguments": {}}</tool_call>',
        specs,
    )
    with pytest.raises(BridgeError) as excinfo:
        validate_tool_calls(
            calls,
            tool_choice="auto",
            specs=specs,
            parallel_tool_calls=False,
        )
    assert excinfo.value.code == "tool_call_invalid"
