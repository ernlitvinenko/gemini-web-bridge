from __future__ import annotations

import pytest

from app.errors import BridgeError
from app.prompting import (
    build_prompt,
    build_prompt_from_responses,
    content_to_text,
    estimate_tokens,
)
from app.schemas import ChatMessage


def test_single_user_prompt_is_sent_verbatim() -> None:
    prompt = build_prompt([ChatMessage(role="user", content="Hello")])
    assert prompt == "Hello"


def test_system_and_history_are_serialized() -> None:
    prompt = build_prompt(
        [
            ChatMessage(role="system", content="Be concise."),
            ChatMessage(role="user", content="First question"),
            ChatMessage(role="assistant", content="First answer"),
            ChatMessage(role="user", content="Second question"),
        ]
    )
    assert "SYSTEM INSTRUCTIONS:\nBe concise." in prompt
    assert "USER:\nFirst question" in prompt
    assert "ASSISTANT:\nFirst answer" in prompt
    assert "USER:\nSecond question" in prompt
    assert "Answer the final USER message" in prompt


def test_content_parts_are_joined() -> None:
    prompt = build_prompt(
        [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "First"},
                    {"type": "text", "text": "Second"},
                ],
            )
        ]
    )
    assert prompt == "First\nSecond"


def test_image_parts_are_rejected() -> None:
    with pytest.raises(BridgeError) as error:
        build_prompt(
            [
                ChatMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "Look"},
                        {"type": "image_url", "image_url": {"url": "data:..."}},
                    ],
                )
            ]
        )
    assert error.value.code == "unsupported_content"


def test_json_response_format_directive() -> None:
    prompt = build_prompt(
        [ChatMessage(role="user", content="Return a user")],
        response_format={"type": "json_object"},
    )
    assert "Return only a valid JSON object" in prompt


def test_responses_input_conversion() -> None:
    prompt = build_prompt_from_responses(
        [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}],
        instructions="You are a test.",
    )
    assert "SYSTEM INSTRUCTIONS:\nYou are a test." in prompt
    assert "USER:\nHi" in prompt


def test_content_to_text_handles_none() -> None:
    assert content_to_text(None) == ""


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100
