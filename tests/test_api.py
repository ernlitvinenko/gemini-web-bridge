from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from gemini_web_bridge.config import Settings
from gemini_web_bridge.errors import BridgeError
from gemini_web_bridge.gemini_web import GenerationChunk
from gemini_web_bridge.main import create_app


class FakeDriver:
    def __init__(self, text: str = "Hello world") -> None:
        self.text = text
        self.prompts: list[tuple[str, str | None]] = []
        self.fail: Exception | None = None
        self.closed = False

    async def stream(self, prompt: str, model: str | None = None):
        self.prompts.append((prompt, model))
        if self.fail:
            raise self.fail
        midpoint = max(1, len(self.text) // 2)
        yield GenerationChunk(text=self.text[:midpoint], model="3.1 Pro")
        yield GenerationChunk(text=self.text, model="3.1 Pro")

    async def close(self) -> None:
        self.closed = True

    def status(self) -> dict[str, object]:
        return {"browser_started": False, "last_model": "3.1 Pro"}


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        profile_dir=tmp_path / "profile",
        debug_dir=tmp_path / "debug",
        default_model="gemini-web",
        model_map={"gemini-web": None, "gemini-web-pro": "Pro"},
        debug_artifacts=False,
    )


@pytest.fixture
def client(settings):
    driver = FakeDriver()
    app = create_app(settings=settings, driver=driver)
    with TestClient(app) as test_client:
        test_client.driver = driver
        yield test_client


def test_models(client) -> None:
    response = client.get("/v1/models")
    assert response.status_code == 200
    ids = [item["id"] for item in response.json()["data"]]
    assert ids == ["gemini-web", "gemini-web-pro"]


def test_chat_completion(client) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemini-web",
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "Hello world"
    assert payload["usage"]["total_tokens"] > 0
    assert client.driver.prompts == [("Hi", "gemini-web")]


def test_chat_completion_builds_transcript(client) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemini-web",
            "messages": [
                {"role": "system", "content": "Be terse"},
                {"role": "user", "content": "Question"},
            ],
        },
    )
    assert response.status_code == 200
    prompt, _ = client.driver.prompts[0]
    assert "SYSTEM INSTRUCTIONS:\nBe terse" in prompt
    assert "USER:\nQuestion" in prompt


def test_chat_streaming(client) -> None:
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gemini-web",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in response.iter_lines() if line]

    data_lines = [line for line in lines if line.startswith("data:")]
    assert data_lines[-1] == "data: [DONE]"
    chunks = [json.loads(line[6:]) for line in data_lines[:-1]]
    text = "".join(
        chunk["choices"][0]["delta"].get("content", "")
        for chunk in chunks
        if chunk.get("choices")
    )
    assert text == "Hello world"
    assert chunks[-1]["choices"] == []
    assert "usage" in chunks[-1]


def test_chat_streaming_without_usage(client) -> None:
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gemini-web",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        lines = [line for line in response.iter_lines() if line.startswith("data:")]

    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line[6:]) for line in lines[:-1]]
    assert all(chunk.get("choices") for chunk in chunks)


def test_legacy_completion(client) -> None:
    response = client.post(
        "/v1/completions",
        json={"model": "gemini-web", "prompt": "Hi"},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == "Hello world"


def test_responses_api(client) -> None:
    response = client.post(
        "/v1/responses",
        json={
            "model": "gemini-web",
            "instructions": "Be brief",
            "input": "Hi",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "response"
    assert payload["status"] == "completed"
    assert payload["output"][0]["content"][0]["text"] == "Hello world"
    assert payload["usage"]["total_tokens"] > 0


def test_responses_api_stream(client) -> None:
    with client.stream(
        "POST",
        "/v1/responses",
        json={"model": "gemini-web", "input": "Hi", "stream": True},
    ) as response:
        assert response.status_code == 200
        lines = [line for line in response.iter_lines() if line]

    events = [line[7:] for line in lines if line.startswith("event: ")]
    assert "response.created" in events
    assert "response.output_text.delta" in events
    assert "response.completed" in events

    data = [json.loads(line[6:]) for line in lines if line.startswith("data: ")]
    deltas = [
        item["delta"] for item in data if item["type"] == "response.output_text.delta"
    ]
    assert "".join(deltas) == "Hello world"


def test_unknown_model(client) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "nope", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "model_unavailable"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

TOOL_CALL_TEXT = (
    '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
)


def test_chat_tool_call_is_emulated(client) -> None:
    client.driver.text = TOOL_CALL_TEXT
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemini-web",
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": TOOLS,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    choice = payload["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    call = choice["message"]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["id"].startswith("call_")
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}

    prompt, _ = client.driver.prompts[0]
    assert "TOOLS:" in prompt
    assert "TOOL CALL PROTOCOL:" in prompt


def test_chat_tool_call_streams_arguments(client) -> None:
    client.driver.text = TOOL_CALL_TEXT
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gemini-web",
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": TOOLS,
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        lines = [line for line in response.iter_lines() if line.startswith("data:")]

    chunks = [json.loads(line[6:]) for line in lines[:-1]]
    deltas = [
        delta
        for chunk in chunks
        for delta in chunk["choices"][0]["delta"].get("tool_calls", [])
    ]
    assert deltas[0]["index"] == 0
    assert deltas[0]["function"]["name"] == "get_weather"
    assert deltas[0]["function"]["arguments"] == ""
    assert json.loads(deltas[1]["function"]["arguments"]) == {"city": "Paris"}
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_chat_tool_result_is_rendered_in_prompt(client) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemini-web",
            "messages": [
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": '{"temperature": 21}',
                },
            ],
            "tools": TOOLS,
        },
    )
    assert response.status_code == 200
    prompt, _ = client.driver.prompts[0]
    assert "ASSISTANT TOOL CALLS:" in prompt
    assert 'TOOL RESULT (call_1):\n{"temperature": 21}' in prompt


def test_responses_tool_call_is_emulated(client) -> None:
    client.driver.text = TOOL_CALL_TEXT
    flat_tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {"type": "object"},
        }
    ]
    response = client.post(
        "/v1/responses",
        json={
            "model": "gemini-web",
            "input": "Weather in Paris?",
            "tools": flat_tools,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    item = payload["output"][0]
    assert item["type"] == "function_call"
    assert item["status"] == "completed"
    assert item["call_id"].startswith("call_")
    assert item["name"] == "get_weather"
    assert json.loads(item["arguments"]) == {"city": "Paris"}
    assert payload["tools"] == flat_tools


def test_responses_tool_call_stream(client) -> None:
    client.driver.text = TOOL_CALL_TEXT
    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gemini-web",
            "input": "Weather in Paris?",
            "tools": TOOLS,
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        lines = [line for line in response.iter_lines() if line]

    events = [line[7:] for line in lines if line.startswith("event: ")]
    assert "response.output_item.added" in events
    assert "response.function_call_arguments.done" in events
    assert "response.completed" in events

    data = [json.loads(line[6:]) for line in lines if line.startswith("data: ")]
    done = next(
        item for item in data if item["type"] == "response.function_call_arguments.done"
    )
    assert json.loads(done["arguments"]) == {"city": "Paris"}
    completed = next(item for item in data if item["type"] == "response.completed")
    assert completed["response"]["output"][0]["type"] == "function_call"


def test_chat_tool_mode_falls_back_to_text(client) -> None:
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gemini-web",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": TOOLS,
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        lines = [line for line in response.iter_lines() if line.startswith("data:")]

    chunks = [json.loads(line[6:]) for line in lines[:-1]]
    text = "".join(
        chunk["choices"][0]["delta"].get("content") or ""
        for chunk in chunks
        if chunk.get("choices")
    )
    assert text == "Hello world"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_responses_tool_mode_falls_back_to_text(client) -> None:
    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gemini-web",
            "input": "Hi",
            "tools": TOOLS,
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        lines = [line for line in response.iter_lines() if line]

    data = [json.loads(line[6:]) for line in lines if line.startswith("data: ")]
    deltas = [
        item["delta"] for item in data if item["type"] == "response.output_text.delta"
    ]
    assert "".join(deltas) == "Hello world"


def test_responses_tool_result_is_rendered_in_prompt(client) -> None:
    response = client.post(
        "/v1/responses",
        json={
            "model": "gemini-web",
            "input": [
                {"type": "message", "role": "user", "content": "Weather in Paris?"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city": "Paris"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": '{"temperature": 21}',
                },
            ],
            "tools": TOOLS,
        },
    )
    assert response.status_code == 200
    prompt, _ = client.driver.prompts[0]
    assert "ASSISTANT TOOL CALLS:" in prompt
    assert 'TOOL RESULT (call_1):\n{"temperature": 21}' in prompt


def test_tools_can_be_disabled(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        tool_calling="off",
        profile_dir=tmp_path / "profile",
        debug_dir=tmp_path / "debug",
        default_model="gemini-web",
        model_map={"gemini-web": None},
        debug_artifacts=False,
    )
    app = create_app(settings=settings, driver=FakeDriver())
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-web",
                "messages": [{"role": "user", "content": "Hi"}],
                "tools": TOOLS,
            },
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_tools"


def test_validation_error_shape(client) -> None:
    response = client.post("/v1/chat/completions", json={"model": "gemini-web"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "validation_error"


def test_driver_error_is_returned(client) -> None:
    client.driver.fail = BridgeError("Login required", status_code=503, code="login")
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gemini-web", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "login"


def test_streaming_driver_error_before_first_chunk(client) -> None:
    client.driver.fail = BridgeError("Login required", status_code=503, code="login")
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gemini-web",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 503


def test_api_key_is_enforced(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        api_key="secret",
        profile_dir=tmp_path / "profile",
        debug_dir=tmp_path / "debug",
        model_map={"gemini-web": None},
        debug_artifacts=False,
    )
    driver = FakeDriver()
    app = create_app(settings=settings, driver=driver)
    with TestClient(app) as client:
        assert client.get("/v1/models").status_code == 401
        assert (
            client.get(
                "/v1/models", headers={"Authorization": "Bearer secret"}
            ).status_code
            == 200
        )
        assert (
            client.get("/v1/models", headers={"x-api-key": "secret"}).status_code == 200
        )
        assert (
            client.get(
                "/v1/models", headers={"Authorization": "Bearer wrong"}
            ).status_code
            == 401
        )


def test_health_does_not_require_key(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        api_key="secret",
        profile_dir=tmp_path / "profile",
        debug_dir=tmp_path / "debug",
        debug_artifacts=False,
    )
    app = create_app(settings=settings, driver=FakeDriver())
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
