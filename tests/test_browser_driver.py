from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from app.config import Settings
from app.errors import LoginRequired
from app.gemini_web import GeminiWebDriver

FIXTURE = Path(__file__).parent / "fixtures" / "gemini_mock.html"


def _chrome_available() -> bool:
    return bool(
        shutil.which("google-chrome")
        or shutil.which("chrome")
        or shutil.which("chromium")
        or Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome").exists()
    )


pytestmark = pytest.mark.skipif(
    not _chrome_available(), reason="Google Chrome is not installed"
)


def _settings(tmp_path: Path, url: str, **overrides) -> Settings:
    values = {
        "gemini_url": url,
        "profile_dir": tmp_path / "profile",
        "debug_dir": tmp_path / "debug",
        "headless": True,
        "browser_channel": "chrome",
        "accept_cookies": False,
        "require_login": False,
        "default_model": "gemini-web-pro",
        "model_map": {"gemini-web-pro": "Pro"},
        "stability_ms": 350,
        "poll_interval_ms": 40,
        "response_start_timeout_s": 15,
        "request_timeout_s": 25,
        "page_timeout_ms": 10_000,
        "lock_wait_timeout_s": 20,
        "debug_artifacts": False,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.mark.asyncio
async def test_browser_generation_and_markdown(tmp_path) -> None:
    driver = GeminiWebDriver(_settings(tmp_path, FIXTURE.as_uri()))
    chunks = []
    try:
        async for chunk in driver.stream("Hello from test", "gemini-web-pro"):
            chunks.append(chunk)
    finally:
        await driver.close()

    assert chunks
    text = chunks[-1].text
    assert "## Answer" in text
    assert "Mode: **Pro**" in text
    assert '```python\nprint("hello")\n```' in text
    assert "- one" in text
    assert "- two" in text
    assert chunks[-1].model == "3.6 Pro"


@pytest.mark.asyncio
async def test_browser_login_required(tmp_path) -> None:
    driver = GeminiWebDriver(
        _settings(tmp_path, FIXTURE.as_uri() + "?signin=1", require_login=True)
    )
    try:
        with pytest.raises(LoginRequired):
            async for _chunk in driver.stream("Hello", "gemini-web-pro"):
                pass
    finally:
        await driver.close()
