from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MODELS: dict[str, str | None] = {
    "gemini-web": None,
    "gemini-web-auto": None,
    "gemini-web-pro": "Pro",
    "gemini-web-flash": "Flash",
    "gemini-web-flash-lite": "Flash-Lite",
    "gemini-web-argon": "Argon",
    "gemini-2.5-pro": "Pro",
    "gemini-2.5-flash": "Flash",
    "gemini-3-pro": "Pro",
    "gemini-3-flash": "Flash",
}


class Settings(BaseSettings):
    """Runtime configuration. All variables use the GEMINI_BRIDGE_ prefix."""

    model_config = SettingsConfigDict(
        env_prefix="GEMINI_BRIDGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str = ""

    gemini_url: str = "https://gemini.google.com/app"
    profile_dir: Path = Field(
        default_factory=lambda: Path.home() / ".gemini-web-bridge" / "profile"
    )
    headless: bool = False
    browser_channel: str = "chrome"
    browser_executable: str | None = None
    browser_args: str = "--disable-blink-features=AutomationControlled"
    accept_cookies: bool = True
    require_login: bool = True

    default_model: str = "gemini-web-pro"
    model_map: dict[str, str | None] = Field(
        default_factory=lambda: dict(DEFAULT_MODELS)
    )
    model_option_timeout_s: float = 12.0

    request_timeout_s: float = 300.0
    response_start_timeout_s: float = 120.0
    lock_wait_timeout_s: float = 300.0
    stability_ms: int = 1500
    poll_interval_ms: int = 250
    page_timeout_ms: int = 15000
    max_prompt_chars: int = 200_000

    log_level: str = "INFO"
    debug_artifacts: bool = True
    debug_dir: Path = Field(
        default_factory=lambda: Path.home() / ".gemini-web-bridge" / "debug"
    )

    @field_validator("profile_dir", "debug_dir", mode="before")
    @classmethod
    def _expand_path(cls, value: object) -> object:
        if isinstance(value, str):
            return Path(value).expanduser()
        return value

    @field_validator("browser_executable", mode="before")
    @classmethod
    def _empty_executable(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("api_key", mode="before")
    @classmethod
    def _strip_api_key(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    def prepare_dirs(self) -> None:
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        if self.debug_artifacts:
            try:
                self.debug_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                # Debug artifacts are optional; failure to create their directory
                # must not prevent the bridge from starting.
                pass

    def model_target(self, model: str | None) -> str | None:
        """Map an OpenAI model id to the label shown in Gemini's mode picker."""
        requested = (model or self.default_model).strip()
        normalized = requested.lower()
        for name, label in self.model_map.items():
            if name.lower() == normalized:
                return label
        raise KeyError(requested)
