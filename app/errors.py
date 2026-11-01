from __future__ import annotations


class BridgeError(Exception):
    """An error that can be returned to an OpenAI-compatible client."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 502,
        code: str = "gemini_bridge_error",
        error_type: str = "api_error",
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.error_type = error_type
        self.param = param

    def payload(self) -> dict[str, object]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


class LoginRequired(BridgeError):
    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message
            or (
                "Gemini Web is not signed in. Run `python -m app.login` once, "
                "sign in to your Google account in the opened browser, and retry."
            ),
            status_code=503,
            code="gemini_login_required",
            error_type="service_unavailable",
        )


class GeminiTimeout(BridgeError):
    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            status_code=504,
            code="gemini_timeout",
            error_type="timeout_error",
        )


class GeminiBusy(BridgeError):
    def __init__(
        self, message: str = "Gemini Web is busy with another request"
    ) -> None:
        super().__init__(
            message,
            status_code=503,
            code="gemini_busy",
            error_type="service_unavailable",
        )


class ModelUnavailable(BridgeError):
    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            status_code=400,
            code="model_unavailable",
            error_type="invalid_request_error",
            param="model",
        )
