from __future__ import annotations

import secrets

from fastapi import Depends, Header, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .errors import BridgeError

bearer = HTTPBearer(auto_error=False)


async def require_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),  # noqa: B008
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
) -> None:
    expected = request.app.state.settings.api_key
    if not expected:
        return

    supplied = credentials.credentials if credentials is not None else x_api_key
    if supplied and secrets.compare_digest(supplied, expected):
        return

    raise BridgeError(
        "Invalid or missing API key. Send `Authorization: Bearer <key>` "
        "or `x-api-key: <key>`.",
        status_code=401,
        code="invalid_api_key",
        error_type="authentication_error",
    )
