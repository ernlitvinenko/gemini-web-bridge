from __future__ import annotations

import argparse

import uvicorn

from .config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Gemini Web Bridge")
    parser.add_argument(
        "--reload", action="store_true", help="Enable uvicorn autoreload"
    )
    args = parser.parse_args()
    settings = Settings()
    uvicorn.run(
        "gemini_web_bridge.main:app",
        host=settings.host,
        port=settings.port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
