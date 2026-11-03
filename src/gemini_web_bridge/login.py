from __future__ import annotations

import argparse
import shlex
import sys

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from .config import Settings


def _launch_context(settings: Settings, headless: bool):
    kwargs: dict[str, object] = {
        "user_data_dir": str(settings.profile_dir),
        "headless": headless,
        "args": shlex.split(settings.browser_args),
        "ignore_default_args": ["--enable-automation"],
        "viewport": {"width": 1360, "height": 900},
        "locale": "en-US",
    }
    if settings.browser_channel:
        kwargs["channel"] = settings.browser_channel
    elif settings.browser_executable:
        kwargs["executable_path"] = settings.browser_executable
    return kwargs


def _signed_in(page) -> bool:
    if "accounts.google.com" in page.url:
        return False
    try:
        state = page.evaluate(
            """() => {
              const visible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
              if (visible(document.querySelector('[data-test-id="sign-in-button"]'))) return false;
              if (visible(document.querySelector('[data-test-id="sidenav-error-action-link"]'))) return false;
              if (visible(document.querySelector('a[href*="ServiceLogin"]'))) return false;
              return !Array.from(document.querySelectorAll('button, a')).some((el) =>
                visible(el) && /^(sign in|войти|anmelden|se connecter|iniciar sesión|accedi)$/i.test(el.innerText.trim())
              );
            }"""
        )
        return bool(state)
    except PlaywrightError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Sign in to Gemini Web for the bridge")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check whether the saved profile is already signed in",
    )
    args = parser.parse_args()

    settings = Settings()
    settings.prepare_dirs()
    if settings.api_key:
        print("API key is configured; it is not needed for this login helper.")

    with sync_playwright() as playwright:
        try:
            context = playwright.chromium.launch_persistent_context(
                **_launch_context(settings, headless=args.check or settings.headless)
            )
        except PlaywrightError as exc:
            print(f"Could not open Chrome: {exc}", file=sys.stderr)
            print(
                "Check that Chrome is installed and that the bridge server is not "
                "using the same profile directory.",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc

        pages = context.pages
        page = pages[0] if pages else context.new_page()
        ok = False
        try:
            page.goto(
                settings.gemini_url, wait_until="domcontentloaded", timeout=60_000
            )
            try:
                accept = page.locator('[data-test-id="accept-button"]')
                if accept.count() and accept.first.is_visible():
                    accept.first.click(timeout=3000)
            except PlaywrightError:
                pass

            if args.check:
                ok = _signed_in(page)
                print("Signed in." if ok else "Not signed in.")
                context.close()
                raise SystemExit(0 if ok else 1)

            print()
            print("=" * 72)
            print("Chrome is open. Sign in to the Google account that has Gemini Pro.")
            print(f"Profile: {settings.profile_dir}")
            print("Complete the login (and 2FA if asked) in the browser window.")
            print("Then return here and press Enter.")
            print("=" * 72)
            print()
            try:
                input("Press Enter when you are signed in... ")
            except EOFError, KeyboardInterrupt:
                print("\nCancelled.")
                context.close()
                raise SystemExit(130)

            page.wait_for_timeout(1500)
            ok = _signed_in(page)
            print(
                "Signed in. The bridge can now reuse this profile."
                if ok
                else "Still signed out. Run the login helper again."
            )
        finally:
            try:
                context.close()
            except Exception:  # noqa: BLE001,S110
                pass

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
