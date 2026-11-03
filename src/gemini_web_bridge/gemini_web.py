from __future__ import annotations

import asyncio
import logging
import re
import shlex
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from playwright.async_api import (
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)
from playwright.async_api import (
    Error as PlaywrightError,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)

from .config import Settings
from .errors import (
    BridgeError,
    GeminiBusy,
    GeminiTimeout,
    LoginRequired,
    ModelUnavailable,
)

logger = logging.getLogger(__name__)

EXTRACT_JS = (Path(__file__).with_name("extract.js")).read_text(encoding="utf-8")

STATE_JS = r"""
(() => {
  const visible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
  const signIn =
    visible(document.querySelector('[data-test-id="sign-in-button"]')) ||
    visible(document.querySelector('[data-test-id="sidenav-error-action-link"]')) ||
    /accounts\.google\.com/.test(location.href) ||
    visible(document.querySelector('a[href*="ServiceLogin"]')) ||
    Array.from(document.querySelectorAll('button, a')).some((el) =>
      visible(el) && /^(sign in|войти|anmelden|se connecter|iniciar sesión|accedi)$/i.test(el.innerText.trim())
    );
  const captcha =
    /\/sorry\//.test(location.href) ||
    !!document.querySelector('iframe[src*="recaptcha"]');
  let generating = false;
  const container = document.querySelector('[data-test-id="send-button-container"]');
  if (container) {
    generating = Array.from(container.querySelectorAll(
      '[data-test-id="stop-button"], mat-icon[data-mat-icon-name="stop"], ' +
      'mat-icon[data-mat-icon-name="stop_circle"], .stop-button'
    )).some(visible);
    const buttons = Array.from(container.querySelectorAll('button'));
    for (const button of buttons) {
      const label = (button && button.getAttribute('aria-label')) || '';
      if (visible(button) && /stop|cancel|останов|стоп|прекрат|abbrechen|arrêter|detener|annulla|interrompi/i.test(label)) {
        generating = true;
        break;
      }
    }
  }
  if (Array.from(document.querySelectorAll('[data-test-id="stop-button"]')).some(visible)) {
    generating = true;
  }
  const responses = Array.from(document.querySelectorAll('model-response'));
  const lastResponse = responses[responses.length - 1];
  const footer = lastResponse && lastResponse.querySelector('.response-footer');
  return {
    signIn,
    captcha,
    generating,
    responses: responses.length,
    hasFooter: !!footer,
    complete: !!(footer && footer.classList.contains('complete')),
    editor: !!document.querySelector('rich-textarea .ql-editor[contenteditable="true"], div[contenteditable="true"][role="textbox"]')
  };
})()
"""

EDITOR_SELECTOR = (
    'rich-textarea .ql-editor[contenteditable="true"], '
    '.ql-editor[contenteditable="true"], '
    'rich-textarea [contenteditable="true"], '
    'div[contenteditable="true"][role="textbox"], '
    'textarea[aria-label*="prompt" i], '
    'textarea[aria-label*="запрос" i]'
)


@dataclass
class GenerationChunk:
    text: str
    model: str | None = None


class GeminiWebDriver:
    """Serialized Playwright driver for gemini.google.com.

    A persistent Chrome profile stores the Google session. One page is used at a
    time because Gemini's web UI is not designed for parallel conversations.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pw: Playwright | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._lock = asyncio.Lock()
        self._last_model: str | None = None
        self._last_error: str | None = None

    @property
    def started(self) -> bool:
        return self._context is not None

    @property
    def page_url(self) -> str | None:
        return (
            self._page.url
            if self._page is not None and not self._page.is_closed()
            else None
        )

    def status(self) -> dict[str, object]:
        return {
            "browser_started": self.started,
            "page_url": self.page_url,
            "last_model": self._last_model,
            "last_error": self._last_error,
            "profile_dir": str(self.settings.profile_dir),
        }

    async def stream(
        self, prompt: str, model: str | None = None
    ) -> AsyncIterator[GenerationChunk]:
        prompt = prompt.strip()
        if not prompt:
            raise BridgeError(
                "The rendered prompt is empty.",
                status_code=400,
                code="empty_prompt",
                error_type="invalid_request_error",
            )
        if len(prompt) > self.settings.max_prompt_chars:
            raise BridgeError(
                f"Prompt is too large ({len(prompt)} characters; limit is "
                f"{self.settings.max_prompt_chars}).",
                status_code=413,
                code="prompt_too_large",
                error_type="invalid_request_error",
            )

        await self._acquire()
        try:
            try:
                await self._ensure_started()
                baseline = await self._new_chat()
                model_label = await self._select_model(model)
                await self._submit(prompt)
                async for chunk in self._stream_answer(
                    baseline=baseline,
                    submitted_at=time.monotonic(),
                    model_label=model_label,
                ):
                    yield chunk
            except PlaywrightError as exc:
                self._last_error = str(exc)
                await self._save_debug("playwright-error")
                await self._reset_browser()
                raise BridgeError(
                    f"Chrome/Playwright error: {exc}. The browser was restarted; retry the request.",
                    status_code=502,
                    code="browser_error",
                    error_type="api_error",
                ) from exc
        finally:
            self._lock.release()

    async def close(self) -> None:
        try:
            if self._context is not None:
                await self._context.close()
        except Exception:
            logger.debug("Error closing browser context", exc_info=True)
        finally:
            self._context = None
            self._page = None
            if self._pw is not None:
                try:
                    await self._pw.stop()
                except Exception:
                    logger.debug("Error stopping Playwright", exc_info=True)
                self._pw = None

    async def _acquire(self) -> None:
        try:
            async with asyncio.timeout(self.settings.lock_wait_timeout_s):
                await self._lock.acquire()
        except TimeoutError as exc:
            raise GeminiBusy(
                "Gemini Web is still busy with a previous request. Try again later."
            ) from exc

    async def _ensure_started(self) -> None:
        if self._context is not None:
            if self._page is None or self._page.is_closed():
                pages = self._context.pages
                self._page = pages[0] if pages else await self._context.new_page()
                self._configure_page(self._page)
            return

        self.settings.prepare_dirs()
        logger.info("Starting Chrome with profile %s", self.settings.profile_dir)
        self._pw = await async_playwright().start()

        launch_kwargs: dict[str, object] = {
            "user_data_dir": str(self.settings.profile_dir),
            "headless": self.settings.headless,
            "args": shlex.split(self.settings.browser_args),
            "ignore_default_args": ["--enable-automation"],
            "viewport": {"width": 1360, "height": 900},
            "locale": "en-US",
            "no_viewport": False,
        }
        if self.settings.browser_channel:
            launch_kwargs["channel"] = self.settings.browser_channel
        elif self.settings.browser_executable:
            launch_kwargs["executable_path"] = self.settings.browser_executable

        try:
            self._context = await self._pw.chromium.launch_persistent_context(
                **launch_kwargs
            )
        except PlaywrightError as exc:
            await self._stop_playwright()
            message = str(exc)
            hint = (
                "Check that Google Chrome is installed, or run "
                "`playwright install chromium` and clear GEMINI_BRIDGE_BROWSER_CHANNEL. "
                "Also make sure another bridge/login process is not using the same profile."
            )
            raise BridgeError(
                f"Could not start Chrome: {message}. {hint}",
                status_code=503,
                code="browser_start_failed",
                error_type="service_unavailable",
            ) from exc

        pages = self._context.pages
        self._page = pages[0] if pages else await self._context.new_page()
        self._configure_page(self._page)
        await self._open_base()

    def _configure_page(self, page: Page) -> None:
        page.set_default_timeout(self.settings.page_timeout_ms)
        page.set_default_navigation_timeout(max(30_000, self.settings.page_timeout_ms))

    async def _open_base(self) -> None:
        page = self._require_page()
        try:
            await page.goto(
                self.settings.gemini_url,
                wait_until="domcontentloaded",
                timeout=max(30_000, self.settings.page_timeout_ms),
            )
        except PlaywrightTimeoutError:
            logger.warning("Timed out waiting for Gemini page; continuing")
        await page.bring_to_front()
        await page.wait_for_timeout(700)
        await self._dismiss_consent()
        await self._assert_access()
        await self._wait_editor()

    async def _new_chat(self) -> int:
        page = self._require_page()
        clicked = False
        try:
            links = page.locator('a[href="/app"]')
            for index in range(min(await links.count(), 5)):
                link = links.nth(index)
                if await link.is_visible():
                    await link.click(timeout=3000)
                    clicked = True
                    break
        except PlaywrightError:
            clicked = False

        if clicked:
            await page.wait_for_timeout(800)

        state = await self._state()
        if state.get("responses", 0) > 0:
            clicked = False

        if not clicked:
            await self._open_base()
        else:
            await self._dismiss_consent()
            await self._assert_access()
            await self._wait_editor()

        baseline = int((await self._state()).get("responses", 0))
        logger.debug("New chat ready, existing responses=%s", baseline)
        return baseline

    async def _select_model(self, model: str | None) -> str | None:
        try:
            target = self.settings.model_target(model)
        except KeyError as exc:
            raise ModelUnavailable(
                f"Unknown model {str(exc)!r}. Available: {', '.join(self.settings.model_map)}"
            ) from exc
        if target is None:
            self._last_model = await self._current_model()
            return self._last_model

        page = self._require_page()
        picker = page.locator('[data-test-id="bard-mode-menu-button"]')
        for _ in range(10):
            if await picker.count():
                break
            await page.wait_for_timeout(100)
        if await picker.count() == 0:
            raise ModelUnavailable(
                "Gemini's model picker was not found. The web UI may have changed; "
                "leave `model` empty to use the currently selected mode."
            )

        current = (await picker.first.inner_text()).strip()
        if target.casefold() in current.casefold():
            self._last_model = current
            return current

        await picker.first.click()
        option_locator = page.locator('gem-menu-item[role="menuitem"]')
        if await option_locator.count() == 0:
            option_locator = page.locator('[role="menuitem"]')

        deadline = time.monotonic() + self.settings.model_option_timeout_s
        options: list[tuple[int, str]] = []
        while time.monotonic() < deadline:
            count = await option_locator.count()
            options = []
            for index in range(count):
                option = option_locator.nth(index)
                try:
                    if not await option.is_visible():
                        continue
                    text = (await option.inner_text()).strip()
                except PlaywrightError:
                    continue
                if text and "sign in for all models" not in text.casefold():
                    options.append((index, text))
            if options:
                break
            await page.wait_for_timeout(200)

        if not options:
            await page.keyboard.press("Escape")
            raise ModelUnavailable(
                f"Gemini's mode menu did not open or is empty. Requested mode: {target!r}."
            )

        selected_index: int | None = None
        available: list[str] = []
        for index, text in options:
            first_line = re.sub(r"^\d+(?:\.\d+)?\s+", "", text.splitlines()[0].strip())
            available.append(first_line)
            if first_line.casefold() == target.casefold():
                selected_index = index
                break
        if selected_index is None:
            for index, text in options:
                if target.casefold() in text.casefold():
                    selected_index = index
                    break

        if selected_index is None:
            await page.keyboard.press("Escape")
            raise ModelUnavailable(
                f"Mode {target!r} is not offered by this account. "
                f"Visible modes: {', '.join(available)}. "
                "Update GEMINI_BRIDGE_MODEL_MAP if the label changed."
            )

        try:
            await option_locator.nth(selected_index).click()
        except PlaywrightError as exc:
            raise ModelUnavailable(
                f"Could not select Gemini mode {target!r}: {exc}"
            ) from exc

        await page.wait_for_timeout(500)
        current_after = (await picker.first.inner_text()).strip()
        if target.casefold() not in current_after.casefold():
            raise ModelUnavailable(
                f"Gemini did not switch to {target!r} (still {current_after!r}). "
                "The Pro mode may be unavailable for this account or an upgrade dialog appeared."
            )

        self._last_model = current_after
        logger.info("Gemini mode selected: %s", current_after)
        return current_after

    async def _submit(self, prompt: str) -> None:
        page = self._require_page()
        editor = await self._wait_editor()
        await self._dismiss_consent()
        await editor.click()

        await editor.fill("")
        await editor.fill(prompt)
        await page.wait_for_timeout(150)

        actual = await editor.inner_text()
        if not actual.strip():
            await editor.fill("")
            await editor.click()
            await page.keyboard.insert_text(prompt)
            await page.wait_for_timeout(150)
            actual = await editor.inner_text()

        if not actual.strip():
            raise BridgeError(
                "Failed to enter the prompt into Gemini's input box.",
                status_code=502,
                code="prompt_insert_failed",
                error_type="api_error",
            )

        send = page.locator('[data-test-id="send-button-container"] button')
        if await send.count() == 0:
            send = page.locator(
                'button[aria-label*="Send" i], button[aria-label*="Отправ" i]'
            )

        if await send.count() > 0:
            button = send.first
            try:
                await button.wait_for(state="visible", timeout=5000)
                await button.click()
            except PlaywrightTimeoutError:
                await page.keyboard.press("Enter")
        else:
            await page.keyboard.press("Enter")

        await page.wait_for_timeout(300)

    async def _stream_answer(
        self,
        baseline: int,
        submitted_at: float,
        model_label: str | None,
    ) -> AsyncIterator[GenerationChunk]:
        page = self._require_page()
        start_deadline = time.monotonic() + self.settings.response_start_timeout_s
        first_text = ""

        while time.monotonic() < start_deadline:
            state = await self._state()
            await self._assert_access(state)
            if int(state.get("responses", 0)) > baseline:
                first_text = await self._extract_text()
                if first_text:
                    break
            await page.wait_for_timeout(self.settings.poll_interval_ms)

        if not first_text:
            await self._save_debug("no-response")
            raise GeminiTimeout(
                "Gemini did not produce an answer in time. Check the browser window: "
                "the account may be signed out, rate-limited, or a captcha may be blocking it."
            )

        last_text = first_text
        last_change = time.monotonic()
        overall_deadline = submitted_at + self.settings.request_timeout_s
        yield GenerationChunk(text=first_text, model=model_label)

        while True:
            now = time.monotonic()
            if now > overall_deadline:
                await self._save_debug("response-timeout")
                raise GeminiTimeout(
                    "Gemini did not finish the answer within "
                    f"{self.settings.request_timeout_s:.0f}s. Increase "
                    "GEMINI_BRIDGE_REQUEST_TIMEOUT_S for long Pro answers."
                )

            state = await self._state()
            await self._assert_access(state)
            text = await self._extract_text()

            if text and text != last_text:
                last_text = text
                last_change = now
                yield GenerationChunk(text=text, model=model_label)

            stable_for = now - last_change
            if state.get("hasFooter"):
                finished = bool(state.get("complete"))
            else:
                # Older UI builds do not expose .response-footer; fall back to
                # the stop button and text stability.
                finished = not bool(state.get("generating"))
            settled = (
                text
                and finished
                and stable_for >= self.settings.stability_ms / 1000
                and (now - submitted_at) >= 2.0
            )
            if settled:
                final_text = await self._extract_text()
                if final_text and final_text != last_text:
                    yield GenerationChunk(text=final_text, model=model_label)
                logger.info(
                    "Gemini answer complete: %s chars in %.1fs",
                    len(final_text or last_text),
                    time.monotonic() - submitted_at,
                )
                return

            await page.wait_for_timeout(self.settings.poll_interval_ms)

    async def _state(self) -> dict[str, object]:
        page = self._require_page()
        try:
            return dict(await page.evaluate(STATE_JS))
        except PlaywrightError:
            return {}

    async def _extract_text(self) -> str:
        page = self._require_page()
        try:
            result = await page.evaluate(EXTRACT_JS)
        except PlaywrightError:
            return ""
        if not isinstance(result, dict):
            return ""
        return str(result.get("text") or "").strip()

    async def _current_model(self) -> str | None:
        page = self._require_page()
        picker = page.locator('[data-test-id="bard-mode-menu-button"]')
        for _ in range(10):
            if await picker.count():
                break
            await page.wait_for_timeout(100)
        if await picker.count() == 0:
            return None
        try:
            return (await picker.first.inner_text()).strip()
        except PlaywrightError:
            return None

    async def _wait_editor(self) -> object:
        page = self._require_page()
        deadline = time.monotonic() + self.settings.page_timeout_ms / 1000
        while time.monotonic() < deadline:
            state = await self._state()
            await self._assert_access(state)
            candidates = page.locator(EDITOR_SELECTOR)
            for index in range(min(await candidates.count(), 8)):
                editor = candidates.nth(index)
                try:
                    if await editor.is_visible():
                        return editor
                except PlaywrightError:
                    continue
            await page.wait_for_timeout(150)
        raise BridgeError(
            "Gemini's prompt input box was not found. The web UI may have changed, "
            "or the page is still loading. Open the browser window and check it.",
            status_code=502,
            code="input_not_found",
            error_type="api_error",
        )

    async def _assert_access(self, state: dict[str, object] | None = None) -> None:
        state = state or await self._state()
        if self.settings.require_login and state.get("signIn"):
            raise LoginRequired()
        if state.get("captcha"):
            raise BridgeError(
                "Google is showing a captcha or 'unusual traffic' page. "
                "Open Chrome, complete the challenge, then retry.",
                status_code=503,
                code="captcha_required",
                error_type="service_unavailable",
            )

    async def _dismiss_consent(self) -> None:
        if not self.settings.accept_cookies:
            return
        page = self._require_page()
        button = page.locator('[data-test-id="accept-button"]')
        try:
            if await button.count() == 0:
                return
            # The consent dialog appears asynchronously after the page loads and
            # intercepts clicks on the prompt input.
            for _ in range(12):
                if await button.first.is_visible():
                    try:
                        await button.first.click(timeout=3000)
                    except PlaywrightError:
                        await button.first.click(timeout=3000, force=True)
                    await page.wait_for_timeout(500)
                    logger.info("Accepted Gemini cookie consent")
                    return
                await page.wait_for_timeout(200)
        except PlaywrightError:
            logger.debug("Could not dismiss cookie consent", exc_info=True)

    def _require_page(self) -> Page:
        if self._page is None or self._page.is_closed():
            raise BridgeError(
                "The browser page is closed. Retry the request.",
                status_code=503,
                code="browser_page_closed",
                error_type="service_unavailable",
            )
        return self._page

    async def _save_debug(self, label: str) -> None:
        if (
            not self.settings.debug_artifacts
            or self._page is None
            or self._page.is_closed()
        ):
            return
        try:
            self.settings.debug_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = self.settings.debug_dir / f"{stamp}-{label}.png"
            await self._page.screenshot(path=str(path), full_page=False)
            logger.warning("Saved browser screenshot to %s", path)
        except Exception:
            logger.debug("Could not save debug screenshot", exc_info=True)

    async def _reset_browser(self) -> None:
        try:
            if self._context is not None:
                await self._context.close()
        except Exception:
            logger.debug("Error closing context during reset", exc_info=True)
        self._context = None
        self._page = None
        await self._stop_playwright()

    async def _stop_playwright(self) -> None:
        if self._pw is not None:
            try:
                await self._pw.stop()
            except Exception:
                logger.debug("Error stopping Playwright", exc_info=True)
            self._pw = None
