# Gemini Web Bridge

OpenAI-совместимый HTTP-сервер, который принимает запросы как обычный API, а выполняет их в **веб-интерфейсе Gemini** через залогиненный Chrome-профиль.

Это не официальный API Google. Прослойка управляет браузером через Playwright: вводит промпт в поле Gemini, нажимает «Send», ждёт ответ и извлекает Markdown из DOM. Поэтому используется именно ваша веб-подписка **Google AI Pro**, а не отдельный API-ключ.

> **Важно:** автоматизация веб-интерфейса может нарушать условия использования Google, а UI Gemini меняется без предупреждения. Используйте на свой риск, не публикуйте сервер в интернет и не подключайте к нему чужие аккаунты.

## Что поддерживается

| Возможность | Статус |
|---|---|
| `POST /v1/chat/completions` | Да, включая SSE streaming |
| `POST /v1/completions` | Да, базовый legacy-режим |
| `POST /v1/responses` | Да, базовый режим + streaming events |
| `GET /v1/models` | Да |
| `GET /health` | Да |
| Выбор модели (`gemini-web-pro`, `gemini-web-flash`, …) | Да, через меню режимов Gemini |
| История сообщений | Передаётся в одном промпте, новый чат на каждый запрос |
| Streaming | Да, по мере роста текста в DOM |
| Markdown / code fences | Да, DOM конвертируется в Markdown |
| Function calling / tools | Нет |
| Изображения, аудио, файлы | Нет |
| Точный подсчёт токенов | Нет, оценка по символам |
| Параллельные запросы | Нет, один браузер и очередь (concurrency = 1) |

## Архитектура

```
OpenAI client
     |
     v
FastAPI  /v1/chat/completions  (auth, OpenAI schemas, SSE)
     |
     v
Prompt builder  (messages -> один текстовый промпт)
     |
     v
GeminiWebDriver (Playwright, один Chrome-профиль)
     |  click "New chat"
     |  select mode (Pro/Flash/...)
     |  fill prompt + click "Send"
     |  poll DOM, detect completion
     v
gemini.google.com/app  (ваша веб-подписка Pro)
```

## Требования

- macOS, Linux или Windows.
- Python 3.11+.
- Google Chrome (по умолчанию используется установленный Chrome, а не bundled Chromium).
- Аккаунт Google с Gemini Pro и доступом к веб-интерфейсу Gemini из вашей сети. Если Gemini недоступен напрямую, сначала настройте VPN/system proxy для Chrome.

## Установка

```bash
cd gemini-web-bridge
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Если Chrome не установлен, можно использовать Chromium из Playwright:

```bash
playwright install chromium
# затем в .env:
# GEMINI_BRIDGE_BROWSER_CHANNEL=
```

## 1. Войти в Gemini один раз

Сервер и helper используют отдельный профиль (`~/.gemini-web-bridge/profile` по умолчанию), а не ваш повседневный Chrome. Это важно: Playwright не любит, когда тот же профиль уже открыт другим Chrome.

```bash
python -m app.login
```

Откроется Chrome. Войдите в Google-аккаунт с Pro, пройдите 2FA, затем вернитесь в терминал и нажмите Enter. Проверить сессию без открытия окна:

```bash
python -m app.login --check
```

Не запускайте login helper одновременно с сервером: профиль может быть заблокирован. Если сервер уже работает, остановите его, выполните вход, затем запустите снова.

## 2. Настроить `.env`

Минимально достаточно:

```dotenv
GEMINI_BRIDGE_API_KEY=sk-local-change-me
GEMINI_BRIDGE_DEFAULT_MODEL=gemini-web-pro
```

Полный список — в `.env.example`. Основные параметры:

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `GEMINI_BRIDGE_HOST` | `127.0.0.1` | Адрес сервера |
| `GEMINI_BRIDGE_PORT` | `8000` | Порт |
| `GEMINI_BRIDGE_API_KEY` | пусто | Если задан, требуется `Authorization: Bearer ...` |
| `GEMINI_BRIDGE_PROFILE_DIR` | `~/.gemini-web-bridge/profile` | Chrome-профиль с сессией Google |
| `GEMINI_BRIDGE_HEADLESS` | `false` | Запускать Chrome без окна. Для Gemini надёжнее `false` |
| `GEMINI_BRIDGE_BROWSER_CHANNEL` | `chrome` | `chrome`, `chromium` или пусто для bundled Chromium |
| `GEMINI_BRIDGE_DEFAULT_MODEL` | `gemini-web-pro` | Модель, если клиент не указал `model` |
| `GEMINI_BRIDGE_MODEL_MAP` | JSON | Соответствие API-моделей меткам в меню Gemini |
| `GEMINI_BRIDGE_REQUEST_TIMEOUT_S` | `300` | Максимальное время одного ответа |
| `GEMINI_BRIDGE_RESPONSE_START_TIMEOUT_S` | `120` | Сколько ждать начала ответа |
| `GEMINI_BRIDGE_STABILITY_MS` | `1500` | Сколько текст должен не меняться, чтобы считать ответ готовым |
| `GEMINI_BRIDGE_MAX_PROMPT_CHARS` | `200000` | Лимит размера промпта |
| `GEMINI_BRIDGE_DEBUG_ARTIFACTS` | `true` | Сохранять screenshot при ошибках |

### Модели

По умолчанию:

| API model | Что выбирается в Gemini |
|---|---|
| `gemini-web-pro` | режим с меткой `Pro` (сейчас `3.1 Pro`) |
| `gemini-web-flash` | `Flash` |
| `gemini-web-flash-lite` | `Flash-Lite` |
| `gemini-web-argon` | `Argon` |
| `gemini-web`, `gemini-web-auto` | текущий выбранный режим, без переключения |
| `gemini-2.5-pro`, `gemini-3-pro` | алиасы на `Pro` |
| `gemini-2.5-flash`, `gemini-3-flash` | алиасы на `Flash` |

Метки в меню Gemini зависят от аккаунта и региона. Если у вас, например, `3.2 Pro` вместо `3.1 Pro`, матчинг по слову `Pro` всё равно сработает. Если метка другая, задайте свой JSON:

```dotenv
GEMINI_BRIDGE_MODEL_MAP={"gemini-web":null,"gemini-web-pro":"Pro","gemini-web-thinking":"Deep Think"}
```

## 3. Запуск

```bash
python -m app
# или
./run.sh
```

Сервер поднимется на `http://127.0.0.1:8000`. Swagger: http://127.0.0.1:8000/docs.

Первый запрос запустит Chrome. Если окно не появилось или Gemini показывает логин, выполните `python -m app.login`. Браузер остаётся открытым между запросами; каждый запрос начинается с нового чата.

## Примеры

### curl

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-local-change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-web-pro",
    "messages": [
      {"role": "system", "content": "Отвечай кратко."},
      {"role": "user", "content": "Почему небо голубое?"}
    ]
  }'
```

Streaming:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-local-change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-web-pro",
    "stream": true,
    "messages": [{"role": "user", "content": "Напиши Python-функцию fib"}]
  }'
```

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="sk-local-change-me",
)

response = client.chat.completions.create(
    model="gemini-web-pro",
    messages=[
        {"role": "system", "content": "Ты senior Python-разработчик."},
        {"role": "user", "content": "Напиши FastAPI endpoint с SSE."},
    ],
)
print(response.choices[0].message.content)

stream = client.chat.completions.create(
    model="gemini-web-pro",
    messages=[{"role": "user", "content": "Расскажи про asyncio"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### OpenAI Node SDK

```js
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://127.0.0.1:8000/v1",
  apiKey: "sk-local-change-me",
});

const response = await client.chat.completions.create({
  model: "gemini-web-pro",
  messages: [{ role: "user", content: "Hello" }],
});
console.log(response.choices[0].message.content);
```

## API

- `GET /health` — статус сервиса и браузера; не требует ключ.
- `GET /v1/models` — список доступных id моделей.
- `GET /v1/bridge/status` — состояние браузера, профиль, последняя выбранная модель.
- `POST /v1/chat/completions` — основной OpenAI Chat Completions.
- `POST /v1/completions` — legacy text completions.
- `POST /v1/responses` — базовый Responses API.

Ошибки возвращаются в формате OpenAI:

```json
{
  "error": {
    "message": "Gemini Web is not signed in...",
    "type": "service_unavailable",
    "param": null,
    "code": "gemini_login_required"
  }
}
```

Основные коды: `gemini_login_required`, `captcha_required`, `gemini_timeout`, `gemini_busy`, `browser_start_failed`, `model_unavailable`, `unsupported_tools`, `prompt_too_large`.

## Как это работает внутри

1. `messages` превращаются в один промпт. System/developer-сообщения становятся блоком `SYSTEM INSTRUCTIONS`, остальные — транскриптом `CONVERSATION`. Для одного user-сообщения без system промпт отправляется как есть.
2. Драйвер открывает новый чат, при необходимости переключает режим Gemini и вставляет промпт в `rich-textarea .ql-editor`.
3. Нажимается кнопка из `[data-test-id="send-button-container"]`.
4. Драйвер наблюдает за `model-response`, извлекает последний ответ и конвертирует DOM в Markdown. Готовность определяется по исчезновению stop-кнопки и стабилизации текста.
5. API отдаёт обычный OpenAI JSON или SSE-дельты.

## Ограничения и подводные камни

- **Один запрос за раз.** Это ограничение одного браузерного окна. Второй запрос ждёт освобождения блокировки; таймаут ожидания — `GEMINI_BRIDGE_LOCK_WAIT_TIMEOUT_S`.
- **UI Gemini может измениться.** Селекторы вынесены в `app/gemini_web.py` и `app/extract.js`; при обновлении интерфейса их нужно править.
- **Deep Think / длинные ответы.** Увеличьте `GEMINI_BRIDGE_REQUEST_TIMEOUT_S` и `GEMINI_BRIDGE_RESPONSE_START_TIMEOUT_S`.
- **Google может показать captcha.** Откройте окно Chrome, пройдите проверку, повторите запрос.
- **Не используйте основной Chrome-профиль.** Playwright создаёт отдельный профиль; это защищает ваши куки и сессии.
- **Веб-подписка ≠ API-квота.** Лимиты веб-интерфейса действуют как в обычном чате; при частых запросах возможны rate limit и предупреждения Google.
- **Токены считаются приблизительно** (`len(text)/4`). Не используйте эти цифры для биллинга.
- **Параметры `temperature`, `top_p`, `max_tokens`, `stop`, `logprobs` игнорируются** веб-интерфейсом. `response_format=json_object/json_schema` добавляет инструкцию и снимает Markdown-ограждение с ответа, но гарантии валидного JSON нет.
- **Инструменты не поддерживаются.** Запрос с `tools`/`function_call` завершится ошибкой 400, чтобы клиент не завис в ожидании tool call.
- **Контекст диалога.** Каждый API-запрос — новый чат; история передаётся в промпте. `previous_response_id` в Responses API не поддерживается.
- **Приватность.** Промпты и ответы обрабатываются локальным Chrome, но уходят в Google как обычные сообщения Gemini. Логи сервера не содержат текст промптов.

## Разработка

```bash
pip install -r requirements-dev.txt
pytest -q
```

Тесты:

- `tests/test_api.py` — OpenAI-эндпоинты на fake-драйвере, стриминг, auth, ошибки.
- `tests/test_prompting.py` — сборка промпта, content parts, JSON-режим.
- `tests/test_browser_driver.py` — реальный Playwright на локальной mock-странице Gemini: ввод, выбор Pro, ожидание, Markdown. Требует установленный Chrome; иначе тест пропускается.

## Безопасность

- По умолчанию сервер слушает только `127.0.0.1`.
- Всегда задавайте `GEMINI_BRIDGE_API_KEY`, если меняете `HOST` или пробрасываете порт.
- API-ключ фактически даёт доступ к вашему Gemini-аккаунту. Через подключённые к Gemini приложения и расширения (Workspace, Gmail, Drive и т.п.) клиент с ключом может попросить Gemini работать с вашими данными. Не выдавайте ключ третьим лицам.
- Не коммитьте `.env` и каталог профиля: в профиле лежат cookies Google-аккаунта.
- Не давайте серверу доступ к чужим аккаунтам и не используйте его как публичный прокси.

## Лицензия

Код можно использовать и изменять свободно. Ответственность за соблюдение условий Google и локального законодательства остаётся на пользователе.
