# MTI Medical chat UI

React + Vite + TypeScript, plain CSS, Lucide icons, and self-hosted Manrope / Noto
Sans Arabic fonts. The existing FastAPI service is the only answer provider.

## Install and run

Use **Node 22.12 or newer** and the existing Python environment. From the repository
root, start the backend in one terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn medical_rag.api.main:app --app-dir src --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open **http://127.0.0.1:5173**. Ollama and the existing embedding/generation models
must be available for supported medical questions; the UI never downloads models.

This workstation has system Node 21. A project-local Node 22.23.4 was installed
with approval under `.tooling/` for development. To use that already-installed
runtime, run this inside `frontend/` before the npm commands:

```powershell
$frontendNodeBin = (Resolve-Path .\.tooling\node_modules\node\bin).Path
$env:Path = $frontendNodeBin + ';' + $env:Path
node --version
```

`.tooling/` is local and ignored. Fresh checkouts should use a supported installed
Node runtime. `package-lock.json` fixes the frontend dependency versions.

## Connection configuration

By default the browser posts to `/api/chat`; Vite forwards it to
`http://127.0.0.1:8000/chat`, stripping `/api`. The same configuration applies to
`npm run preview`. **No backend or CORS changes were needed.**

Optional `frontend/.env`:

```dotenv
API_PROXY_TARGET=http://127.0.0.1:8000
```

`VITE_API_BASE_URL` can instead point the browser directly to a separately
configured API origin. Such an origin needs its own CORS configuration; this
milestone does not change the backend to allow it. Vite environment variables
are public build configuration, never a place for secrets. Restart Vite after
environment changes. A static production host will need a reverse proxy or an
appropriately configured API origin; the development proxy is not deployed code.

## Components and request flow

| File | Job |
| --- | --- |
| `src/App.tsx` | Interface language, draft, layout, and clear-conversation confirmation |
| `src/useChat.ts` | In-memory turns, submission lock, retries, cancellation, and late-response protection |
| `src/api.ts` | Exactly one non-streaming POST, response/status validation, and technical error classification |
| `src/components/Composer.tsx` | Labelled textarea, blank-input validation, Enter/Shift+Enter/IME handling, and sending state |
| `src/components/Conversation.tsx` | User/assistant turns, visible status, loading/error state, retry and copy actions |
| `src/components/Sources.tsx` | Native expandable citations using only returned title, ID, section and snippet |
| `src/components/BidiText.tsx` | Per-paragraph direction and script isolation while preserving exact text/newlines |
| `src/components/EmptyState.tsx` | Starter questions that fill the composer, not prewritten answers |
| `src/copy.ts` | English/Arabic interface text; no medical answers |
| `src/styles.css` | Palette, spacing, responsive layout, focus, safety states, and reduced motion |

`App` passes the draft to `useChat.send()`. The hook locks concurrent requests and
calls `requestChat()` with `{message, session_id:null}`. Once FastAPI returns,
`Conversation` displays the exact `answer` and `Sources` displays its citation
metadata. The interface language toggle translates controls only, not medical
answers. Answer language is preserved from the API.

No conversation history is sent to the model. Each question is independent.
Turns live only in React state; refreshing or confirming New conversation clears
them. Clearing also aborts the browser request and invalidates late results. It
does not guarantee that already-started server computation stops immediately.

Answered responses need citations; other statuses must have none. HTTP 422/500,
network problems, malformed responses, and the 40-second browser timeout are
technical errors with retry controls, never substituted medical scope responses.
Runtime text is rendered as text, not executable HTML or untrusted Markdown.

## Build and verify

```powershell
npm run build
npm test
npm run test:live
```

Browser tests use the installed **Google Chrome** by default; no browser binary
was downloaded. `BROWSER_CHANNEL` can select another installed Playwright-supported
browser. The normal tests intercept API requests only within isolated test browser
contexts. The live suite requires FastAPI and Ollama and makes eleven real requests
through the Vite proxy. Tests reuse a running Vite server or start one if needed.

Results and actual screenshots are saved in `../reports/frontend/` (ignored local
artifacts). See [the Stage 5 report](../docs/stage5.md) for results and screenshots.

Stage 5.1 uses a neutral **Response / رد المساعد** heading for `out_of_scope`.
That status covers greetings, clarification, and unsupported questions under the
unchanged API contract. The exact backend body explains which applies; the frontend
does not classify or rewrite medical text. See [Stage 5.1 verification](../docs/stage51.md).

## Boundaries

No server persistence, history sidebar, streaming, uploads, frontend medical-answer
logic, admin, analytics, Docker, or additional RAG features. Citations show source
provenance, not medical validation. Automated accessibility checks do not replace
screen-reader and physical-device testing.
