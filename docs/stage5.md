# Stage 5 — bilingual medical chat UI

The functional frontend is complete in `frontend/`. It connects to the existing
FastAPI app; every runtime medical answer comes from `POST /chat`. There is no
second backend, frontend answer generator, fake streaming, or persistent history.

## Design and skills

Used the installed Frontend Design, UI/UX Pro Max, and Web Design Guidelines
skills. The interface follows the requested navy `#102438`, teal `#087F8C`, and
off-white `#F4F7F8`, with white surfaces, muted `#526674` text and `#DCE5E8` borders.
Manrope and Noto Sans Arabic are bundled locally; no font CDN receives requests.
Lucide provides consistent, labelled controls. The paper/source illustration is
code-native SVG/CSS, not a fake medical result or statistic.

The welcome screen gives the composer and three editable starter questions a
clear hierarchy. The conversation has readable text, distinct user turns, actual
status notices, and source excerpts close to the answer. Amber escalations remain
fully visible. Copy, retry, language switching, source disclosure, and New
conversation all work; there are no decorative dashboard controls.

Search calibration: the generated healthcare design suggestion was assessed
against the actual chat brief. The RTL product searches returned no verified
match, so product-specific decisions use the user's brief and general skill
guidance. The retrieved Minimalism & Swiss Style guidance explicitly supported
RTL and informed spacing/clarity. No missing search result was treated as evidence.
The [Web Interface Guidelines](https://github.com/vercel-labs/web-interface-guidelines/blob/main/command.md)
were checked against the implementation. A failing footer contrast pair and a
320px Arabic header overflow were fixed during browser verification.

## Architecture and learning guide

```text
App (locale, draft, clear dialog)
  ├─ EmptyState → choose starter → Composer
  ├─ Composer → useChat.send()
  │              └─ api.requestChat() → /api/chat → Vite proxy → FastAPI /chat
  └─ Conversation ← validated response/error
        ├─ BidiText: exact answer and native text direction
        └─ Sources: actual returned citation metadata
```

`useChat` holds current turns and an abort controller. Its immediate submission
lock prevents duplicate requests before React has rendered the disabled button.
Retry replaces the failed turn instead of adding a duplicate question. Clearing
increments the request version, aborts the browser request, and ignores any late
completion from the previous conversation.

`api.ts` sends only `{message, session_id:null}`. It validates the known response
fields, allowed status/language values, and citation/status invariants. HTTP 422,
server errors, network failures, unexpected payloads, and a 40-second client
timeout have separate technical-error handling. None becomes `out_of_scope`.

`Conversation` preserves the exact answer text, including emergency instructions.
Escalated text is also an accessible alert. `Sources` uses native `details/summary`
disclosure and displays only actual title, KB ID, section and snippet. It creates
no page numbers or external reference links. Source provenance is explicitly
distinguished from medical validation.

`BidiText` selects each paragraph's direction from its original first letter,
then isolates English and Arabic runs with HTML `bdi`. This avoids a subtle
`dir=auto` issue: the direction algorithm skips nested isolated runs, which can
misclassify an English-first mixed source heading as Arabic. Paragraphs, source
separators, medicine names and newlines retain their text; tests compare exact
DOM `textContent` with the backend answer. Textarea direction remains `auto`.
No strings are reversed and no invisible direction characters are inserted.

The language toggle changes interface labels only. It does not translate backend
medical statements. Each question is independent, as the interface explains.
No browser storage, database, or server session is created.

See [frontend/README.md](../frontend/README.md) for the component-by-component
table, configuration, install/run instructions, and test commands.

## Verification

| Check | Result |
| --- | --- |
| TypeScript check and production build | Passed |
| Chrome browser interaction/regression tests | 19 passed |
| Real browser → Vite → FastAPI integration | Passed; five actual `/chat` requests |
| Responsive widths | 320, 375, 768, 1024, 1440 CSS px |
| Axe accessibility scans | No violations in 20 scans across widths and English/Arabic states |
| Existing backend suite | 290 passed, 6 opt-in tests skipped |
| Protected source/data/index/evaluation files | 120/120 hashes unchanged |

Browser tests cover blank input, editable starters, request shape, exact response
text, citations, duplicate submits, Shift+Enter/IME, clear/cancel/late responses,
copy, reload, HTTP 422/500/502, network/timeout/malformed responses, keyboard source
disclosure, skip link, visible focus, reduced motion, long answers and unbroken
text, and LTR/RTL overflow. The normal suite uses synthetic responses only inside
isolated test interception. Runtime code has no mock or preview-answer mode.

The separate live test uses the real `/api/health` and five real questions:
English, Arabic and Franco-Arabic Cataflam questions returned `answered`; malaria
returned `out_of_scope`; the Arabic respiratory-distress example returned
`escalated`. Exact response text and real citation metadata were asserted in the
rendered UI. Live observations are attached to the JSON browser report.

Raw reports: `reports/frontend/browser-tests.json` and
`reports/frontend/live-browser-tests.json`. They and screenshots are ignored local
artifacts. Chrome was already installed; no browser binary was downloaded.

## Actual browser screenshots

These are screenshots of the running interface, not generated mockups.

- [Desktop welcome](../reports/frontend/screenshots/empty-1440.png)
- [Mobile welcome](../reports/frontend/screenshots/empty-375.png)
- [Actual English answer and expanded source](../reports/frontend/screenshots/live-answer-desktop.png)
- [Actual Arabic/mixed answer on mobile](../reports/frontend/screenshots/live-arabic-mobile-answer.png)
- [Actual emergency escalation on mobile](../reports/frontend/screenshots/live-arabic-mobile.png)

`long-answer-mobile-test.png` separately uses an explicitly synthetic test response
to exercise unusually long content; it is not a real medical answer.

## Files created and modified

Created in `frontend/`:

- `package.json`, `package-lock.json`, `tsconfig.json`, `vite.config.ts`
- `index.html`, `.env.example`, `.gitignore`, `README.md`, `public/favicon.svg`
- `src/{main,App}.tsx`, `src/{api,copy,useChat}.ts`, `src/styles.css`
- `src/components/{Brand,BidiText,EmptyState,Composer,Conversation,Sources}.tsx`
- `playwright.config.ts`, `playwright.live.config.ts`
- `tests/chat.spec.ts`, `tests/chat.live.spec.ts`

Updated root `README.md`, `.gitignore` (ignore generated UI reports), and
`docs/architecture/README.md`; created this report. Existing frontend scaffold
directories remain intact. Stage 4A's pre-existing uncommitted work was preserved.

No backend configuration change was necessary. No Python source, public schema,
LLM prompt, safety rule, ranking code, approved document/chunk, evaluation dataset,
alias registry, vector manifest, or numerical index file changed during Stage 5.

## Run together

Backend terminal, repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn medical_rag.api.main:app --app-dir src --host 127.0.0.1 --port 8000
```

Frontend terminal, repository root, using a supported Node 22.12+ runtime:

```powershell
cd frontend
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`. The README documents the approved project-local
Node 22.23.4 runtime for this workstation, whose system Node is 21. Vite's
development proxy forwards `/api/*` to the existing backend on port 8000. Runtime
dependencies are React/React DOM, Lucide, and fonts; test dependencies stay out of
the production bundle. CSS is plain CSS, without a UI framework or router.

## Limits and remaining work

Automated Chrome/axe checks and visual screenshot review are not certification.
Physical mobile keyboards, Safari/Firefox, and assistive-technology hardware
were not tested. The browser UI inherits the existing retrieval/medical-safety
limitations and does not establish clinical correctness. Aborting a fetch cannot
guarantee cancellation of work already running on the server.

Messages are temporary React state, not persistent server history or multi-turn
model context. Long conversations are not virtualized. A production static host
needs its own reverse proxy or CORS-enabled API configuration; no deployment was
performed. A clear reload or New conversation is available for this local UI.

No Docker, administration, analytics, uploads, persistence, or further RAG features
were implemented. This milestone stops at the functional chatbot UI.
