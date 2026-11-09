# Stage 5.1 — conversational behavior and grounded answer readability

Implemented and evaluated on 2026-11-09. No Docker work, model downloads, index rebuilds, or public schema changes.

## What was wrong

| Reported behavior | Diagnostic finding |
| --- | --- |
| Greetings returned a knowledge-base refusal | Every safe request reached hybrid retrieval. Greetings had no medical intent, so evidence admission rejected them. |
| A medicine mentioned without a question was rejected | There was no clarification route. The entire introductory sentence became a proposed medical subject. |
| `Ibprofen` was rejected | Relevant passages were retrieved, but the misspelling failed exact subject matching. Even correctly spelled `ibuprofen` subsequently failed the brand-column-only ingredient-row gate. |
| Arabic Metformin answer was English-heavy | The prompt explicitly preferred exact English clinical wording. The model nevertheless proposed an Arabic paraphrase that the verifier could not validate. The safe fallback returned the original mixed-language source. |
| Cataflam answer looked like a table dump | Qwen added a `Support` note to an otherwise correct sentence. That extra content failed the verifier, which discarded the entire draft and returned the raw row. |
| Friendly responses looked like failures | The UI assigned the same “Not supported by the available sources” heading to every `out_of_scope` response. |

The reported concatenation was **not reproduced in the current checkout**. The actual Metformin `/chat` JSON had five newline-separated warnings. Rendering that captured response in Chrome produced exactly the same `textContent`, six block lines including the introduction, and `white-space: pre-wrap`. Neither backend nor frontend removed those newlines in this trace. The backend still lacked visible bullets and produced five repetitive source cards; those have been improved. An older running build or a different response is not ruled out.

Diagnostic artifacts: `reports/evaluation/stage51_before.json`, `stage51_dom_before.json`, `stage51_after_trace.json`, and `stage51_retrieval_probe.json`. The first two preserve the pre-change behavior; the after trace is an intermediate diagnostic, not the final evaluation.

## How the assistant now decides

1. Screen the **original message** for emergency symptoms, dosing, dangerous requests, and personal treatment decisions. These rules are unchanged and run before greeting handling.
2. Recognize a small set of complete greetings, thanks, and permission-to-ask requests. Return a friendly reply without retrieval or generation. A greeting followed by a medical question continues to medical processing.
3. Ask for clarification when a verified medicine is named without a specific question, when the reviewed registry marks a medicine description ambiguous, or when spelling candidates are uncertain.
4. Run the existing hybrid top-five search. Evidence must explicitly match the subject and requested field. If no passage qualifies and the subject is an exact known medicine name, try the existing lexical retriever once, also at five results. Apply all the same evidence checks to it.
5. Make at most one Qwen generation call. Ingredient and brand-table quotations are constrained to actual eligible source rows/blocks. This also prevents the model from inventing a header or returning literal escaped newlines as a quotation.
6. Recheck the active corpus identity; validate labels, exact quotes, subject association, subsection, and claims. Ingredient and contraindication quotes must preserve complete source rows/lines. Reject truncated salt names or warning qualifiers.
7. Publish verified prose, or discard its wording and format verified source excerpts. A verified trade-name/ingredient row can be rendered as a natural sentence. Generic ingredient queries use the value column and explicitly distinguish the ingredient from a source-listed brand. All other fallback facts remain literal source text in bullets.

Think of the model as a draft writer. The source checker decides which facts are usable; the formatter makes those verified facts readable. A confident model or a high search score cannot override the checker.

## Contract and spelling behavior

The four public fields and three statuses are unchanged. Friendly replies and clarification use `out_of_scope`, with no citations. Medical `answered` replies still require authentic citations. The UI uses the neutral heading **Response / رد المساعد** for `out_of_scope`; the exact backend body continues to explain genuinely unsupported questions. Escalation banners and technical error states are unchanged. The frontend never rewrites medical response text.

Spelling candidates come from explicitly headed medicine tables and available confirmed drug/brand aliases. The alias registry is not edited. Exact recognized names are never replaced. Automatic recovery requires one unknown subject token, at least seven characters, a unique source-vocabulary target, and exactly one missing letter. The answer explicitly states the interpretation. Substitutions, transpositions, shorter names, and multiple candidates ask for confirmation instead. Salt names, combinations, and other residual query terms are not discarded to make a match.

Examples: `Ibprofen` → acknowledged `Ibuprofen`; independent `Paractamol` → acknowledged `Paracetamol`; `Omeprozole` → clarification, not automatic substitution. This is a bounded vocabulary heuristic, not a comprehensive medication spelling authority. A name outside the approved corpus can still be unknown or ambiguous. No cross-drug therapeutic equivalence is inferred.

## Before and after

| Input | Before | After, captured live |
| --- | --- | --- |
| `goodmorning` | Source-insufficiency refusal | “Hello! What would you like to know?” |
| `can i ask you a question` | Source-insufficiency refusal | “Of course! What would you like to know?” |
| `goodmorning. i need to ask a question about metformin` | Source-insufficiency refusal | “Hello! What would you like to know about Metformin? Include the medicine name in your question.” |
| `What is the active ingredient in Ibprofen?` | `out_of_scope` | Explicit interpretation notice, then “Ibuprofen is itself an active ingredient. The source lists Brufen as a brand containing Ibuprofen.” Authentic KB-04 table citation. |
| `What is the active ingredient in Cataflam?` | “The approved source states:” followed by the raw table row | “The active ingredient in Cataflam is Diclofenac potassium.” Authentic KB-03 row citation. |
| `Metformin ممنوع امتى؟` | Five English-heavy source lines, five citations | Egyptian Arabic introduction, a blank line, five exact-source bullets, one contiguous genuine KB-07 citation. |

The Metformin fallback is:

```text
المصدر بيذكر الآتي (المصطلحات الطبية محفوظة زي ما وردت فيه):

• Severe renal impairment: eGFR أقل من 30 mL/min/1.73m².
• Metabolic acidosis: بما فيها Diabetic Ketoacidosis.
• Severe hypoxia: مثل بعض حالات Shock أو Acute respiratory failure.
• Severe hepatic impairment: زيادة خطر Lactic acidosis.
• Severe dehydration: خصوصاً مع القيء والإسهال الشديد.
```

**Arabic localization remains partial.** The prompt now requests respectful Egyptian Arabic, separate bullets, and evidence-supported terminology. It no longer universally mandates English wording. However, the verifier still cannot establish the meaning of unrestricted translations of these clinical phrases. The displayed fallback therefore deliberately preserves their original wording. No unchecked Arabic medical translation was introduced merely to make the answer sound fluent. A reviewed bilingual terminology/claim-validation layer is the next appropriate improvement.

Citation consolidation uses the shortest contiguous source substring spanning the verified quotations, capped at 2,000 characters. The result must remain a substring of the same retrieved chunk. It does not concatenate disconnected excerpts into a fabricated quotation. Existing `Sources` and `BidiText` already preserve raw text, line spacing, source metadata, and script isolation, so their implementation and CSS needed no changes.

## Evaluation

Canonical reports: `reports/evaluation/stage51_original_final.json` and `stage51_new_final.json`. The original ten-question dataset, all Stage 4A additional questions, and Stage 3.1 fixtures are unchanged. New questions are in `tests/evaluation/datasets/chat_stage51.json`; they are never imported by application code.

| Set | Status matches | Fact spot-checks for expected answers | Notes |
| --- | ---: | ---: | --- |
| Original MTI | 10/10 | 6/6 | All four expected abstention/escalation cases correct. |
| Existing Stage 4A additional | 18/18 | 7/7 | Arabic Losec now answers through lexical fallback. Historical Arabic salbutamol abstention remains an abstention after quote verification, despite recovered evidence. |
| Reported Stage 5.1 examples | 6/6 | 3/3 | All three friendly/clarification replies have the expected internal reason and no retrieval. |
| Independent Stage 5.1 cases | 23/25 | 7/9 | Two false abstentions; zero answers on negative labels. |

The 25 independent cases were written before their first live run, and their expected outcomes were not changed to fit results. They are engineering holdouts, not a blinded clinical validation set. Conversation and ambiguity grammar was already part of the implementation; the evaluation includes related but independently worded medicine questions. Broad ambiguous-description handling was subsequently tightened without changing the medical admission rules.

The remaining misses are `امتى ممنوع Glimepiride؟` and `What are the undesirable effects of cetirizine?`. Both retrieve and generate, then fail quotation/subject verification (`unverified_answer`). Class-level passages with several distinct table ingredients are not automatically treated as proof for every individual drug. Their original expected `answered` labels are retained; they are not counted as successes.

Across 59 evaluated requests: 23 answered; all citation and language consistency checks passed; 11 appropriate conversation/clarification replies; all 11 expected escalations succeeded; no HTTP 500 responses. Fifteen model paraphrases were discarded: eleven became deterministic table/brand presentations and four remained literal-source bullet fallbacks. Fallback use was **not eliminated**. Automated citation authenticity and phrase matching do not establish clinical correctness or semantic entailment.

The lexical probe recovered both targeted missed passages with approximately 60–90 ms additional search time; none of its unsupported controls admitted evidence. In the final original-set run, two requests invoked the fallback: Losec became a supported answer, while salbutamol still failed downstream verification. This is a small measured probe, not a precision estimate for arbitrary medical traffic. BM25, RRF, dense ranking, candidate counts, and the vector index are unchanged.

Observed MTI warm median: **2.77 s**, maximum **6.33 s**. Additional-set median: **1.89 s**. New independent-set median: **1.44 s**, maximum **6.92 s**. Cold Cataflam: **6.43 s**, with configured Ollama runners confirmed unloaded beforehand; retrieval 2.78 s, Qwen load 1.80 s, generation 1.10 s, 918 prompt / 72 generated tokens. OS, disk, and GPU caches were not cleared. The two evaluation processes briefly overlapped near the boundary between sets; these timings are observed local measurements, not isolated throughput guarantees. All 59 warm requests and the cold request finished below 30 seconds.

Warm token totals: original sets **12,638 prompt / 2,253 generated**; new sets **10,092 / 1,762**. Whole-device sampled peak VRAM: **5,935 MiB** original run and **6,021 MiB** new run. Sampling includes desktop/browser/other processes and can miss brief peaks.

Regression verification:

- Backend offline: **324 passed, 6 opt-in checks skipped**. Existing tests remain; two assertions now expect the improved deterministic sentence instead of the old raw-row fallback.
- Integration with `RUN_CHAT_TESTS=1` and `RUN_OLLAMA_TESTS=1`: **8 passed**, including five real Qwen requests and one real embedding check.
- Frontend production build: TypeScript and Vite passed.
- Browser tests: **21 passed**, including 22 Axe scans with no violations across the existing responsive coverage and new conversational/formatting cases.
- Real browser integration: **2 passed, 11 real requests**, with exact JSON-to-DOM equality. One additional Axe scan of the actual Arabic mobile conversation passed; **23 scans total**. See `reports/frontend/live-browser-tests.json` for captured responses and `reports/frontend/screenshots/stage51-arabic-mobile.png` for the actual UI.
- Protected-file audit: **67 existing files unchanged**, covering source DOCX/artifacts, vector-store files, retrieval code, safety policy, API schema, and original evaluation datasets. No alias changes or model replacement.

UI review retained keyboard focus, neutral status semantics, visible escalation, Unicode text equality, responsive overflow checks, and reduced-motion behavior, following the installed design skills and [Web Interface Guidelines](https://raw.githubusercontent.com/vercel-labs/web-interface-guidelines/main/command.md). Browser checks use installed Chrome with viewport emulation; they do not certify physical keyboards, screen readers, Safari, or Firefox.

## Exact deliverable changes

| File | Change |
| --- | --- |
| `src/medical_rag/services/understanding.py` | New conversational grammar, corpus vocabulary, conservative spelling candidates. |
| `src/medical_rag/services/chat.py` | Conversation/clarification routes, one bounded lexical retry, constrained brand quotations, formatting, private telemetry. |
| `src/medical_rag/generation/evidence.py` | Generic ingredient matching, exact brand quote choices, full-line validation, authentic citation consolidation, readable generation instructions. |
| `src/medical_rag/generation/formatting.py` | New deterministic ingredient rendering and literal-source bullet fallback. |
| `frontend/src/copy.ts` | Neutral bilingual heading for uncited replies. |
| `frontend/src/components/Conversation.tsx` | Neutral conversation icon instead of a source-failure icon. |
| `tests/chat/test_pipeline.py` | Update two fallback-string assertions; retain safety/grounding expectations. |
| `tests/chat/test_stage51.py` | New conversation, ambiguity, spelling, provenance, formatting, and adversarial regressions. |
| `tests/evaluation/datasets/chat_stage51.json` | Separate reported/held-out evaluation set. |
| `scripts/evaluation/chat.py` | Capture fallback retrieval provenance; summarize arbitrary dataset groups, tokens, abstentions, and fallback frequency. |
| `frontend/tests/chat.spec.ts` | Conversation status and exact whitespace/citation-count browser regressions. |
| `frontend/tests/chat.live.spec.ts` | Six additional real browser-to-FastAPI examples and Arabic mobile inspection. |
| `docs/stage51.md` | Diagnosis, implementation explanation, examples, evaluation, limitations, and file inventory. |
| `README.md`, `frontend/README.md` | Link this milestone and explain the neutral status heading. |

Ignored `tmp/stage51_*.py`, `tmp/stage51-dom.mjs`, reports, and screenshots are local diagnostic artifacts, not runtime dependencies. Earlier uncommitted Stage 4A/5 changes were preserved.

## Reproduce and next steps

From the project root:

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --basetemp F:/Omar/Medical-Rag-System/tmp/stage51-your-new-run
.\.venv\Scripts\python.exe -B scripts/evaluation/chat.py --cold --output reports/evaluation/stage51_original_repeat.json
.\.venv\Scripts\python.exe -B scripts/evaluation/chat.py --dataset tests/evaluation/datasets/chat_stage51.json --output reports/evaluation/stage51_new_repeat.json
```

Run evaluations serially for comparable latency. Use the frontend README commands for both servers and `npm test` / `npm run test:live`.

Stage 5.1 engineering changes are ready for user review. Review the actual Arabic mobile answer and the two retained false abstentions before expanding clinical language coverage. A clinician-reviewed bilingual terminology layer and explicit, reviewed class-to-member source scope would address the remaining limitations more safely than weakening the verifier. No Dockerization has begun; it remains a separate milestone.
