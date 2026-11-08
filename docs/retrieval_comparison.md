# Stage 3.1 retrieval comparison - 2026-11-08

All numbers below are actual local runs against the existing BGE-M3 index. No ingestion or vector rebuild occurred. The original dense code and 24-question fixture are unchanged. The aliases, parameters, and new fixture were fixed before comparison; no held-out-driven changes were applied.

The frozen set has 21 answerable queries and three negative controls. The held-out set has 24 answerable queries (six per language) and five negative controls. Its questions were written separately before running it, with additional source drug names (Losec, Augmentin, Montelukast, Cozaar, Panadol Extra, Fastum Gel). This is an engineering-held-out set, not externally blinded clinical validation. Some established topics are reused.

Hit@k means at least one labeled relevant chunk was found; Recall@k measures the fraction of all labeled relevant chunks found, macro-averaged over answerable questions. Repeated evidence can lower recall even when one adequate source is located. Negative controls are excluded from quality denominators and included in latency. Labels are source-snippet checked and bound to the corpus fingerprint.

## All-mode results

| Dataset | Mode | Hit@1 | Hit@3 | Hit@5 | Recall@1 | Recall@3 | Recall@5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen | dense | 76.19% | 76.19% | 80.95% | 52.99% | 64.22% | 67.21% |
| frozen | lexical | 85.71% | 95.24% | 95.24% | 61.36% | 73.47% | 82.45% |
| frozen | hybrid | 95.24% | 95.24% | 100.00% | 62.99% | 79.46% | 87.41% |
| held_out | dense | 50.00% | 54.17% | 66.67% | 24.29% | 36.73% | 47.08% |
| held_out | lexical | 62.50% | 87.50% | 91.67% | 34.70% | 60.89% | 76.43% |
| held_out | hybrid | 79.17% | 91.67% | 95.83% | 35.95% | 74.52% | 83.81% |

## Results by language

Language counts below include only answerable questions. The frozen fixture has six English, six Arabic, six mixed, and three Franco questions. The held-out fixture has six in every group.

| Dataset | Language | Mode | Hit@1 | Hit@3 | Hit@5 | Recall@1 | Recall@3 | Recall@5 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen | en | dense | 100.00% | 100.00% | 100.00% | 64.05% | 77.14% | 79.52% |
| frozen | en | lexical | 83.33% | 100.00% | 100.00% | 60.71% | 72.38% | 81.43% |
| frozen | en | hybrid | 100.00% | 100.00% | 100.00% | 64.05% | 83.81% | 89.52% |
| frozen | ar | dense | 66.67% | 66.67% | 66.67% | 44.05% | 57.14% | 59.52% |
| frozen | ar | lexical | 83.33% | 83.33% | 83.33% | 55.71% | 62.38% | 69.05% |
| frozen | ar | hybrid | 83.33% | 83.33% | 100.00% | 55.71% | 67.14% | 81.19% |
| frozen | mixed | dense | 83.33% | 83.33% | 100.00% | 60.71% | 73.81% | 79.52% |
| frozen | mixed | lexical | 83.33% | 100.00% | 100.00% | 61.67% | 79.05% | 88.10% |
| frozen | mixed | hybrid | 100.00% | 100.00% | 100.00% | 64.05% | 83.81% | 91.90% |
| frozen | franco | dense | 33.33% | 33.33% | 33.33% | 33.33% | 33.33% | 33.33% |
| frozen | franco | lexical | 100.00% | 100.00% | 100.00% | 73.33% | 86.67% | 100.00% |
| frozen | franco | hybrid | 100.00% | 100.00% | 100.00% | 73.33% | 86.67% | 86.67% |
| held_out | en | dense | 83.33% | 83.33% | 83.33% | 25.71% | 50.48% | 52.86% |
| held_out | en | lexical | 83.33% | 100.00% | 100.00% | 39.05% | 65.71% | 78.10% |
| held_out | en | hybrid | 83.33% | 100.00% | 100.00% | 25.71% | 77.14% | 82.86% |
| held_out | ar | dense | 33.33% | 33.33% | 33.33% | 19.05% | 19.05% | 19.05% |
| held_out | ar | lexical | 33.33% | 66.67% | 66.67% | 19.05% | 42.38% | 49.05% |
| held_out | ar | hybrid | 50.00% | 66.67% | 83.33% | 22.38% | 50.48% | 72.86% |
| held_out | mixed | dense | 33.33% | 33.33% | 66.67% | 25.00% | 33.33% | 53.33% |
| held_out | mixed | lexical | 33.33% | 83.33% | 100.00% | 25.00% | 51.67% | 83.33% |
| held_out | mixed | hybrid | 83.33% | 100.00% | 100.00% | 40.00% | 86.67% | 90.00% |
| held_out | franco | dense | 50.00% | 66.67% | 83.33% | 27.38% | 44.05% | 63.10% |
| held_out | franco | lexical | 100.00% | 100.00% | 100.00% | 55.71% | 83.81% | 95.24% |
| held_out | franco | hybrid | 100.00% | 100.00% | 100.00% | 55.71% | 83.81% | 89.52% |

## Query timing

One neutral embedding warm-up preceded comparison; each query measurement includes approved-source/index validation and ranking, including live identity checks for dense/hybrid. Runs were sequential (dense, lexical, hybrid; frozen, then held-out), not randomized repeated performance trials. BM25 construction is included in the first lexical query. Hardware target: GTX 1080 Ti, 16 GB RAM; no new model was downloaded.

| Dataset | Mode | Mean ms | Median ms | Maximum ms |
| --- | --- | ---: | ---: | ---: |
| frozen | dense | 83.97 | 82.31 | 101.21 |
| frozen | lexical | 87.79 | 47.59 | 994.53 |
| frozen | hybrid | 140.48 | 138.63 | 168.95 |
| held_out | dense | 129.38 | 142.09 | 161.46 |
| held_out | lexical | 50.97 | 48.84 | 62.54 |
| held_out | hybrid | 199.15 | 197.19 | 214.84 |

The first lexical query took 994.53 ms to build BM25; its frozen median was 47.59 ms. Held-out lexical mean was 50.97 ms with the cache warm. A one-shot CLI process pays construction again. Dense latency variation between datasets is an observation, not a code improvement: dense code is unchanged. The prior Stage 3 warm mean (~152 ms) was a different run.

A separate read-only cold probe observed no models in `/api/ps` before its first query. First query: 2,287.51 ms; immediate second: 88.10 ms. Ollama reported first model load duration 2,176.29 ms and second 1.33 ms. This measures a model-unloaded state, not a reboot or empty OS file cache. It is not evidence of a cold-start optimization over the earlier 36.61-second Stage 3 probe. Raw timings and process snapshots: `reports/evaluation/stage31_cold.json`. That probe did not force an unload. A later manual diagnostic reload is recorded below; no scheduler or keep-alive policy was implemented.

## Representative before/after rankings

Numbers are the first relevant rank in the top five; `miss` means no labeled relevant chunk in that list. The original question is preserved verbatim.

| Dataset / query ID | Question | Dense | Lexical | Hybrid |
| --- | --- | ---: | ---: | ---: |
| frozen / cataflam_ar | المادة الفعالة في كتافلام إيه؟ | miss | 1 | 1 |
| frozen / arb_ar | إيه موانع استعمال حاصرات مستقبلات الأنجيوتنسين اتنين؟ | miss | 1 | 1 |
| frozen / salbutamol_ar | إيه الأعراض الجانبية للسالبوتامول؟ | 1 | miss | 5 |
| frozen / cataflam_franco | eh el mada el fa3ala fe Cataflam? | miss | 1 | 1 |
| frozen / metronidazole_franco | yenfa3 ashrab ko7ol ma3 metronidazole? | miss | 1 | 1 |
| held_out / h_losec_en | Which ingredient is listed for Losec? | miss | 1 | 3 |
| held_out / h_cata_ar | عاوز أعرف المَادّة الفعّالة المكتوبة جنب كــتافلام. | miss | 2 | 1 |
| held_out / h_arb_ar | ما موانعُ استعمال حاصرات مستقبلات الأنجيوتنسين ٢؟ | miss | 1 | 1 |
| held_out / h_metformin_ar | اذكر الموانع المرتبطة بالميتفورمين كما هي في المستند. | 1 | miss | 3 |
| held_out / h_losec_ar | لُوسيك فيه مادّة دوائيّة اسمها إيه؟ | miss | 2 | 4 |
| held_out / h_fastum_ar | الفاستم جل مكوّنه الفعّال إيه؟ | miss | miss | miss |
| held_out / h_cata_mixed | اسم المركب في ＣＡＴＡＦＬＡＭ مكتوب إزاي في الـ table؟ | miss | 3 | 1 |
| held_out / h_panadol_mixed | بانادول إكسترا: عايز الـ composition من الجدول، مش بنادول العادي. | miss | 1 | 1 |
| held_out / h_cata_franco | Cataflam byetkawen men eh? | miss | 1 | 1 |

## Source inspection and remaining failures

The following examples link exact returned chunk IDs to their source block positions. Block positions are ingestion positions in the source DOCX, not page numbers. Full source text and citation/build metadata remain in the machine-readable report.

- `cataflam_ar`: first relevant `KB-03-460112a8ac48257e89fc1ae6c353d6cb07407f6860d264190f327ef356d64caf-0002`; section start 16; blocks 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36. Section: 1. Pain & Fever – الألم والحرارة > 2. NSAIDs – Non-Steroidal Anti-Inflammatory Drugs.
- `arb_ar`: first relevant `KB-08-f2aee38480bb73ad3e534c6cd6e63759b8276e88172cfe7a4a96e06ba3fd7530-0002`; section start 15; blocks 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28. Section: 1. أدوية الضغط – Antihypertensive Drugs > 2. ARBs – Angiotensin II Receptor Blockers.
- `metronidazole_franco`: first relevant `KB-02-4097825171799a8781eef32803bc1a1c394a311d4cc4ffa7635a37f80fea3b9a-0007`; section start 96; blocks 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119. Section: 1. أدوية قرحة المعدة والحموضة – Peptic Ulcer & GERD > 7. Antibiotics – H. pylori Eradication.
- `h_panadol_mixed`: first relevant `KB-03-460112a8ac48257e89fc1ae6c353d6cb07407f6860d264190f327ef356d64caf-0011`; section start 166; blocks 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 185, 186. Section: 2. Headache & Migraine – الصداع > 4. Caffeine-containing Analgesics.
- `salbutamol_ar`: first relevant `KB-01-c12394488aa70ae1a4e11b3608cd892c4c0a3a9e37c2e137525e08d8069848e0-0010`; section start 119; blocks 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132. Section: 2. أدوية الانسداد الرئوي المزمن – COPD > 1. SABA – Short-Acting Beta-2 Agonists.
- `h_metformin_ar`: first relevant `KB-07-e059acfcc1408535577093d906a1f60591fa74f17e43c51fa8460cd787b1d35e-0001`; section start 2; blocks 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14. Section: 2. أدوية السكر – Antidiabetic Drugs > 1. Biguanides.
- `h_losec_en`: first relevant `KB-02-4097825171799a8781eef32803bc1a1c394a311d4cc4ffa7635a37f80fea3b9a-0001`; section start 2; blocks 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16. Section: 1. أدوية قرحة المعدة والحموضة – Peptic Ulcer & GERD > 1. Proton Pump Inhibitors (PPIs).
- `h_fastum_ar`: irrelevant top result `KB-08-f2aee38480bb73ad3e534c6cd6e63759b8276e88172cfe7a4a96e06ba3fd7530-0014`; section start 214; blocks 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235. Section: 2. أدوية الكوليسترول والدهون – Dyslipidemia > 5. Bile Acid Sequestrants.

Cataflam Arabic/Franco now reaches the NSAID source containing its trade-name row, warning text, and contraindications. ARB queries reach KB-08 section start 15 with the class table and precautions intact. Metronidazole Franco reaches the KB-02 antibiotic source containing the alcohol warning by matching the drug name, without an alias that inserts that warning. Panadol Extra retains the brand qualifier and reaches KB-03 section start 166. These are source lookup observations, not medical recommendations.

Fusion can regress top-one ranking: frozen Salbutamol Arabic falls from rank 1 to rank 5, and held-out Metformin Arabic from rank 1 to rank 3. Attached `??`/`???` prevent name alias matching; generic words contribute weak lexical signals. Held-out Losec English ranks first lexically but third after fusion, with an unrelated antibiotic section first. Mixed Cozaar improves from rank 5 to rank 2 but remains below unrelated topical analgesic content. Four of the six held-out Arabic questions were hits at rank 3; five at rank 5.

The remaining hybrid Hit@5 miss is `h_fastum_ar`: the Arabic spelling has no reviewed alias, lexical overlap is zero, and dense returns an unrelated KB-08 section at position 214. Expected Fastum Gel evidence is KB-03 section starts 37 and 53. No alias was added after seeing this failure. This exposes vocabulary coverage rather than just a scoring-parameter problem.

All four frozen dense Hit@5 misses are recovered by hybrid. Eight held-out dense misses fall to one, but hybrid still has five top-one misses among 24 answerable queries. Broad mixed-topic chunks and common table headers can outweigh a precise brand match. Hit@5 improvement does not mean each retrieved result is relevant or sufficient. There was no normalization-only ablation, so gains cannot be assigned separately to normalization and fusion.

## Negative controls

No scope classifier or cosine threshold was added. These questions deliberately have no evidence labels. A returned neighbor does not mean the system can answer the question.

| Dataset / query ID | Question | Dense count / first section | Lexical count / first section | Hybrid count / first section |
| --- | --- | --- | --- | --- |
| frozen / unrelated_weather | Will it rain in Cairo tomorrow? | 5 / KB-02 @ 174 | 3 / KB-09 @ 210 | 5 / KB-08 @ 2 |
| frozen / unrelated_car | أصلح موتور العربية إزاي؟ | 5 / KB-08 @ 147 | 0 / none | 5 / KB-02 @ 2 |
| frozen / out_of_kb_medical | إيه جرعة chemotherapy لعلاج glioblastoma؟ | 5 / KB-07 @ 15 | 5 / KB-02 @ 124 | 5 / KB-02 @ 124 |
| held_out / h_negative_router | How do I reset a Wi-Fi router? | 5 / KB-01 @ 209 | 5 / KB-08 @ 214 | 5 / KB-01 @ 209 |
| held_out / h_negative_football | نتيجة ماتش الزمالك كام امبارح؟ | 5 / KB-03 @ 67 | 1 / KB-08 @ 79 | 5 / KB-03 @ 67 |
| held_out / h_negative_recipe | عايز recipe للبيتزا. | 5 / KB-07 @ 90 | 0 / none | 5 / KB-07 @ 90 |
| held_out / h_negative_laptop | 3ayez asla7 el laptop battery | 5 / KB-01 @ 21 | 0 / none | 5 / KB-01 @ 21 |
| held_out / h_negative_vaccine | Which vaccination schedule prevents Japanese encephalitis? | 5 / KB-04 @ 57 | 0 / none | 5 / KB-04 @ 57 |

Dense and hybrid return five neighbors for every negative control. Lexical zero results indicate lack of matching terms only; some negative controls still have lexical overlap and return unrelated passages. These results must not be interpreted as successful abstention or safety detection.

## Validation and reproducibility

- Complete offline suite: **213 passed, 1 skipped**, one pre-existing Starlette/HTTPX deprecation warning. The skipped case is the opt-in live Ollama embedding test.
- 33 new normalization, alias, BM25, RRF, metadata, approval, stale-index, and hybrid single-inference test cases pass. Existing API contract tests remain unchanged and pass.
- Actual integration evidence includes all six comparison runs against the installed Ollama model and published source index. No mocked benchmark results are reported.
- A subsequent opt-in live integration check **failed**: HTTP 200 responses contained 1,024 zero-valued elements per vector. The unchanged embedding validator correctly rejected these. A lexical CLI search passed; a hybrid CLI search failed on the same runtime response. One manual unload/reload of the idle BGE-M3 runner did not clear the failure. This is separate from the successful earlier comparison; it must not be counted as a passing live check.
- A direct runtime probe confirmed zero norms for a neutral single input and both inputs of an English/Arabic batch. `reports/evaluation/stage31_runtime_probe.json` records version, model digest, loaded-runner metadata, response timings, dimensions, and nonzero counts. The observed runner reports context 4,096 while the model advertises training context 8,192. The cause is unresolved; no runtime configuration, model, client, or baseline change was made to conceal it.
- Index, original benchmark/report, and dense code SHA-256 values match pre-change captures. Aliases and held-out fixture match their pre-evaluation captures.

Corpus SHA-256: `3e4bb5da73e9b335e1e1bd15ed1bbfe971e262870d23336e0fd74aba8fa68888`.
Model: `bge-m3:latest`; digest `7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab`.
Normalization identity: `ec5b01d34a7defcc81258bc95404cae6495b440987b2e21f5eb9b2923c762453`.

Fixed parameters: BM25 k1=1.2, b=0.75; RRF k=60; 20 candidates per branch. These were selected before evaluation, not optimized on either result set.

| Frozen input | SHA-256 |
| --- | --- |
| aliases_sha256 | `448ba781d0c22a2e0864f6163a0fa4ad2285873a25c859a92736385e2dcef39c` |
| held_out_sha256 | `5d3ebef868bd4e55f1913146ad43a7ed69f0739a514fd0295f58ba91eaf00c32` |
| frozen_dataset_sha256 | `dcb4dcd582630f334ad97a5c7175017789b4a63468091db60d1a12b3bfb7949f` |
| dense_code_sha256 | `73acfbcec5ce3a3e1c1d0c58e9ea2561edabe40205a3aeb492b223a410ddeadc` |
| Original Stage 3 report | `c122aab6ce469f7c468d866c39624375d1e76cdea87cf151e0d3398843a454ee` |
| Published index manifest | `96589377e8f3f45b8e47941f7d36fec06a2817292078ef9e37156cd6de903ec5` |

The full comparison is `reports/evaluation/stage31_comparison.json`, including each question, latency, expected IDs, returned text, source spans, source/build checksums, component ranks/scores, and normalization trace. Generated report artifacts remain ignored by Git under existing project rules; this Markdown snapshot is reviewable source documentation.

## Files and decision

New implementation: `retrieval/normalization.py`, `retrieval/search/lexical.py`, `retrieval/search/hybrid.py`. New reviewed data/tests: `retrieval_aliases.json`, `retrieval_heldout.json`, `test_multilingual.py`. Small updates: retrieval settings (alias path), evaluator (accept any search-compatible retriever), maintenance CLI (mode selection/comparison), README, `.env.example`, model decision record, and retrieval documentation. No dependency, API, ingestion, source, or vector changes belong to Stage 3.1.

Hybrid is the stronger candidate for a controlled Stage 4 development experiment while dense remains the default experimental baseline. **Do not proceed to Stage 4 until the current zero-vector Ollama failure is resolved and the live integration check passes.** The small non-blinded benchmark, remaining brand spelling miss, top-one regressions, mixed-topic evidence, and negative-control neighbors also prevent a claim of clinical readiness. Stage 4 will need evidence-grounded answer construction and evaluation of unsupported questions; none was built here. Independent terminology review and broader unseen evaluation are warranted before real medical use.

See [the engineering walkthrough](retrieval_multilingual.md) for equations, commands, source-preservation rules, and how the generic components can support a future explicitly approved corpus.
