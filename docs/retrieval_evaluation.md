# Stage 3 retrieval evaluation - 2026-11-08

This is a dated baseline on the approved Stage 2.1 corpus. The 24 questions and
manual evidence labels were fixed before retrieving results. No query-specific
ranking rules, safety threshold, answer generation, or benchmark-driven query
rewrites were introduced.

## Reproduction and identity

```powershell
.\.venv\Scripts\python.exe scripts/maintenance/retrieval.py build
.\.venv\Scripts\python.exe scripts/maintenance/retrieval.py evaluate
```

- Corpus SHA-256: `3e4bb5da73e9b335e1e1bd15ed1bbfe971e262870d23336e0fd74aba8fa68888`.
- Model: `bge-m3:latest`; digest `7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab`.
- Local Ollama runtime observed: `0.40.15`; no model download was performed.
- Index: 9 documents, 113 chunks, 1024 dimensions, float32 unit vectors.
- Indexing time: **8.79 seconds**; embedding calls: **8.74 seconds**.
- Vector file: 462,976 bytes (462,848 numerical payload bytes).
- These are warm-model timings. The initial model-load probe took 36.61 seconds.
- Query latency: mean **151.5 ms**, median **147.1 ms**, maximum **236.0 ms**.

Per-query timing includes current corpus/source checks, index validation, model
identity checks, question embedding, exact cosine search, and metadata copying.
It excludes Python process startup and final report serialization.

## Metrics

Recall@k is the fraction of all labeled relevant chunks found among the top k.
Hit@k is whether any relevant chunk appears. Metrics are macro averages over
the 21 answerable queries; the three out-of-KB controls are recorded separately.
The Brufen topic has seven relevant chunks, so Recall@5 has a 5/7 ceiling for
those queries. Labels describe source evidence, not clinical answer suitability.

| Group | Queries | Recall@1 | Recall@3 | Recall@5 | Hit@1 | Hit@3 | Hit@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All answerable | 21 | 52.99% | 64.22% | 67.21% | 76.19% | 76.19% | 80.95% |
| en | 6 | 64.05% | 77.14% | 79.52% | 100.00% | 100.00% | 100.00% |
| ar | 6 | 44.05% | 57.14% | 59.52% | 66.67% | 66.67% | 66.67% |
| mixed | 6 | 60.71% | 73.81% | 79.52% | 83.33% | 83.33% | 100.00% |
| franco | 3 | 33.33% | 33.33% | 33.33% | 33.33% | 33.33% | 33.33% |

## Every query

The source shorthand is document ID plus original `section_start` body position.
The full JSON includes exact chunk IDs, all top-five scores, complete source
text, document/build identities, and expected evidence IDs.

| ID | Question | Language | Relevant chunks | R@1 / R@3 / R@5 | Latency ms | Top-1 source | Cosine |
| --- | --- | --- | ---: | --- | ---: | --- | ---: |
| cataflam_en | What is the active ingredient in Cataflam? | en | 5 | 20.0% / 20.0% / 20.0% | 236.0 | KB-03 @16 | 0.4793 |
| cataflam_ar | المادة الفعالة في كتافلام إيه؟ | ar | 5 | 0.0% / 0.0% / 0.0% | 154.3 | KB-03 @37 | 0.4718 |
| cataflam_mixed | Cataflam مادته الفعالة إيه بالإنجليزي؟ | mixed | 5 | 0.0% / 0.0% / 20.0% | 158.1 | KB-03 @53 | 0.4803 |
| brufen_en | Which other brands contain the same active ingredient as Brufen? | en | 7 | 14.3% / 42.9% / 57.1% | 150.5 | KB-04 @25 | 0.4987 |
| brufen_ar | إيه بدائل بروفين اللي فيها نفس المادة الفعالة؟ | ar | 7 | 14.3% / 42.9% / 57.1% | 151.4 | KB-03 @16 | 0.5032 |
| brufen_mixed | Brufen بديله إيه بنفس الـ active ingredient؟ | mixed | 7 | 14.3% / 42.9% / 57.1% | 145.8 | KB-04 @25 | 0.5582 |
| arb_en | What are the contraindications of angiotensin II receptor blockers (ARBs)? | en | 1 | 100.0% / 100.0% / 100.0% | 145.9 | KB-08 @15 | 0.6247 |
| arb_ar | إيه موانع استعمال حاصرات مستقبلات الأنجيوتنسين اتنين؟ | ar | 1 | 0.0% / 0.0% / 0.0% | 153.1 | KB-09 @185 | 0.4662 |
| arb_mixed | إيه الـ contraindications بتاعة ARBs؟ | mixed | 1 | 100.0% / 100.0% / 100.0% | 146.5 | KB-08 @15 | 0.6042 |
| salbutamol_en | What side effects can Salbutamol cause? | en | 2 | 50.0% / 100.0% / 100.0% | 150.2 | KB-01 @7 | 0.6720 |
| salbutamol_ar | إيه الأعراض الجانبية للسالبوتامول؟ | ar | 2 | 50.0% / 100.0% / 100.0% | 144.4 | KB-01 @119 | 0.6348 |
| salbutamol_mixed | Salbutamol ممكن يعمل side effects إيه؟ | mixed | 2 | 50.0% / 100.0% / 100.0% | 146.3 | KB-01 @7 | 0.6711 |
| metformin_en | What are the contraindications for Metformin? | en | 1 | 100.0% / 100.0% / 100.0% | 144.7 | KB-07 @2 | 0.6455 |
| metformin_ar | الميتفورمين ممنوع في أنهي حالات؟ | ar | 1 | 100.0% / 100.0% / 100.0% | 143.1 | KB-07 @2 | 0.5513 |
| metformin_mixed | إيه موانع استعمال Metformin؟ | mixed | 1 | 100.0% / 100.0% / 100.0% | 154.1 | KB-07 @2 | 0.6174 |
| metronidazole_en | What does the knowledge base say about Metronidazole and alcohol? | en | 1 | 100.0% / 100.0% / 100.0% | 148.1 | KB-02 @96 | 0.4824 |
| metronidazole_ar | ينفع أشرب كحول وأنا باخد ميترونيدازول؟ | ar | 1 | 100.0% / 100.0% / 100.0% | 152.5 | KB-02 @96 | 0.5329 |
| metronidazole_mixed | Metronidazole مع الكحول، إيه التحذير المذكور؟ | mixed | 1 | 100.0% / 100.0% / 100.0% | 141.9 | KB-02 @96 | 0.5105 |
| cataflam_franco | eh el mada el fa3ala fe Cataflam? | franco | 5 | 0.0% / 0.0% / 0.0% | 142.3 | KB-03 @2 | 0.4200 |
| metformin_franco | metformin mamno3 fe anhy 7alat? | franco | 1 | 100.0% / 100.0% / 100.0% | 146.3 | KB-07 @2 | 0.5008 |
| metronidazole_franco | yenfa3 ashrab ko7ol ma3 metronidazole? | franco | 1 | 0.0% / 0.0% / 0.0% | 141.4 | KB-09 @154 | 0.4767 |
| unrelated_weather | Will it rain in Cairo tomorrow? | en | 0 | not scored | 147.7 | KB-02 @174 | 0.4233 |
| unrelated_car | أصلح موتور العربية إزاي؟ | ar | 0 | not scored | 143.7 | KB-08 @147 | 0.4284 |
| out_of_kb_medical | إيه جرعة chemotherapy لعلاج glioblastoma؟ | mixed | 0 | not scored | 148.3 | KB-07 @15 | 0.4878 |

## Inspected evidence and failures

- English Cataflam: rank 1 is KB-03 @16 (NSAIDs), with the original row `Cataflam | Diclofenac potassium`. Only one of five labeled source chunks appears in the top five.
- Arabic Metformin: rank 1 is KB-07 @2 (Biguanides), containing the drug/ingredient rows and its own contraindication paragraphs.
- Mixed-language ARBs: rank 1 is KB-08 @15, containing the ARB table and contraindications through source block 28.
- English Brufen ingredient alternatives: the top result is KB-04 @25; its table contains Brufen and Nurofen with Ibuprofen. This is ingredient evidence, not a recommendation to substitute a particular product.
- Arabic Salbutamol: both labeled standalone SABA sections are ranks 1 and 2 (KB-01 @119 and @7).
- Arabic Cataflam: top result is KB-03 @37 (Topical NSAIDs), which does not contain Cataflam. None of the five labeled ingredient chunks is returned in the top five.
- Arabic ARB paraphrase: top result is KB-09 @185 (JAK Inhibitors); the ARB evidence is absent from the top five.
- Mixed Cataflam: relevant evidence appears only at rank 4; the top result is Topical Analgesics.
- Franco-Arabic Metformin succeeds at rank 1. Franco-Arabic Cataflam and Metronidazole miss all labeled evidence in the top five; their top results are Paracetamol and Conventional DMARDs respectively.

The negative controls also return five neighbors: weather retrieves ORS first,
car repair retrieves Ezetimibe first, and the out-of-KB chemotherapy question
retrieves Sulfonylureas first. These are false-positive evidence candidates,
not answers or operational scope classifications. No threshold is used to
misrepresent such results as medically safe.

## Validation and remaining limits

- Full offline suite: **180 passed, 1 skipped, 1 existing warning in 10.49 seconds**. All previous 103 tests remain passing.
- Opt-in live Ollama test: **1 passed in 2.62 seconds**.
- Independent live re-embedding of KB-03 @16, KB-07 @2, and KB-08 @15 reproduced stored vectors at cosine approximately 1.0.
- All 113 source chunks were accepted with truncation disabled. Mock tests cover over-context rejection, failures/timeouts, invalid counts/dimensions/nonfinite vectors, and model identity mismatches.
- Index tests cover synthetic cosine ranking, deterministic row alignment, metadata preservation, missing/corrupt data, stale source/build identities, and bootstrap-only selection.
- The existing KB-04 blank source classification value remains a preserved review issue.

This small, related-query benchmark does not establish clinical readiness or
general multilingual reliability. Dense-only retrieval currently needs further
review for Arabic brand spellings and Franco-Arabic input. Relevant retrieval
does not establish answerability, safety, or completeness. No Stage 4 work was
performed.

[Full measured JSON and retrieved source text](../reports/evaluation/stage3_retrieval.json).
[Frozen questions and manual evidence labels](../tests/evaluation/datasets/retrieval_queries.json).
[Implementation guide](retrieval.md).
