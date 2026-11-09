# Evaluation Methodology

Stage 3 retrieval evaluation remains frozen. Stage 4A separately evaluates actual
`POST /chat` responses on ten user-supplied MTI questions and 18 additional cases.
It reports status, fact phrase spot-checks, citation validity, abstention, safety,
actual answer language, latency, token counts, model load, and sampled VRAM.

Run `scripts/evaluation/chat.py --cold`. It never imports evaluation fixtures into
application decisions or modifies retrieval data. Full clinical correctness needs
human review; these development checks are not blinded clinical validation.
See [the Stage 4A walkthrough](../stage4a.md) and [results](stage4a_results.md).
