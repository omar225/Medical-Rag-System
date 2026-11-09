# Medical Safety Policy

Stage 4A runs conservative multilingual rules before retrieval. Severe breathing
distress, blue lips, obvious emergencies, dosing requests, personal diagnosis or
treatment decisions, and dangerous requests escalate without citations. General
medical information must pass explicit evidence subject/field checks and source
verification. Operational errors return HTTP 500, not a medical scope decision.

Rules are incomplete, can over-escalate negated/hypothetical cases, and require
clinician review. A citation proves a source excerpt exists, not clinical truth.
See [the detailed safety and grounding limits](../stage4a.md).
