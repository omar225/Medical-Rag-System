# Medical RAG System

## Overview

This repository is the structural foundation for a multilingual, evidence-grounded
medical retrieval-augmented generation system. The approved source documents are
preserved under `data/knowledge_base/source/` and are organized by KB ID.

## Project status

The project is initialized for implementation. Application logic, model integrations,
processed documents, embeddings, vector indexes, and evaluation results have not yet
been added.

## Structure

- `src/medical_rag/`: application packages for configuration, API, ingestion,
  retrieval, generation, safety, sessions, analytics, handoff, and observability.
- `frontend/`: user chat and administrative interface boundaries.
- `data/knowledge_base/source/`: immutable approved source documents.
- `data/knowledge_base/processed/`: future extracted and normalized content.
- `data/knowledge_base/chunks/`: future chunks and chunk metadata.
- `data/knowledge_base/metadata/`: future catalog and document manifests.
- `storage/`: runtime vector stores, sessions, analytics, and logs.
- `tests/`: unit, integration, API, safety, retrieval, and evaluation tests.
- `scripts/`: ingestion, evaluation, benchmarking, and maintenance entry points.
- `docs/`: architecture, API, safety, evaluation, and deployment documentation.

## Setup and usage

Implementation, dependency decisions, API usage, and Docker instructions will be
documented here as the system is built.

## Documentation

- [Architecture](docs/architecture/README.md)
- [API](docs/api/README.md)
- [Medical safety](docs/safety/README.md)
- [Evaluation](docs/evaluation/README.md)
- [Deployment](docs/deployment/README.md)
- [Model decision record](MODEL_DECISION.md)
- [Token and performance report](TOKEN_USAGE.md)
