# Learning AI Python

Python and AI learning path for transcription, Bedrock, notebooks, agent workflows, retrieval, and evals.

Last verified: 2026-06-20

## Baseline

- Python 3.14 stable/bugfix
- `uv` for environment and dependency management
- `pytest` for tests
- `ruff` for lint/format
- type checking for maintained examples

## What This Repo Teaches

This repo is for AI-assisted application building with Python, not a pile of notebooks.

Every example should explain:

- what input data is used and whether it is safe to publish
- which model or external API is required
- how credentials are provided without committing secrets
- how much the run can cost
- how to evaluate whether the output is acceptable
- what failure modes should be retried, rejected, or escalated to a human

## Source Repositories

- `python-sandbox`
- `transcribe`
- `bedrock-sandbox`
- `ai`
- `langchain-gen`

## Learning Path

1. Python project setup
2. API and DB basics
3. audio transcription workflow
4. Bedrock and model API usage
5. agent workflows and tool calling
6. retrieval and file search boundaries
7. evals, cost notes, and safety checks

## Planned Structure

```text
apps/
  transcription-service/
lessons/
  python-api-basics/
  bedrock-basics/
  agent-tool-calling/
  retrieval-basics/
notebooks/
  README.md
evals/
  expected-behavior/
docs/
  2026-learning-items.md
  repository-profile.md
```

## Safety First

- Never commit API keys, credentials, or private datasets.
- Provide `.env.example` for every external service.
- Add cost notes to model/API examples.
- Keep notebook output small and reproducible.

## Study Loop

1. run the smallest local Python example first
2. add tests before connecting external model APIs
3. add `.env.example` and cost notes before committing an API sample
4. add an eval dataset for behavior that should not regress
5. keep notebooks as explanations, not as the only executable source

## What Belongs Elsewhere

- frontend clients belong in `learning-frontend-typescript`
- backend framework comparison belongs in `learning-backend-ddd`
- vector database experiments belong in `learning-data-stores` unless they are part of a retrieval lesson
- deployment, traces, and CI templates belong in `learning-platform-engineering`

## Repository Profile

See [docs/repository-profile.md](docs/repository-profile.md) for GitHub description, topics, public safety notes, and first milestones.
