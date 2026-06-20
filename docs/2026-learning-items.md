# 2026 Learning Items: AI Python

Last verified: 2026-06-20

## Must Learn

### Python project hygiene

- Python 3.14
- `uv`
- `pytest`
- `ruff`
- typing and data validation
- packaging basics

Projects:

- Move `python-sandbox` into `lessons/python-api-basics`.
- Add test and lint commands.

### Audio and transcription

- file upload boundaries
- batch transcription
- realtime transcription concepts
- chunking and retry strategy
- cost and privacy notes

Projects:

- Move `transcribe` into `apps/transcription-service`.
- Add `.env.example`.
- Add sample request/response docs.

### Model API and Bedrock

- credential handling
- request/response shape
- retries and timeouts
- structured output
- streaming
- cost notes

Projects:

- Move `bedrock-sandbox` into `lessons/bedrock-basics`.
- Keep notebooks under `notebooks/` with run order and expected outputs.

### Agentic workflows

- tool calling
- structured output
- guardrails
- evals
- traces and observability
- human review points

Projects:

- Rebuild `langchain-gen` as `lessons/agent-tool-calling`.
- Add `evals/` with a tiny expected-behavior dataset.

## Definition of Done

- Every lesson has a README.
- Every runnable sample has `Run`, `Test`, and `Last verified`.
- Every external API sample has `.env.example`.
- Every AI sample has cost and safety notes.

