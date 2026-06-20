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

### LLM internals (from scratch)

- byte-pair-encoding tokenizers (byte-level base, learn merges, lossless
  encode/decode, regex pre-tokenization, special tokens)
- statistical language models (n-gram counts, add-k/Laplace smoothing,
  interpolation/backoff, perplexity, seeded sampling with temperature/top-k)
- retrieval-augmented generation mechanics (chunking + overlap, embeddings,
  cosine retrieval, grounded prompt assembly, "I don't know" guardrail)
- LLM-as-judge evaluation (rubric design, token-F1/groundedness, pass/fail
  thresholds, deterministic-rubric stand-in vs. a real model judge)
- the bridge from statistical to neural LMs (next-token prediction objective)

Projects:

- `lessons/llm-internals` — pure-Python, stdlib-only hands-on covering all four
  (`demo.py` runs them; `test_llm_internals.py` is the `unittest` gate).
- Upgrade path: real tokenizers (`tiktoken`/Hugging Face), real embeddings + a
  vector store, the real model API via `projects/llm-tool-calling-lab`, and a
  from-scratch Transformer (nanoGPT) as the next step.

## Definition of Done

- Every lesson has a README.
- Every runnable sample has `Run`, `Test`, and `Last verified`.
- Every external API sample has `.env.example`.
- Every AI sample has cost and safety notes.

