# LLM Tool-Calling Lab

An Anthropic Messages-API-**shaped** tool-calling loop that runs with **no
credentials and no network**. It teaches the real agentic loop — tool calling,
structured output, retries/timeouts, streaming, and evals — against a
deterministic `FakeTransport`, then shows exactly how to swap in the real
`anthropic` client without touching the agent loop.

Python 3.14, standard library only. No `pip install`.

Last verified: 2026-06-21

## What it teaches

This mirrors the named learning target for this repo: *Anthropic-style Model API
+ agentic workflows*. Concretely:

- **Tools as JSON-schema dicts** — `tools.py` defines tools in the exact
  `{"name", "description", "input_schema"}` shape the real `tools=[...]` field
  takes.
- **The tool-calling loop** — `agent.py` runs the real four-step pattern: call →
  append assistant turn verbatim → execute `tool_use` blocks → append one user
  message of `tool_result` blocks → loop until `stop_reason == "end_turn"`.
- **Messages-API response shape** — `FakeTransport` returns `role`/`content`
  blocks of type `text`/`tool_use` with `stop_reason` of `tool_use` / `end_turn`,
  matching `POST /v1/messages`.
- **Retries with exponential backoff** — `call_with_retry` survives a simulated
  transient `529 overloaded_error`.
- **Timeout concept** — `with_timeout` enforces a wall-clock deadline per call.
- **Streaming** — `stream_text` yields the answer in chunks like `text_stream`.
- **Structured output validation** — the final object is validated against a
  JSON schema (the same subset the API's `output_config.format` supports).
- **Evals** — `evals/` has a tiny input→expected dataset and a pass/fail runner.

## Files

| File | Role |
|------|------|
| `tools.py` | Tool JSON schemas + local implementations + dispatch |
| `transport.py` | `Transport` interface (the seam), `FakeTransport`, retry, timeout |
| `agent.py` | The tool-calling loop, streaming, structured-output build/validate |
| `schema.py` | Tiny stdlib JSON-schema validator |
| `evals/dataset.json` | Input → expected-behavior cases |
| `evals/run_evals.py` | Eval runner (exits non-zero on failure) |
| `test_agent.py` | Non-interactive test suite |
| `.env.example` | Placeholder for `ANTHROPIC_API_KEY` (used only on the upgrade path) |

## Run

```bash
python3 projects/llm-tool-calling-lab/agent.py
```

Run the evals:

```bash
python3 projects/llm-tool-calling-lab/evals/run_evals.py
```

## Test

```bash
python3 projects/llm-tool-calling-lab/test_agent.py
```

Exits non-zero on any failure (no pytest needed). The suite includes the eval
runner, so green tests mean green evals too.

## The seam

The agent loop only ever calls `transport.create(model=..., max_tokens=...,
system=..., tools=..., messages=...)` and reads `response["content"]` /
`response["stop_reason"]`. It never imports `anthropic` and never branches on the
transport type. `transport.Transport` is that interface; `FakeTransport`
implements it offline. That single boundary is what makes the upgrade below a
drop-in.

## Upgrade path

Swap `FakeTransport` for the real Anthropic client **without changing
`agent.py`**. The loop already speaks the Messages API contract.

1. Install the SDK (this is the step the rest of the repo defers):

   ```bash
   pip install anthropic        # or: uv add anthropic
   ```

2. Provide credentials (never commit them):

   ```bash
   cp projects/llm-tool-calling-lab/.env.example \
      projects/llm-tool-calling-lab/.env
   # edit .env and set ANTHROPIC_API_KEY, then: export $(cat .env | xargs)
   ```

3. Write a thin adapter that satisfies the same `Transport` interface. It
   translates the loop's call into `client.messages.create(...)` and returns the
   response as a dict in the same shape `FakeTransport` returns:

   ```python
   # real_transport.py  (new file — agent.py is untouched)
   import anthropic

   class AnthropicTransport:
       """Implements transport.Transport against the real Messages API."""
       def __init__(self, model: str) -> None:
           # Reads ANTHROPIC_API_KEY from the environment.
           self._client = anthropic.Anthropic()
           self._model = model

       def create(self, *, model, max_tokens, system, tools, messages):
           resp = self._client.messages.create(
               model=self._model,        # choose a current model from the official docs
               max_tokens=max_tokens,
               system=system,
               tools=tools,              # the SAME tools.TOOLS dicts
               messages=messages,        # the SAME message/blocks shape
           )
           # Normalize the SDK object to the dict shape the loop expects.
           return {
               "id": resp.id,
               "role": resp.role,
               "model": resp.model,
               "stop_reason": resp.stop_reason,
               "content": [b.model_dump() for b in resp.content],
               "usage": {
                   "input_tokens": resp.usage.input_tokens,
                   "output_tokens": resp.usage.output_tokens,
               },
           }
   ```

   Then run the unchanged loop against it:

   ```python
   import os
   from agent import run_agent
   from real_transport import AnthropicTransport
   run = run_agent("What is the weather in Tokyo?", AnthropicTransport(os.environ["ANTHROPIC_MODEL"]))
   ```

   `agent.py`, `tools.py`, `schema.py`, the eval dataset, and the assertions all
   stay the same. The real model now decides which tools to call; your tools and
   evals are the contract that keeps it honest.

4. (Optional, recommended) Once on the real client, let the API enforce the
   schema and retries instead of the hand-rolled versions:
   - structured output: pass `output_config={"format": {"type": "json_schema",
     "schema": FINAL_OUTPUT_SCHEMA}}` to `messages.create`.
   - retries: the SDK already retries 429/5xx with backoff (`max_retries`,
     default 2); the local `call_with_retry` becomes redundant.
   - streaming: use `with client.messages.stream(...) as s: for t in
     s.text_stream` in place of `stream_text`.

## Cost & safety notes

Part of this repo's Definition of Done — every AI sample documents cost and
safety.

**Cost**
- This lab is **free**: it makes zero API calls. FakeTransport runs locally.
- After the upgrade, you pay per token. Verify the current model list and
  pricing in the provider's official docs before running live evals. A
  tool-calling loop pays for **every** turn: the request grows each turn because
  you resend the full history, so an N-tool task is N+1 billed requests over a
  growing prompt.
- Control spend: cap `max_turns`, keep `max_tokens` tight, prefer a smaller
  current model for simple routing, and use prompt caching for the stable system
  + tool definitions where the provider supports it.

**Safety**
- **Never commit credentials.** `.env` is gitignored repo-wide; only
  `.env.example` (a placeholder) is committed.
- **Tools run YOUR code from model-chosen inputs.** The model only emits a tool
  name + input; your harness executes it. Validate every tool input against its
  schema, and gate side-effecting / irreversible tools (sending email, deleting
  data, spending money) behind explicit confirmation. The lab's tools are pure
  and read-only on purpose.
- **Tool inputs and outputs are untrusted.** Treat `tool_use.input` like user
  input; never `eval` it or pass it to a shell unsanitized.
- **Validate the final structured output** before acting on it (this lab does),
  and handle the real API's `stop_reason == "refusal"` and `max_tokens` cases
  before reading `content`.

## Exercises

1. **Add a parallel-safe tool.** Add a `currency_convert` tool to `tools.py`
   (schema + impl), make `FakeTransport` emit two `tool_use` blocks in one
   assistant turn (weather + convert), and confirm the loop returns both
   `tool_result` blocks in a single user message. Add an eval case.
2. **Tighten the output schema.** Add a `"confidence"` number field to
   `FINAL_OUTPUT_SCHEMA`, populate it in `structured_result`, and add a test
   that a missing/out-of-type confidence is rejected by `schema.validate`.
3. **Make retries observable.** Extend `call_with_retry` to record each
   attempt's delay into a list, assert the delays grow exponentially in a test,
   and print a one-line retry summary at the end of a run.
4. **Wire the real client behind the seam.** Implement `real_transport.py` from
   the upgrade path, run the *unchanged* eval dataset against it, and note which
   cases pass/fail with a live model versus FakeTransport. (Costs tokens.)
5. **Add an LLM-as-judge eval.** Add a case whose answer is open-ended and score
   it with a rubric function (offline) instead of a substring match — then,
   after the upgrade, replace the rubric function with a second `messages.create`
   call that grades the answer.
