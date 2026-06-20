# Further learning resources

Curated primary sources for this repo's named technologies: Anthropic-style
Model API + agentic workflows (tool calling, structured output, retries/timeouts,
streaming, evals), plus the Python/AWS Bedrock foundations the repo builds on.

Last verified: 2026-06-21

## Anthropic Model API & agentic workflows

- **Anthropic / Claude Developer Platform docs** — https://docs.anthropic.com
  The canonical reference for the Messages API, model IDs, and pricing. Start
  here for the request/response shape this repo's lab is modeled on.
- **Tool use (function calling) overview** — https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview
  How tools are defined (`name`/`description`/`input_schema`), how the model
  returns `tool_use` blocks, and how you return `tool_result`. This is the exact
  contract the lab's agent loop implements.
- **Structured outputs** — https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs
  Constraining responses to a JSON schema via `output_config.format` and strict
  tool use. The upgrade path replaces the lab's local validator with this.
- **Streaming** — https://docs.anthropic.com/en/docs/build-with-claude/streaming
  Server-sent event types (`message_start`, `content_block_delta`, …) and the
  SDK `text_stream` helper the lab's `stream_text` stands in for.
- **Handling stop reasons** — https://docs.anthropic.com/en/api/handling-stop-reasons
  `end_turn`, `tool_use`, `max_tokens`, `refusal`, `pause_turn` — the loop and
  error handling you must get right before going to production.
- **Errors & rate limits** — https://docs.anthropic.com/en/api/errors
  Which HTTP codes are retryable (429/5xx) and how the SDK's built-in
  exponential backoff works — the concept the lab's `call_with_retry` teaches.
- **Building effective agents (engineering blog)** — https://www.anthropic.com/research/building-effective-agents
  Anthropic's guidance on when to use a workflow vs. an agent, and how to keep
  tool surfaces simple. Read before scaling past this lab.
- **Anthropic Python SDK** — https://github.com/anthropics/anthropic-sdk-python
  The `anthropic` package the upgrade path installs. Source, examples, and the
  client config (timeouts, `max_retries`) referenced in the README.
- **Model Context Protocol (MCP)** — https://modelcontextprotocol.io
  The open standard for connecting models to external tools/data. Relevant to
  the sibling `mcp-stdio-tool-server` project and to scaling tool use.

## LLM internals (from scratch)

Primary sources for the `lessons/llm-internals` hands-on (BPE tokenizer,
statistical n-gram LM, toy RAG, LLM-as-judge):

- **"Neural Machine Translation of Rare Words with Subword Units" (Sennrich,
  Haddow & Birch, 2016)** — https://aclanthology.org/P16-1162/
  The paper that introduced byte-pair encoding to NLP. The merge-most-frequent-pair
  algorithm the lesson's `bpe.py` implements.
- **"Language Models are Unsupervised Multitask Learners" (GPT-2; Radford et
  al., 2019)** — https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
  Source of the **byte-level** BPE design (a 256-byte base alphabet so any
  Unicode encodes losslessly) the lesson uses.
- **"Attention Is All You Need" (Vaswani et al., 2017)** — https://arxiv.org/abs/1706.03762
  The original Transformer paper. The architecture the from-scratch upgrade path
  (nanoGPT) leads to, beyond the statistical n-gram model.
- **Karpathy — `minbpe`** — https://github.com/karpathy/minbpe
  Minimal, readable from-scratch BPE (train/encode/decode, regex split, special
  tokens). The direct reference for the tokenizer exercises.
- **Karpathy — `nanoGPT`** — https://github.com/karpathy/nanoGPT
  Minimal from-scratch GPT training/inference. The named next step after the
  n-gram model on the lesson's upgrade path.
- **Jurafsky & Martin, *Speech and Language Processing* — N-gram LM chapter** —
  https://web.stanford.edu/~jurafsky/slp3/
  The canonical text on n-gram language models, add-k/Laplace smoothing,
  interpolation/backoff, and perplexity (the lesson's Exercise 4).
- **Hugging Face — Tokenizers / NLP course** — https://huggingface.co/docs
  Production tokenizer library and the BPE/WordPiece/Unigram explainer; the
  upgrade target that replaces the hand-rolled tokenizer.
- **OpenAI — `tiktoken`** — https://github.com/openai/tiktoken
  The fast BPE tokenizer used by GPT models; the other real-tokenizer upgrade
  target.
- **scikit-learn — feature hashing (`HashingVectorizer`)** — https://scikit-learn.org/stable/modules/feature_extraction.html
  The "hashing trick" the lesson's toy embedding (`rag.embed`) is a tiny version
  of; the real-embedding upgrade path also starts here.

## Evaluation

- **OpenAI Evals (framework & methodology)** — https://github.com/openai/evals
  A widely used reference for structuring input→expected eval datasets and
  graders; the patterns transfer directly to the lab's `evals/` runner.
- **Anthropic — testing & evaluating prompts** — https://docs.anthropic.com/en/docs/test-and-evaluate/develop-tests
  How to build eval suites for LLM behavior, including LLM-as-judge graders
  (Exercise 5 in the lab).

## Safety

- **OWASP** — https://owasp.org
  Application security fundamentals. For LLM-specific risks (prompt injection,
  insecure tool/output handling) see the OWASP Top 10 for LLM Applications,
  linked from the OWASP project index.

## Python & foundations

- **Python 3 documentation** — https://docs.python.org/3/
  The standard library this repo restricts itself to (`json`, `dataclasses`,
  `typing`, etc.).
- **AWS Bedrock documentation** — https://docs.aws.amazon.com/bedrock/
  The managed model API the repo's Bedrock lessons target; Claude models are
  available through Bedrock with an `anthropic.`-prefixed model ID.
