# LLM internals from scratch (pure Python)

A rigorous, **stdlib-only** hands-on that builds the four pieces every modern
LLM stack is made of, from scratch, so you understand what the libraries do
before you reach for them:

1. **Byte-pair-encoding (BPE) tokenizer** — `bpe.py`. Trains merges on a bundled
   corpus; lossless byte-level encode/decode.
2. **Tiny statistical language model** — `ngram_lm.py`. A character trigram with
   add-k smoothing that can **score** (log-prob, perplexity) and **sample**
   (seeded) text.
3. **Toy RAG pipeline** — `rag.py`. chunk → hashed-bag embed → cosine retrieve
   top-k → assemble a grounded prompt.
4. **LLM-as-judge eval harness** — `judge.py`. A deterministic rubric scorer
   that returns pass/fail + a score over a tiny dataset.

Python 3.14, standard library only. No `pip install`, no network, fully seeded.

Last verified: 2026-06-21

## Run

Runs all four components and prints results:

```bash
python3 lessons/llm-internals/demo.py
```

## Test

Non-interactive `unittest` suite (33 cases). Exits non-zero on any failure, so
it works as a CI gate without pytest:

```bash
python3 lessons/llm-internals/test_llm_internals.py
```

## Files

| File | Role |
|------|------|
| `bpe.py` | Byte-level BPE: `train` (learn merges), `encode`/`decode` (lossless) |
| `ngram_lm.py` | Char n-gram LM: `train`, `prob`/`log_prob`/`perplexity`, seeded `sample` |
| `rag.py` | `chunk_text`, `embed` (hashing trick), `RagIndex.retrieve`, `assemble_prompt` |
| `judge.py` | `token_f1`, rubric `score`, `run_judge`, bundled dataset |
| `corpus.py` | Small bundled in-domain corpus + RAG documents (no downloads) |
| `demo.py` | Runs all four and prints results |
| `test_llm_internals.py` | `unittest` test suite |

## Walkthrough

### 1. BPE tokenizer (`bpe.py`)

A tokenizer turns text into integer ids a model can embed. BPE is the dominant
family (GPT-2/3/4 via tiktoken, many Hugging Face models). The algorithm:

- **Start from bytes.** Every byte 0–255 is already a token, so the vocabulary
  is complete from the start and the tokenizer is **lossless for any input** —
  there is never an `<unk>`. This is the GPT-2 "byte-level BPE" trick, and it is
  why `test_roundtrip_lossless_on_unicode` passes on emoji and CJK.
- **Learn merges.** Repeatedly count adjacent token pairs (`get_pair_counts`),
  take the most frequent (`max` with a deterministic tie-break), and `merge` it
  into a new id 256, 257, … Record the merge. The **ordered merge list is the
  model**.
- **Encode** by replaying merges in learned order: at each step apply the
  present pair with the **lowest learned id** (= learned earliest). That faithful
  replay is what makes encode∘decode the identity.
- **Decode** by concatenating each id's byte string and UTF-8 decoding.

Signature observation the tests pin down: merges **reduce token count** vs. raw
bytes (`test_merges_reduce_token_count`), and more merges compress at least as
well (`test_more_merges_compress_at_least_as_well`). The demo prints a ~2.5×
compression on the sample sentence.

### 2. N-gram language model (`ngram_lm.py`)

The classical, pre-neural language model. It is the cleanest place to see the
two ideas that survive into Transformers:

- **A language model is a conditional distribution over the next token** given a
  context. We estimate `P(token | context)` by counting n-grams. Everything else
  (`log_prob`, `perplexity`, `sample`) is derived from that one object.
- **No zero probabilities (smoothing).** Maximum likelihood assigns 0 to any
  unseen n-gram, sending whole-sentence probability to 0 and log-prob to −∞.
  **Add-k smoothing** adds a pseudo-count `k` to every possible continuation:
  `P = (count + k) / (total + k·|V|)`. `test_probabilities_form_a_distribution`
  confirms the smoothed probs sum to 1; `test_no_zero_probability_for_unseen_continuation`
  confirms unseen events get positive mass. Neural LMs get this for free from
  the softmax.

The model **scores** (in-domain text gets higher log-prob / lower perplexity
than random — `test_in_domain_scores_higher_than_random`) and **samples**
token-by-token via inverse-CDF on a seeded `random.Random` (reproducible —
`test_sampling_is_seeded_and_reproducible`). Sequences are padded with `BOS`
markers and terminated by an `EOS` token the model learns to emit.

### 3. Toy RAG pipeline (`rag.py`)

RAG grounds a model in *your* documents. The pipeline is always the same four
stages:

- **Chunk** — overlapping word windows (`chunk_text`); overlap keeps a fact from
  being split across a boundary.
- **Embed** — a deterministic **hashed bag-of-words** (the "hashing trick", as in
  scikit-learn's `HashingVectorizer`): hash each word into one of `dim` buckets
  via a stable SHA-1 digest, count, then L2-normalize. No model file, fully
  reproducible across processes (unaffected by Python's hash randomization).
- **Retrieve** — score chunks against the query with **cosine similarity** (a dot
  product of normalized vectors) and keep top-k.
- **Assemble** — build a grounded prompt: numbered context blocks first, then an
  instruction to answer *only* from that context and otherwise say "I don't
  know". That guardrail is the core of reducing hallucination
  (`test_assembled_prompt_contains_context_and_guardrail`).

Honest limitation the tests reveal: the hashing trick weights all words equally,
so short queries full of stopwords ("what is …") are weak. The fix is on the
upgrade path (real embeddings, or IDF/stopword weighting).

### 4. LLM-as-judge eval harness (`judge.py`)

When output is open-ended, there is no string to diff against, so exact/substring
evals break. The industry answer is **LLM-as-judge**: a second model scores the
candidate against a rubric (and optional reference) and returns a score +
verdict. The **interface** is the lesson:

```
score(case) -> JudgeResult(score in [0,1], passed: bool, reasons: [...])
```

Here the judge body is a **deterministic rubric** (token-F1 vs. reference +
required-keyword coverage + groundedness + a length gate) so the harness is
reproducible, free, and testable. `test_good_scores_higher_than_bad` and
`test_good_passes_bad_fails` pin the behavior the spec asks for. To make it a
*real* LLM judge you replace **only** `_rubric_score` (see Upgrade path); the
dataset, runner, and pass/fail gate stay identical.

## Upgrade path

This lesson is the from-scratch floor. Grow each piece toward real tooling:

- **Tokenizer → real BPE.** Swap `BPETokenizer` for OpenAI's
  [`tiktoken`](https://github.com/openai/tiktoken) or a Hugging Face
  `tokenizers` BPE. Add the missing production pieces: a regex pre-tokenization
  split (the GPT-2 pattern) and special tokens (`<|endoftext|>`). Karpathy's
  [`minbpe`](https://github.com/karpathy/minbpe) is the reference for exactly
  these next steps.
- **Embeddings → real embeddings.** Replace `embed`'s hashing trick with a
  learned embedding model (a sentence-transformer locally, or the embeddings
  endpoint of a model API) and put the vectors in a real vector store
  (`learning-data-stores`). Add IDF/stopword weighting to fix weak stopword
  queries.
- **Model → a real API.** Plug the RAG prompt and the judge into the sibling
  [`projects/llm-tool-calling-lab`](../../projects/llm-tool-calling-lab): its
  `Transport` seam lets you call the real Anthropic Messages API. Replace the
  rubric judge's `_rubric_score` with a `messages.create` call that grades
  against the same rubric and returns JSON.
- **Statistical LM → a neural one.** The n-gram model is the conceptual bridge
  to a from-scratch Transformer. The natural next step is Karpathy's
  [`nanoGPT`](https://github.com/karpathy/nanoGPT) (and the `makemore`/`minGPT`
  lineage): same next-token-prediction objective, but a neural network and a
  softmax instead of counts and add-k.

## Exercises

Progressive — each builds on the last:

1. **Regex pre-tokenization for BPE.** Add a GPT-2-style regex split so merges
   never cross word/whitespace boundaries, train per-piece, and confirm
   round-trips stay lossless. Compare token counts against the current
   whitespace-agnostic version.
2. **Special tokens.** Reserve an id for `<|endoftext|>`, make `encode` keep it
   atomic (never merged or split), and have `decode` render it. Add a test that
   it survives a round-trip embedded in normal text.
3. **Top-k / temperature sampling for the n-gram LM.** Add `temperature` (divide
   log-probs) and `top_k` truncation to `sample`, and show that low temperature
   makes the seeded output more repetitive and high temperature more diverse.
4. **Interpolated / backoff smoothing.** Replace add-k with linear interpolation
   between trigram, bigram, and unigram estimates (or Katz backoff) and show it
   lowers perplexity on held-out in-domain text vs. plain add-k.
5. **Better RAG scoring.** Add IDF weighting (down-weight stopwords) to `embed`
   or switch retrieval to BM25, then show the short stopword query
   (`"what is perplexity and smoothing"`) now retrieves the `ngram` doc.
6. **Real LLM-as-judge.** Implement an `AnthropicJudge` that replaces
   `_rubric_score` with a `messages.create` call returning
   `{"score", "passed", "reasons"}` validated against a JSON schema, run it over
   the same `default_dataset()`, and compare its verdicts to the rubric's.
   (Costs tokens — see the tool-calling lab's cost/safety notes.)
