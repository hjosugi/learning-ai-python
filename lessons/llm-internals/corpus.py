"""A small bundled, in-domain corpus shared by the demo and tests.

Keeping the corpus in-repo (no downloads) is what makes this lesson offline and
reproducible. The text is intentionally about LLM/NLP internals so that:

* the BPE tokenizer learns real, repeated subword pieces ("token", "ing", ...),
* the n-gram LM assigns clearly higher probability to in-domain than to random
  text, and
* the RAG index has documents with distinct, retrievable facts.
"""

from __future__ import annotations

# One topical paragraph; BPE trains on this and the n-gram LM trains on its
# sentences. Repetition of subword pieces is deliberate -- it is what BPE
# compresses and what the n-gram model learns to predict.
TRAINING_TEXT = (
    "a language model predicts the next token given the previous tokens. "
    "tokenization splits text into tokens. byte pair encoding learns to merge "
    "the most frequent pair of tokens into a new token. a tokenizer encodes "
    "text into tokens and decodes tokens back into text. an n-gram model "
    "estimates the probability of the next token from the previous tokens. "
    "smoothing gives unseen tokens a small probability. retrieval finds the "
    "most relevant chunks of text for a query. a grounded prompt puts the "
    "retrieved context before the question so the model answers from the text."
)

# The n-gram LM trains line-by-line; one sentence per line.
TRAINING_LINES = [s.strip() + "." for s in TRAINING_TEXT.split(".") if s.strip()]


# Documents for the RAG index. Each carries one clearly retrievable fact so a
# keyword query can be matched to exactly one document.
RAG_DOCUMENTS = {
    "bpe": (
        "Byte pair encoding is a tokenizer training algorithm. It starts from "
        "raw bytes and repeatedly merges the most frequent adjacent pair of "
        "tokens into a single new token. The learned list of merges is the "
        "tokenizer. GPT models use a byte level variant of byte pair encoding "
        "so that any Unicode input can be encoded without an unknown token."
    ),
    "ngram": (
        "An n-gram language model estimates the probability of the next token "
        "from the previous n minus one tokens by counting n-grams in a corpus. "
        "Add-k smoothing adds a pseudo count to every continuation so that "
        "unseen n-grams receive a small nonzero probability instead of zero. "
        "Perplexity measures how surprised the model is by held out text."
    ),
    "rag": (
        "Retrieval augmented generation grounds a language model in your own "
        "documents. The pipeline chunks documents, embeds each chunk into a "
        "vector, retrieves the chunks most similar to the query using cosine "
        "similarity, and assembles a grounded prompt that instructs the model "
        "to answer only from the retrieved context."
    ),
    "eval": (
        "Evaluation with an LLM as judge uses a second model to score an "
        "open ended answer against a rubric and an optional reference answer. "
        "The judge returns a score and a pass or fail verdict. A deterministic "
        "rubric scorer can stand in for the judge during development so the "
        "eval harness stays reproducible and free to run."
    ),
}
