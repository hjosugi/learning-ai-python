"""Run all four LLM-internals components end to end and print results.

    python3 lessons/llm-internals/demo.py

Everything is offline, stdlib only, and seeded, so the output is reproducible.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bpe import BPETokenizer  # noqa: E402
from corpus import RAG_DOCUMENTS, TRAINING_LINES, TRAINING_TEXT  # noqa: E402
from judge import JudgeCase, run_judge  # noqa: E402
from ngram_lm import NgramLM  # noqa: E402
from rag import RagIndex, assemble_prompt  # noqa: E402


def _rule(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def demo_bpe() -> None:
    _rule("1) Byte-pair-encoding tokenizer")
    tok = BPETokenizer()
    tok.train(TRAINING_TEXT, vocab_size=256 + 40)  # learn 40 merges
    sample = "a tokenizer encodes text into tokens."
    raw_byte_len = len(sample.encode("utf-8"))
    ids = tok.encode(sample)
    decoded = tok.decode(ids)
    print(f"learned merges        : {len(tok.merges)}")
    print(f"sample                : {sample!r}")
    print(f"raw byte length       : {raw_byte_len}")
    print(f"encoded token count   : {len(ids)}  "
          f"(compression {raw_byte_len / len(ids):.2f}x)")
    print(f"round-trip lossless   : {decoded == sample}")
    # Show a couple of the learned multi-byte tokens.
    learned = [tok.vocab[i] for i in sorted(tok.vocab) if i >= 256][:6]
    print(f"first learned tokens  : {[b.decode('utf-8', 'replace') for b in learned]}")


def demo_ngram() -> None:
    _rule("2) N-gram statistical language model (trigram, add-k)")
    lm = NgramLM(n=3, k=0.1)
    lm.train(TRAINING_LINES)
    in_domain = "a language model predicts the next token."
    random_text = "zxqv wkrp mjgb fhld nptz."
    lp_in = lm.log_prob(in_domain)
    lp_rand = lm.log_prob(random_text)
    print(f"vocab size            : {len(lm.vocab)}")
    print(f"in-domain text        : {in_domain!r}")
    print(f"  log-prob            : {lp_in:.2f}  perplexity {lm.perplexity(in_domain):.1f}")
    print(f"random text           : {random_text!r}")
    print(f"  log-prob            : {lp_rand:.2f}  perplexity {lm.perplexity(random_text):.1f}")
    print(f"in-domain > random    : {lp_in > lp_rand}")
    print(f"seeded sample (s=7)   : {lm.sample(seed=7, max_len=80)!r}")


def demo_rag() -> None:
    _rule("3) Toy RAG pipeline (chunk -> embed -> retrieve -> prompt)")
    index = RagIndex.build(RAG_DOCUMENTS, dim=256, chunk_size=30, overlap=8)
    query = "How does byte pair encoding decide what to merge?"
    retrieved = index.retrieve(query, top_k=2)
    print(f"total chunks          : {len(index.chunks)}")
    print(f"query                 : {query!r}")
    for rank, (score_, chunk) in enumerate(retrieved, start=1):
        print(f"  top-{rank} doc={chunk.doc_id!r} score={score_:.3f}")
    print("\n--- assembled grounded prompt ---")
    print(assemble_prompt(query, retrieved))


def demo_judge() -> None:
    _rule("4) LLM-as-judge eval harness (deterministic rubric)")
    good = JudgeCase(
        input="What does retrieval augmented generation do?",
        candidate=(
            "Retrieval augmented generation retrieves relevant chunks and "
            "grounds the model so it answers from the retrieved context."
        ),
        reference=(
            "RAG retrieves relevant document chunks and grounds the model's "
            "answer in that retrieved context."
        ),
        must_include=["retrieval", "context", "grounds"],
    )
    bad = JudgeCase(
        input=good.input,
        candidate="It is a kind of pizza topping popular in Italy.",
        reference=good.reference,
        must_include=good.must_include,
    )
    results, failed = run_judge([good, bad])
    for label, r in zip(("good", "bad "), results):
        print(f"{label} candidate -> score={r.score:.2f} passed={r.passed}")
        for reason in r.reasons:
            print(f"     - {reason}")
    print(f"\ngood scored higher than bad : {results[0].score > results[1].score}")
    print(f"cases failed                : {failed}/2")


def main() -> None:
    demo_bpe()
    demo_ngram()
    demo_rag()
    demo_judge()
    print("\nAll four components ran. See test_llm_internals.py for assertions.")


if __name__ == "__main__":
    main()
