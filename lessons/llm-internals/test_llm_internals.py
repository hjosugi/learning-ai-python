"""Test suite for the LLM-internals lesson (python3 unittest, no deps).

    python3 lessons/llm-internals/test_llm_internals.py

Runs the unittest runner in ``__main__`` and exits non-zero on any failure, so
it works as a CI gate without pytest.
"""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bpe import BPETokenizer, get_pair_counts, merge  # noqa: E402
from corpus import RAG_DOCUMENTS, TRAINING_LINES, TRAINING_TEXT  # noqa: E402
from judge import JudgeCase, run_judge, score, token_f1  # noqa: E402
from ngram_lm import BOS, EOS, NgramLM  # noqa: E402
from rag import RagIndex, assemble_prompt, chunk_text, cosine, embed  # noqa: E402


class TestBPEPrimitives(unittest.TestCase):
    def test_pair_counts(self) -> None:
        self.assertEqual(
            dict(get_pair_counts([1, 2, 3, 1, 2])),
            {(1, 2): 2, (2, 3): 1, (3, 1): 1},
        )

    def test_merge_replaces_all_nonoverlapping(self) -> None:
        self.assertEqual(merge([1, 2, 3, 1, 2], (1, 2), 99), [99, 3, 99])

    def test_merge_handles_adjacent_repeats(self) -> None:
        # (1,1) should merge the first two, then leave the third 1.
        self.assertEqual(merge([1, 1, 1], (1, 1), 7), [7, 1])


class TestBPETokenizer(unittest.TestCase):
    def setUp(self) -> None:
        self.tok = BPETokenizer()
        self.tok.train(TRAINING_TEXT, vocab_size=256 + 50)

    def test_learns_requested_number_of_merges(self) -> None:
        # The corpus is large enough that all 50 merges are learnable.
        self.assertEqual(len(self.tok.merges), 50)
        # Every learned id maps to >= 2 bytes.
        for pair, new_id in self.tok.merges.items():
            self.assertGreaterEqual(len(self.tok.vocab[new_id]), 2)

    def test_roundtrip_is_lossless_on_training_text(self) -> None:
        ids = self.tok.encode(TRAINING_TEXT)
        self.assertEqual(self.tok.decode(ids), TRAINING_TEXT)

    def test_roundtrip_lossless_on_unseen_text(self) -> None:
        # Includes punctuation and a word the corpus never saw.
        for s in ["completely unseen phrase!", "tokens?", "", "a"]:
            with self.subTest(s=s):
                self.assertEqual(self.tok.decode(self.tok.encode(s)), s)

    def test_roundtrip_lossless_on_unicode(self) -> None:
        # Byte-level BPE must handle arbitrary Unicode losslessly.
        for s in ["café naïve", "こんにちは", "emoji \U0001f600 ok"]:
            with self.subTest(s=s):
                self.assertEqual(self.tok.decode(self.tok.encode(s)), s)

    def test_merges_reduce_token_count(self) -> None:
        # Encoding with merges must produce fewer tokens than the raw bytes.
        sample = "a tokenizer encodes text into tokens and decodes tokens."
        raw_bytes = list(sample.encode("utf-8"))
        encoded = self.tok.encode(sample)
        self.assertLess(len(encoded), len(raw_bytes))

    def test_more_merges_compress_at_least_as_well(self) -> None:
        sample = TRAINING_TEXT
        few = BPETokenizer()
        few.train(sample, vocab_size=256 + 10)
        many = BPETokenizer()
        many.train(sample, vocab_size=256 + 60)
        self.assertLessEqual(len(many.encode(sample)), len(few.encode(sample)))

    def test_untrained_tokenizer_is_identity_over_bytes(self) -> None:
        tok = BPETokenizer()  # no training -> no merges
        ids = tok.encode("hi!")
        self.assertEqual(ids, list("hi!".encode("utf-8")))
        self.assertEqual(tok.decode(ids), "hi!")

    def test_train_rejects_tiny_vocab(self) -> None:
        with self.assertRaises(ValueError):
            BPETokenizer().train("x", vocab_size=10)


class TestNgramLM(unittest.TestCase):
    def setUp(self) -> None:
        self.lm = NgramLM(n=3, k=0.1)
        self.lm.train(TRAINING_LINES)

    def test_in_domain_scores_higher_than_random(self) -> None:
        in_domain = "a language model predicts the next token."
        random_text = "zxqv wkrp mjgb fhld nptz."
        self.assertGreater(self.lm.log_prob(in_domain), self.lm.log_prob(random_text))

    def test_in_domain_lower_perplexity_than_random(self) -> None:
        in_domain = "tokenization splits text into tokens."
        random_text = "qqq zzz xxx www."
        self.assertLess(
            self.lm.perplexity(in_domain), self.lm.perplexity(random_text)
        )

    def test_probabilities_form_a_distribution(self) -> None:
        # For any context, the smoothed probs over the full vocab sum to ~1.
        ctx = ("t", "o")
        total = sum(self.lm.prob(ctx, t) for t in self.lm.vocab)
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_no_zero_probability_for_unseen_continuation(self) -> None:
        # Add-k guarantees strictly positive mass even for unseen pairs.
        p = self.lm.prob(("z", "z"), "q")
        self.assertGreater(p, 0.0)

    def test_sampling_is_seeded_and_reproducible(self) -> None:
        a = self.lm.sample(seed=42, max_len=60)
        b = self.lm.sample(seed=42, max_len=60)
        self.assertEqual(a, b)

    def test_different_seeds_can_differ(self) -> None:
        a = self.lm.sample(seed=1, max_len=60)
        b = self.lm.sample(seed=2, max_len=60)
        # Not guaranteed in theory, but with this corpus they differ.
        self.assertNotEqual(a, b)

    def test_sample_never_emits_control_markers(self) -> None:
        s = self.lm.sample(seed=3, max_len=120)
        self.assertNotIn(BOS, s)
        self.assertNotIn(EOS, s)

    def test_constructor_validates_params(self) -> None:
        with self.assertRaises(ValueError):
            NgramLM(n=1)
        with self.assertRaises(ValueError):
            NgramLM(n=2, k=0.0)


class TestRag(unittest.TestCase):
    def setUp(self) -> None:
        self.index = RagIndex.build(
            RAG_DOCUMENTS, dim=256, chunk_size=30, overlap=8
        )

    def test_chunking_overlaps_and_covers(self) -> None:
        words = " ".join(str(i) for i in range(100))
        chunks = chunk_text(words, chunk_size=40, overlap=10)
        self.assertGreater(len(chunks), 1)
        # First two chunks share the overlap region.
        first = chunks[0].split()
        second = chunks[1].split()
        self.assertEqual(first[-10:], second[:10])

    def test_embedding_is_normalized_and_deterministic(self) -> None:
        v1 = embed("byte pair encoding", dim=128)
        v2 = embed("byte pair encoding", dim=128)
        self.assertEqual(v1, v2)
        norm = sum(x * x for x in v1) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=6)

    def test_cosine_self_similarity_is_one(self) -> None:
        v = embed("retrieval augmented generation", dim=128)
        self.assertAlmostEqual(cosine(v, v), 1.0, places=6)

    def test_retrieves_relevant_chunk_for_bpe_query(self) -> None:
        results = self.index.retrieve("how does byte pair encoding merge tokens", top_k=1)
        self.assertEqual(results[0][1].doc_id, "bpe")

    def test_retrieves_relevant_chunk_for_ngram_query(self) -> None:
        # A keyword-rich query reliably retrieves the n-gram doc. (Short queries
        # full of stopwords are weak with the hashing trick -- see the README's
        # upgrade path: real embeddings / IDF weighting fix that.)
        results = self.index.retrieve(
            "n-gram model estimates probability of the next token from counts",
            top_k=1,
        )
        self.assertEqual(results[0][1].doc_id, "ngram")

    def test_retrieves_relevant_chunk_for_eval_query(self) -> None:
        results = self.index.retrieve(
            "llm as judge scores an answer against a rubric pass or fail", top_k=1
        )
        self.assertEqual(results[0][1].doc_id, "eval")

    def test_retrieves_relevant_chunk_for_rag_query(self) -> None:
        results = self.index.retrieve("cosine similarity grounded prompt context", top_k=1)
        self.assertEqual(results[0][1].doc_id, "rag")

    def test_assembled_prompt_contains_context_and_guardrail(self) -> None:
        query = "what does byte pair encoding do"
        retrieved = self.index.retrieve(query, top_k=2)
        prompt = assemble_prompt(query, retrieved)
        self.assertIn("Context:", prompt)
        self.assertIn(query, prompt)
        self.assertIn("don't know", prompt)  # the grounding guardrail
        self.assertIn("source:", prompt)


class TestJudge(unittest.TestCase):
    def _good(self) -> JudgeCase:
        return JudgeCase(
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

    def _bad(self) -> JudgeCase:
        g = self._good()
        return JudgeCase(
            input=g.input,
            candidate="It is a kind of pizza topping popular in Italy.",
            reference=g.reference,
            must_include=g.must_include,
        )

    def test_good_scores_higher_than_bad(self) -> None:
        good = score(self._good())
        bad = score(self._bad())
        self.assertGreater(good.score, bad.score)

    def test_good_passes_bad_fails(self) -> None:
        self.assertTrue(score(self._good()).passed)
        self.assertFalse(score(self._bad()).passed)

    def test_empty_candidate_fails_with_zero(self) -> None:
        g = self._good()
        empty = JudgeCase(input=g.input, candidate="", reference=g.reference,
                          must_include=g.must_include)
        r = score(empty)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.passed)

    def test_score_is_bounded(self) -> None:
        for case in (self._good(), self._bad()):
            r = score(case)
            self.assertGreaterEqual(r.score, 0.0)
            self.assertLessEqual(r.score, 1.0)

    def test_token_f1_bounds(self) -> None:
        self.assertEqual(token_f1("same words here", "same words here"), 1.0)
        self.assertEqual(token_f1("totally", "different"), 0.0)
        self.assertEqual(token_f1("", "x"), 0.0)

    def test_run_judge_counts_failures(self) -> None:
        results, failed = run_judge([self._good(), self._bad()])
        self.assertEqual(len(results), 2)
        self.assertEqual(failed, 1)  # only the bad one fails


if __name__ == "__main__":
    unittest.main(verbosity=2)
