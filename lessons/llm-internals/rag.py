"""A toy Retrieval-Augmented Generation (RAG) pipeline, stdlib only.

RAG grounds a language model in *your* documents instead of relying on what it
memorized. The pipeline is always the same four stages, and this file implements
each one by hand so the shape is concrete:

    chunk  -> embed -> retrieve (cosine top-k) -> assemble grounded prompt

* **Chunk.** Long documents are split into overlapping windows of words. Overlap
  keeps a fact from being cut in half across a chunk boundary.
* **Embed.** A real system uses a learned embedding model. Here we use a
  deterministic **hashed bag-of-words** ("hashing trick"): each word is hashed
  into one of ``dim`` buckets and its count is added there. No training, no
  model file, fully reproducible -- and it still captures lexical overlap, which
  is enough to retrieve the right chunk for a keyword query. The upgrade path
  swaps this one function for a real embedding API.
* **Retrieve.** Score every chunk against the query with **cosine similarity**
  (dot product of L2-normalized vectors) and keep the top-k.
* **Assemble.** Build a single grounded prompt string that puts the retrieved
  context first, then instructs the model to answer *only* from that context and
  to say it doesn't know otherwise. Grounding + an "I don't know" escape hatch is
  the core of reducing hallucination.

The "hashing trick" here is the same idea behind ``HashingVectorizer`` in
scikit-learn and feature hashing in classic ML.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric word tokens."""
    return _WORD_RE.findall(text.lower())


def chunk_text(text: str, *, chunk_size: int = 40, overlap: int = 10) -> list[str]:
    """Split text into overlapping word windows.

    ``chunk_size`` words per chunk, sliding by ``chunk_size - overlap`` each
    step so adjacent chunks share ``overlap`` words. Returns the chunk strings.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    words = text.split()
    if not words:
        return []
    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start:start + chunk_size]
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break  # last window already reached the end
    return chunks


def embed(text: str, *, dim: int = 256) -> list[float]:
    """Deterministic hashed bag-of-words embedding (the hashing trick).

    Each token is hashed to a bucket in ``[0, dim)`` via a stable SHA-1 digest
    (so it is identical across processes and Python's hash randomization is
    irrelevant). The vector is the bucket counts, L2-normalized so cosine
    similarity is just a dot product.
    """
    vec = [0.0] * dim
    for tok in tokenize(text):
        h = hashlib.sha1(tok.encode("utf-8")).digest()
        bucket = int.from_bytes(h[:4], "big") % dim
        vec[bucket] += 1.0
    return _l2_normalize(vec)


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec  # empty/all-zero vector; leave as-is to avoid div-by-zero
    return [x / norm for x in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors.

    For L2-normalized vectors this equals their dot product, in ``[-1, 1]`` (here
    ``[0, 1]`` because all components are non-negative counts).
    """
    return sum(x * y for x, y in zip(a, b))


@dataclass
class Chunk:
    """One retrievable unit: its source doc id, text, and embedding."""

    doc_id: str
    text: str
    vector: list[float]


@dataclass
class RagIndex:
    """An in-memory vector index over chunked documents."""

    chunks: list[Chunk]
    dim: int

    @classmethod
    def build(
        cls,
        documents: dict[str, str],
        *,
        dim: int = 256,
        chunk_size: int = 40,
        overlap: int = 10,
    ) -> "RagIndex":
        """Chunk every document and embed every chunk."""
        chunks: list[Chunk] = []
        for doc_id, text in documents.items():
            for piece in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
                chunks.append(Chunk(doc_id, piece, embed(piece, dim=dim)))
        return cls(chunks=chunks, dim=dim)

    def retrieve(self, query: str, *, top_k: int = 3) -> list[tuple[float, Chunk]]:
        """Return the ``top_k`` ``(score, chunk)`` pairs, highest score first."""
        q = embed(query, dim=self.dim)
        scored = [(cosine(q, c.vector), c) for c in self.chunks]
        # Sort by score desc; tie-break by doc_id then text for determinism.
        scored.sort(key=lambda sc: (-sc[0], sc[1].doc_id, sc[1].text))
        return scored[:top_k]


def assemble_prompt(query: str, retrieved: list[tuple[float, Chunk]]) -> str:
    """Build a grounded prompt from retrieved chunks and the user query.

    The structure -- numbered context blocks, then an instruction to answer only
    from that context and otherwise say "I don't know" -- is the load-bearing
    part of a RAG prompt. It is what turns retrieval into *grounded* generation.
    """
    context_blocks = []
    for i, (_score, chunk) in enumerate(retrieved, start=1):
        context_blocks.append(f"[{i}] (source: {chunk.doc_id}) {chunk.text}")
    context = "\n".join(context_blocks) if context_blocks else "(no context found)"
    return (
        "You are a helpful assistant. Answer the question using ONLY the context "
        "below. If the answer is not in the context, say you don't know.\n\n"
        "Context:\n"
        f"{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )
