"""Byte-pair-encoding (BPE) tokenizer, from scratch, stdlib only.

BPE is the tokenizer family used by GPT-2/3/4 (tiktoken) and many Hugging Face
models. The idea is simple and worth implementing once by hand:

1. Start from the *bytes* of the text. Every byte (0..255) is already a token,
   so the vocabulary is complete from the start and the tokenizer is **lossless**
   for any input (no <unk>, ever).
2. Repeatedly find the most frequent adjacent pair of tokens in the training
   corpus and "merge" it into a single new token. Record the merge.
3. Stop after a target number of merges. The learned merge list *is* the model.

Encoding new text replays the learned merges in the order they were learned;
decoding maps token ids back to byte sequences and decodes UTF-8.

Why bytes and not characters? Working on raw UTF-8 bytes means the tokenizer
handles *any* Unicode (emoji, CJK, accents) with a fixed 256-symbol base
alphabet and never fails to encode. This mirrors the GPT-2 "byte-level BPE"
design and Karpathy's `minbpe`.

This implementation is deliberately the plain/educational variant (no regex
pre-tokenization split, no special tokens). Those are noted as exercises.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

# A token id is an int. Base ids 0..255 are the raw byte values. Merged tokens
# get ids 256, 257, ... in the order they are learned.
Pair = tuple[int, int]


def get_pair_counts(ids: list[int], counts: Counter | None = None) -> Counter:
    """Count occurrences of each adjacent pair in a token-id sequence.

    >>> dict(get_pair_counts([1, 2, 3, 1, 2]))
    {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    counts = Counter() if counts is None else counts
    for pair in zip(ids, ids[1:]):  # zip(seq, seq[1:]) -> consecutive pairs
        counts[pair] += 1
    return counts


def merge(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` with ``new_id``.

    >>> merge([1, 2, 3, 1, 2], (1, 2), 99)
    [99, 3, 99]
    """
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        # Match the pair only if there is a next element and both match.
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


@dataclass
class BPETokenizer:
    """A trained (or trainable) byte-level BPE tokenizer.

    Attributes:
        merges: ordered mapping ``pair -> new_id``. Order is the training order
            and is the order encoding must replay them in.
        vocab: ``id -> bytes`` for decoding. Built from ``merges`` plus the 256
            base byte tokens.
    """

    merges: dict[Pair, int] = field(default_factory=dict)
    vocab: dict[int, bytes] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.vocab:
            self.vocab = {i: bytes([i]) for i in range(256)}

    # ---- training -----------------------------------------------------------

    def train(self, text: str, vocab_size: int, *, verbose: bool = False) -> None:
        """Learn merges from ``text`` until the vocab reaches ``vocab_size``.

        ``vocab_size`` must be >= 256 (the byte base). The number of merges
        learned is ``vocab_size - 256``.
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 (the byte base alphabet)")
        num_merges = vocab_size - 256

        ids = list(text.encode("utf-8"))  # start from raw bytes
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        for k in range(num_merges):
            counts = get_pair_counts(ids)
            if not counts:
                break  # nothing left to merge (sequence shorter than 2)
            # Pick the most frequent pair. Tie-break on the pair itself so the
            # result is fully deterministic regardless of dict iteration order.
            top = max(counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            best_pair, freq = top
            if freq < 2:
                break  # no pair repeats; further merges would not compress
            new_id = 256 + k
            ids = merge(ids, best_pair, new_id)
            self.merges[best_pair] = new_id
            # The new token's bytes are the concatenation of its two parts'.
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            if verbose:
                a = self.vocab[best_pair[0]]
                b = self.vocab[best_pair[1]]
                print(f"merge {k + 1}/{num_merges}: {best_pair} -> {new_id} "
                      f"({a!r}+{b!r}) freq={freq}")

    # ---- inference ----------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids by replaying learned merges greedily.

        At each step we find, among the pairs present in the current sequence,
        the one that was learned *earliest* (lowest new_id) and apply it. This
        reproduces the training merge order, which is what makes encode/decode a
        faithful round-trip.
        """
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            counts = get_pair_counts(ids)
            # Among present pairs that we know how to merge, pick the one with
            # the smallest assigned id == learned earliest. min() over a pair
            # not in self.merges yields +inf so it is never chosen.
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # no more learned merges apply
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to text. The inverse of :meth:`encode`."""
        # Concatenate each id's byte string, then decode UTF-8. ``errors``
        # guards against decoding a *partial* multibyte char if someone hands us
        # an arbitrary id subsequence; full round-trips never hit it.
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")
