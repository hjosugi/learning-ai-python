"""A tiny statistical n-gram language model with add-k smoothing.

This is the classical (pre-neural) language model. It estimates

    P(w_t | w_{t-n+1} ... w_{t-1})

by counting how often each n-gram appears in a training corpus. With n=2 this
is a bigram model; with n=3 a trigram model. We operate over **characters** by
default (so the model needs no external tokenizer and the demo stays small), but
the same code works over any iterable of tokens.

Two ideas this teaches that carry straight over to neural LMs:

* **Next-token prediction.** A language model is just a conditional
  distribution over the next token given the context. Everything else (scoring,
  sampling, perplexity) is derived from that one object.
* **Smoothing / no zero probabilities.** A pure maximum-likelihood estimate
  assigns probability 0 to any n-gram it never saw, which makes the probability
  of any sentence containing it 0 and its log-prob -inf. **Add-k (Laplace)
  smoothing** adds a pseudo-count ``k`` to every possible continuation so
  unseen events get a small non-zero mass. Neural LMs get this "for free" from
  the softmax, which never outputs an exact zero.

Sampling is **seeded** (``random.Random(seed)``) so the demo and tests are
reproducible.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field

BOS = "\x02"  # start-of-sequence marker (ASCII STX), pads the left context
EOS = "\x03"  # end-of-sequence marker (ASCII ETX), lets the model stop


@dataclass
class NgramLM:
    """A character n-gram model with add-k smoothing.

    Attributes:
        n: order of the model (n=2 bigram, n=3 trigram). Context length is n-1.
        k: add-k smoothing pseudo-count (k=1 is classic Laplace smoothing).
        context_counts: context-tuple -> {next_token -> count}.
        vocab: the set of tokens seen in training (plus EOS).
    """

    n: int = 3
    k: float = 1.0
    context_counts: dict[tuple[str, ...], dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    vocab: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError("n must be >= 2 (n=1 is a unigram; use n>=2)")
        if self.k <= 0:
            raise ValueError("k must be > 0 so no continuation has zero mass")

    # ---- training -----------------------------------------------------------

    def _tokens(self, text: str) -> list[str]:
        """Pad a string with BOS markers on the left and one EOS on the right."""
        return [BOS] * (self.n - 1) + list(text) + [EOS]

    def train(self, corpus: list[str]) -> None:
        """Count all n-grams across every line of the corpus."""
        self.context_counts = defaultdict(lambda: defaultdict(int))
        self.vocab = set()
        for line in corpus:
            toks = self._tokens(line)
            self.vocab.update(toks)
            for i in range(self.n - 1, len(toks)):
                context = tuple(toks[i - (self.n - 1):i])
                nxt = toks[i]
                self.context_counts[context][nxt] += 1
        self.vocab.discard(BOS)  # BOS is only ever a context symbol, never predicted

    # ---- probabilities ------------------------------------------------------

    def prob(self, context: tuple[str, ...], token: str) -> float:
        """Add-k smoothed P(token | context).

        P = (count(context, token) + k) / (count(context) + k * |V|)

        The denominator's ``k * |V|`` term is what every possible continuation
        contributes, guaranteeing the distribution sums to 1 over the vocab and
        no continuation is ever exactly 0.
        """
        v = len(self.vocab)
        next_counts = self.context_counts.get(context, {})
        total = sum(next_counts.values())
        return (next_counts.get(token, 0) + self.k) / (total + self.k * v)

    def log_prob(self, text: str) -> float:
        """Total log-probability (base e) of a full string under the model.

        Higher (closer to 0) means the model finds the text more probable. We
        sum log-probs of each token given its preceding context, including the
        final EOS so that sequence length is modeled too.
        """
        toks = self._tokens(text)
        total = 0.0
        for i in range(self.n - 1, len(toks)):
            context = tuple(toks[i - (self.n - 1):i])
            total += math.log(self.prob(context, toks[i]))
        return total

    def perplexity(self, text: str) -> float:
        """Per-token perplexity = exp(-mean log-prob). Lower is better."""
        toks = self._tokens(text)
        num_predicted = len(toks) - (self.n - 1)  # tokens that have a context
        if num_predicted <= 0:
            return float("inf")
        return math.exp(-self.log_prob(text) / num_predicted)

    # ---- sampling -----------------------------------------------------------

    def sample(self, *, seed: int | None = None, max_len: int = 200) -> str:
        """Generate a string by sampling token-by-token from the model.

        We start from the all-BOS context, sample the next token from the
        smoothed conditional distribution, slide the context window, and stop at
        EOS or ``max_len``. ``seed`` makes the output reproducible.
        """
        rng = random.Random(seed)
        context = tuple([BOS] * (self.n - 1))
        out: list[str] = []
        # A stable, sorted vocab list so the cumulative-distribution walk below
        # is deterministic given the seed.
        vocab = sorted(self.vocab)
        for _ in range(max_len):
            # Build the smoothed distribution over the whole vocab for this
            # context, then sample by walking the cumulative distribution.
            weights = [self.prob(context, tok) for tok in vocab]
            tok = _weighted_choice(vocab, weights, rng)
            if tok == EOS:
                break
            out.append(tok)
            # Slide the context window: drop the oldest, append the new token.
            context = (*context[1:], tok)
        return "".join(out)


def _weighted_choice(items: list[str], weights: list[float], rng: random.Random) -> str:
    """Pick one item with probability proportional to its weight.

    Equivalent to ``random.choices(items, weights)[0]`` but written out so the
    sampling mechanism (inverse-CDF on a uniform draw) is visible.
    """
    total = sum(weights)
    r = rng.random() * total
    upto = 0.0
    for item, w in zip(items, weights):
        upto += w
        if upto >= r:
            return item
    return items[-1]  # floating-point fallback
