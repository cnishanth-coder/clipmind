"""Test doubles shared across the suite.

The real ``all-MiniLM-L6-v2`` model is 90 MB of weights and a ~1 s import; unit
tests must not pay that. :class:`FakeEmbedder` implements the same
:class:`~clipmind.embeddings.encoder.Embedder` contract with a deterministic
bag-of-words hashing trick, so keyword overlap between a query and a document
produces a higher cosine score. That is enough to test ranking, thresholds,
filtering and caching without any model or network access.
"""

from __future__ import annotations

import re
import time
import zlib
from typing import Optional, Sequence

import numpy as np

from clipmind.embeddings.codec import VECTOR_DTYPE, normalize

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


class FakeEmbedder:
    """Deterministic, dependency-free stand-in for a sentence encoder.

    Attributes:
        call_count: How many times :meth:`embed_texts` was called (useful to
            assert that an empty index never triggers a model load).
        embedded_texts: Every text ever passed in, in call order.
    """

    def __init__(
        self,
        dimension: int = 16,
        *,
        delay: float = 0.0,
        fail: bool = False,
    ) -> None:
        """Configure the fake.

        Args:
            dimension: Vector size produced.
            delay: Seconds to sleep per :meth:`embed_texts` call (used to prove
                the capture loop does not block on embedding).
            fail: When True, every call raises, to exercise error handling.
        """
        self._dimension = dimension
        self.delay = delay
        self.fail = fail
        self.call_count = 0
        self.embedded_texts: list[str] = []

    @property
    def dimension(self) -> int:
        """Vector dimensionality."""
        return self._dimension

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts`` with a hashed bag-of-words.

        Each token increments one hashed bucket; vectors are L2-normalised so
        the output matches the real embedder's contract.
        """
        self.call_count += 1
        self.embedded_texts.extend(texts)
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("fake embedder failure")

        vectors = np.zeros((len(texts), self._dimension), dtype=VECTOR_DTYPE)
        for row, text in enumerate(texts):
            for token in _TOKEN_RE.findall((text or "").lower()):
                vectors[row, zlib.crc32(token.encode("utf-8")) % self._dimension] += 1.0
        return normalize(vectors)

    def embed_one(self, text: str) -> np.ndarray:
        """Convenience helper returning a single vector."""
        return self.embed_texts([text])[0]


class BrokenEmbedder:
    """Embedder that always raises, for error-path tests."""

    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Vector dimensionality."""
        return self._dimension

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Always raise."""
        raise RuntimeError("embedder unavailable")


def make_vector(*values: float) -> np.ndarray:
    """Build a float32 vector from literal values."""
    return np.asarray(values, dtype=VECTOR_DTYPE)


def embed_into_db(
    db,
    items: Sequence[tuple[str, str]],
    embedder: Optional[FakeEmbedder] = None,
) -> FakeEmbedder:
    """Insert ``(content, content_type)`` pairs and store their embeddings.

    Args:
        db: Target database.
        items: ``(content, content_type)`` pairs.
        embedder: Fake to reuse; a fresh one is created when omitted.

    Returns:
        The embedder that produced the vectors.
    """
    from clipmind.embeddings.codec import encode_vector

    embedder = embedder or FakeEmbedder()
    for content, content_type in items:
        result = db.insert_clip(content, content_type=content_type)
        assert result.inserted and result.record is not None
        vector = embedder.embed_one(content)
        db.set_embedding(result.record.id, encode_vector(vector))
    return embedder
