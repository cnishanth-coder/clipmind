"""Vector similarity search behind a swappable interface.

:class:`VectorIndex` is the seam for Phase 2's brute-force implementation.
Everything else (the searcher, the CLI, the future UI) talks to this protocol
only, so dropping in FAISS or Chroma later means writing a new class, not
touching callers.

Brute force is intentional: cosine similarity over 50,000 stored 384-dim vectors
is a single ``(50000, 384) @ (384,)`` matmul, which numpy does in a few
milliseconds. An approximate index only pays off beyond that scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from ..embeddings.codec import VECTOR_DTYPE, cosine_similarity, normalize


@dataclass(frozen=True, slots=True)
class VectorHit:
    """One scored match from a vector search.

    Attributes:
        key: The id the vector was stored under (an ``items.id``).
        score: Cosine similarity in ``[-1, 1]``; higher is closer.
    """

    key: int
    score: float


@runtime_checkable
class VectorIndex(Protocol):
    """Thin interface over an embeddable-vector store.

    Any replacement (FAISS, Chroma, sqlite-vec) must keep these semantics:
    vectors are L2-normalised, ``search`` returns at most ``top_k`` hits ordered
    by descending similarity, and keys are caller-defined integers.
    """

    @property
    def size(self) -> int:
        """Number of vectors currently held."""
        ...

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        allowed_keys: Optional[set[int]] = None,
    ) -> list[VectorHit]:
        """Return the ``top_k`` closest vectors to ``query``."""
        ...


class NumpyVectorIndex:
    """Brute-force cosine similarity over a contiguous numpy matrix."""

    def __init__(
        self,
        keys: Optional[Sequence[int]] = None,
        vectors: Optional[np.ndarray] = None,
    ) -> None:
        """Build an index from optional keys and vectors.

        Args:
            keys: Integer keys, one per row of ``vectors``.
            vectors: ``(n, dim)`` matrix; rows are normalised on the way in.
        """
        self._keys: np.ndarray = np.zeros(0, dtype=np.int64)
        self._vectors: np.ndarray = np.zeros((0, 0), dtype=VECTOR_DTYPE)
        if keys is not None and vectors is not None:
            self.add(keys, vectors)

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return int(self._vectors.shape[0])

    @property
    def dimension(self) -> int:
        """Vector dimensionality (``0`` while the index is empty)."""
        return int(self._vectors.shape[1]) if self._vectors.ndim == 2 else 0

    def add(self, keys: Sequence[int], vectors: np.ndarray) -> None:
        """Append vectors to the index.

        Args:
            keys: Integer keys, one per row.
            vectors: ``(n, dim)`` matrix of embeddings.

        Raises:
            ValueError: If the key count does not match the row count, or the
                dimensionality disagrees with what is already stored.
        """
        array = np.asarray(vectors, dtype=VECTOR_DTYPE)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[0] != len(keys):
            raise ValueError(
                f"got {len(keys)} keys for {array.shape[0]} vectors"
            )
        if self.size and array.shape[1] != self.dimension:
            raise ValueError(
                f"dimension mismatch: index holds {self.dimension}, got {array.shape[1]}"
            )
        array = normalize(array)
        self._vectors = array if self.size == 0 else np.vstack([self._vectors, array])
        new_keys = np.asarray(keys, dtype=np.int64)
        self._keys = new_keys if self._keys.size == 0 else np.concatenate([self._keys, new_keys])

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        allowed_keys: Optional[set[int]] = None,
    ) -> list[VectorHit]:
        """Return the ``top_k`` most similar vectors, best first.

        Args:
            query: Query vector (normalised internally).
            top_k: Maximum number of hits. ``<= 0`` returns an empty list.
            allowed_keys: Optional whitelist of keys to consider.

        Returns:
            Hits sorted by descending cosine similarity.

        Raises:
            ValueError: If ``query`` dimensionality does not match the index.
        """
        if top_k <= 0 or self.size == 0:
            return []

        query_array = np.asarray(query, dtype=VECTOR_DTYPE).reshape(-1)
        if query_array.shape[0] != self.dimension:
            raise ValueError(
                f"query has {query_array.shape[0]} dims but index holds {self.dimension}"
            )

        scores = cosine_similarity(query_array, self._vectors)
        candidate_indices = np.arange(self.size)
        if allowed_keys is not None:
            mask = np.fromiter(
                (int(key) in allowed_keys for key in self._keys), dtype=bool, count=self.size
            )
            candidate_indices = candidate_indices[mask]
            scores = scores[mask]
            if candidate_indices.size == 0:
                return []

        take = min(top_k, candidate_indices.size)
        # argpartition picks the top `take` in O(n) before sorting just those.
        if take < candidate_indices.size:
            top_local = np.argpartition(-scores, take - 1)[:take]
        else:
            top_local = np.arange(candidate_indices.size)
        top_local = top_local[np.argsort(-scores[top_local], kind="stable")]

        return [
            VectorHit(key=int(self._keys[candidate_indices[i]]), score=float(scores[i]))
            for i in top_local
        ]
