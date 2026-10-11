"""Long-lived, in-memory vector index with metadata for mask-based filtering.

The brute-force :class:`~clipmind.search.vector_index.NumpyVectorIndex` rebuilds
its matrix from decoded SQLite BLOBs. That is fine for a single search but ruinous
per keystroke: every search re-read the whole table. This module holds the matrix
(plus the ids, timestamps and content types needed to filter it) once, and is
updated incrementally as the database changes, so a query is a single matmul.

Filters are **boolean masks** over the row-aligned arrays rather than a rebuilt
matrix: a date window is two string comparisons against the stored ISO-8601
timestamps and a type filter is one equality check. Masking never copies the
matrix, so a filtered search costs the same as an unfiltered one.

Because filters compare the *stored* timestamp text, the index sees exactly the
same ordering as the previous SQL push-down of ``created_at >= ?`` / ``< ?``.
"""

from __future__ import annotations

import threading
from typing import Optional, Sequence

import numpy as np

from ..embeddings.codec import VECTOR_DTYPE, decode_vector, normalize
from ..storage.db import DatabaseChange, coerce_timestamp
from .vector_index import VectorHit

#: Row from :meth:`~clipmind.storage.db.Database.index_snapshot`.
SnapshotRow = tuple[int, str, str, bytes]


class EmbeddingIndex:
    """Row-aligned matrix of normalised embeddings with id/date/type metadata.

    All mutating and reading methods take an internal lock, so one index can be
    shared between the search thread and the embedding worker. Methods never call
    back into the database: callers must take a snapshot first and hand the rows
    in.

    The index starts empty and ``loaded == False``; call :meth:`load` once with a
    snapshot, then :meth:`apply` each :class:`DatabaseChange`, or simply
    :meth:`load` again to rebuild from scratch.
    """

    def __init__(self, dimension: int = 0) -> None:
        """Create an empty index.

        Args:
            dimension: Expected vector dimension (``0`` means \"learn on load\").
        """
        self._lock = threading.RLock()
        self._dimension = dimension
        self._ids = np.zeros(0, dtype=np.int64)
        self._vectors = np.zeros((0, dimension), dtype=VECTOR_DTYPE)
        self._created = np.zeros(0, dtype=object)  # stored ISO-8601 strings
        self._types = np.zeros(0, dtype=object)  # content types
        self._position: dict[int, int] = {}
        self._loaded = False

    # ------------------------------------------------------------------ state

    @property
    def loaded(self) -> bool:
        """True once :meth:`load` has built the index at least once."""
        with self._lock:
            return self._loaded

    @property
    def size(self) -> int:
        """Number of indexed vectors."""
        with self._lock:
            return int(self._ids.shape[0])

    @property
    def dimension(self) -> int:
        """Vector dimensionality (``0`` while empty)."""
        with self._lock:
            return int(self._vectors.shape[1]) if self._vectors.ndim == 2 else 0

    # ------------------------------------------------------------------ building

    def load(self, rows: Sequence[SnapshotRow]) -> None:
        """Replace the whole index with ``rows`` from a database snapshot.

        Args:
            rows: ``(id, content_type, created_at, embedding_blob)`` tuples, in
                the order the caller wants ties to break in.
        """
        ids: list[int] = []
        created: list[str] = []
        types: list[str] = []
        vectors: list[np.ndarray] = []
        for item_id, content_type, created_at, blob in rows:
            vector = decode_vector(blob)
            if vector.size == 0:
                continue
            ids.append(int(item_id))
            created.append(created_at)
            types.append(content_type)
            vectors.append(vector)

        if vectors:
            matrix = np.vstack(vectors).astype(VECTOR_DTYPE, copy=False)
            matrix = normalize(matrix)
            id_array = np.asarray(ids, dtype=np.int64)
            created_array = np.asarray(created, dtype=object)
            type_array = np.asarray(types, dtype=object)
            dimension = int(matrix.shape[1])
        else:
            matrix = np.zeros((0, self._dimension), dtype=VECTOR_DTYPE)
            id_array = np.zeros(0, dtype=np.int64)
            created_array = np.zeros(0, dtype=object)
            type_array = np.zeros(0, dtype=object)
            dimension = self._dimension

        with self._lock:
            self._dimension = dimension
            self._ids = id_array
            self._vectors = matrix
            self._created = created_array
            self._types = type_array
            self._position = {int(key): pos for pos, key in enumerate(id_array)}
            self._loaded = True

    def apply(self, change: DatabaseChange) -> None:
        """Apply one committed :class:`DatabaseChange` incrementally.

        A no-op until the index has been loaded at least once: before that a
        full :meth:`load` is cheaper and always correct.
        """
        with self._lock:
            if not self._loaded:
                return
            if change.kind == "clear":
                self._clear_locked()
            elif change.kind == "delete" and change.item_id is not None:
                self._remove_locked([change.item_id])
            elif change.kind == "delete_many":
                self._remove_locked(list(change.item_ids))
            elif change.kind in {"insert", "embedding"}:
                if (
                    change.embedding
                    and change.item_id is not None
                    and change.content_type is not None
                    and change.created_at is not None
                ):
                    self._upsert_locked(
                        change.item_id,
                        change.embedding,
                        change.created_at,
                        change.content_type,
                    )

    def clear(self) -> None:
        """Drop every vector (used by :meth:`clipmind.search.semantic`)."""
        with self._lock:
            self._clear_locked()

    def _clear_locked(self) -> None:
        self._ids = np.zeros(0, dtype=np.int64)
        self._vectors = np.zeros((0, self._dimension), dtype=VECTOR_DTYPE)
        self._created = np.zeros(0, dtype=object)
        self._types = np.zeros(0, dtype=object)
        self._position = {}

    def _upsert_locked(
        self, item_id: int, blob: bytes, created_at: str, content_type: str
    ) -> None:
        vector = normalize(decode_vector(blob).reshape(1, -1))[0]
        position = self._position.get(int(item_id))
        if position is not None:
            self._vectors[position] = vector
            self._created[position] = created_at
            self._types[position] = content_type
            return
        if self._ids.shape[0] == 0:
            self._vectors = vector.reshape(1, -1)
        else:
            self._vectors = np.vstack([self._vectors, vector.reshape(1, -1)])
        self._dimension = int(self._vectors.shape[1])
        self._ids = np.append(self._ids, np.int64(item_id))
        self._created = np.append(self._created, created_at)
        self._types = np.append(self._types, content_type)
        self._position[int(item_id)] = int(self._ids.shape[0]) - 1

    def _remove_locked(self, item_ids: Sequence[int]) -> None:
        positions = sorted(
            {self._position[int(i)] for i in item_ids if int(i) in self._position}
        )
        if not positions:
            return
        keep = np.ones(self._ids.shape[0], dtype=bool)
        keep[positions] = False
        self._ids = self._ids[keep]
        self._vectors = self._vectors[keep]
        self._created = self._created[keep]
        self._types = self._types[keep]
        self._position = {int(key): pos for pos, key in enumerate(self._ids)}

    # ------------------------------------------------------------------ filters

    def select(
        self,
        *,
        content_type: Optional[str] = None,
        since: Optional[object] = None,
        until: Optional[object] = None,
    ) -> np.ndarray:
        """Return a boolean mask of rows matching the optional filters."""
        with self._lock:
            size = int(self._ids.shape[0])
            mask = np.ones(size, dtype=bool)
            if size == 0:
                return mask
            if content_type:
                mask &= self._types == content_type
            if since is not None:
                mask &= self._created >= coerce_timestamp(since)
            if until is not None:
                mask &= self._created < coerce_timestamp(until)
            return mask

    def count(
        self,
        *,
        content_type: Optional[str] = None,
        since: Optional[object] = None,
        until: Optional[object] = None,
    ) -> int:
        """Return how many rows match the optional filters."""
        return int(self.select(content_type=content_type, since=since, until=until).sum())

    # ------------------------------------------------------------------ search

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        *,
        content_type: Optional[str] = None,
        since: Optional[object] = None,
        until: Optional[object] = None,
    ) -> list[VectorHit]:
        """Return the ``top_k`` closest rows within the optional filters.

        Ordering matches :class:`~clipmind.search.vector_index.NumpyVectorIndex`:
        the best ``take`` rows are picked in O(n) with ``argpartition`` and then
        sorted, so results are ordered by descending cosine similarity.
        """
        if top_k <= 0:
            return []
        with self._lock:
            if self._ids.shape[0] == 0:
                return []
            mask = self.select(content_type=content_type, since=since, until=until)
            candidate_indices = np.flatnonzero(mask)
            if candidate_indices.size == 0:
                return []

            query_array = np.asarray(query, dtype=VECTOR_DTYPE).reshape(-1)
            if query_array.shape[0] != self._vectors.shape[1]:
                raise ValueError(
                    f"query has {query_array.shape[0]} dims but index holds "
                    f"{self._vectors.shape[1]}"
                )
            query_vector = normalize(query_array.reshape(1, -1))[0]
            candidate_vectors = self._vectors[candidate_indices]
            scores = (candidate_vectors @ query_vector).astype(VECTOR_DTYPE, copy=False)

            take = min(top_k, candidate_indices.size)
            if take < candidate_indices.size:
                top_local = np.argpartition(-scores, take - 1)[:take]
            else:
                top_local = np.arange(candidate_indices.size)
            top_local = top_local[np.argsort(-scores[top_local], kind="stable")]

            return [
                VectorHit(
                    key=int(self._ids[candidate_indices[local]]),
                    score=float(scores[local]),
                )
                for local in top_local
            ]
