"""Semantic (meaning-based) search over stored embeddings.

The public entry point is :meth:`SemanticSearcher.search` (and the module-level
:func:`search_semantic` convenience wrapper). It embeds the query with the same
local model used for ingestion, then ranks stored vectors by cosine similarity
through the :class:`~clipmind.search.vector_index.VectorIndex` interface.

Query planning (time phrases, hybrid weighting) is Phase 3 and deliberately
absent here: this module only answers "which stored entries mean the most
similar thing?".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

from ..config import Config
from ..embeddings.codec import stack_vectors
from ..embeddings.encoder import Embedder, get_embedder
from ..storage.db import ClipRecord, Database
from .vector_index import NumpyVectorIndex, VectorIndex

logger = logging.getLogger(__name__)

#: Factory used to build an index from decoded vectors.
IndexFactory = Callable[[list[int], object], VectorIndex]


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One semantic search hit.

    Attributes:
        record: The full stored entry this hit refers to.
        score: Cosine similarity between the query and the entry (higher is
            better; ``1.0`` means identical direction).
    """

    record: ClipRecord
    score: float

    @property
    def id(self) -> int:
        """Database id of the matched entry."""
        return self.record.id

    @property
    def content(self) -> str:
        """Full stored content of the matched entry."""
        return self.record.content

    @property
    def content_type(self) -> str:
        """Detected content type (``url``/``code``/``email``/``number``/``text``)."""
        return self.record.content_type

    @property
    def created_at(self) -> datetime:
        """UTC capture timestamp of the matched entry."""
        return self.record.created_at

    @property
    def source_app(self) -> Optional[str]:
        """Source application name, when known."""
        return self.record.source_app

    def preview(self, max_len: int = 80) -> str:
        """Return a single-line preview of the content."""
        return self.record.preview(max_len)

    def relative_time(self, now: Optional[datetime] = None) -> str:
        """Return a short age string such as ``"3m ago"``."""
        return self.record.relative_time(now)


class SemanticSearcher:
    """Rank stored entries by semantic similarity to a query.

    Rebuilding the numpy matrix on every query would be wasteful for the
    Phase 5 UI (searches per keystroke), so the decoded index is cached and
    invalidated by a cheap ``(row_count, max_id)`` fingerprint. Instances are
    not thread-safe: the ``last_*`` diagnostics describe the most recent call.
    """

    def __init__(
        self,
        db: Database,
        *,
        embedder: Optional[Embedder] = None,
        config: Optional[Config] = None,
        index_factory: Optional[IndexFactory] = None,
    ) -> None:
        """Create a searcher.

        Args:
            db: Database holding the entries and their embeddings.
            embedder: Embedder for the query text; resolved lazily from config
                when omitted (so constructing a searcher never loads the model).
            config: Configuration (default top-k, minimum score).
            index_factory: Builds the vector index; defaults to
                :class:`~clipmind.search.vector_index.NumpyVectorIndex`.
        """
        self.db = db
        self.config = config or Config()
        self._embedder = embedder
        self._index_factory = index_factory or (
            lambda keys, vectors: NumpyVectorIndex(keys, vectors)
        )
        self._cache: dict[
            tuple[Optional[str], Optional[str], Optional[str]],
            tuple[tuple[int, int], list[ClipRecord], VectorIndex],
        ] = {}

        self.last_best_score: Optional[float] = None
        self.last_index_size: int = 0
        self.last_threshold: float = self.config.min_score

    # ------------------------------------------------------------------ embedding

    @property
    def embedder(self) -> Embedder:
        """Return the embedder, creating the configured one on first access."""
        if self._embedder is None:
            self._embedder = get_embedder(self.config)
        return self._embedder

    # ------------------------------------------------------------------ index

    def index_size(
        self,
        content_type: Optional[str] = None,
        *,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> int:
        """Return how many embedded entries the given filter covers."""
        _records, index = self._index(content_type, since, until)
        return index.size

    def invalidate(self) -> None:
        """Drop the cached index (call after external writes)."""
        self._cache.clear()

    def _index(
        self,
        content_type: Optional[str],
        since: Optional[datetime],
        until: Optional[datetime],
    ) -> tuple[list[ClipRecord], VectorIndex]:
        """Return cached ``(records, index)`` for the given filters.

        The cache key includes the time bounds because a different range
        selects a different subset of rows and therefore a different matrix.
        """
        fingerprint = self.db.indexed_fingerprint(content_type, since=since, until=until)
        cache_key = (
            content_type,
            since.isoformat() if since else None,
            until.isoformat() if until else None,
        )
        cached = self._cache.get(cache_key)
        if cached is not None and cached[0] == fingerprint:
            return cached[1], cached[2]

        records = self.db.list_indexed(content_type, since=since, until=until)
        keys = [record.id for record in records]
        vectors = self._decode(records)
        index = self._index_factory(keys, vectors)
        self._cache[cache_key] = (fingerprint, records, index)
        logger.debug(
            "built vector index: %d vectors for type=%s range=%s..%s",
            index.size,
            content_type,
            cache_key[1],
            cache_key[2],
        )
        return records, index

    @staticmethod
    def _decode(records: list[ClipRecord]):
        """Decode the embedding BLOBs of ``records`` into an ``(n, dim)`` matrix.

        ``Database.list_indexed`` only returns rows with a non-NULL embedding,
        so the blob list aligns 1:1 with ``records`` and with the index keys.
        """
        return stack_vectors([record.embedding or b"" for record in records])

    # ------------------------------------------------------------------ search

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        *,
        content_type: Optional[str] = None,
        min_score: Optional[float] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> list[SearchResult]:
        """Find the stored entries that mean the most similar thing to ``query``.

        Args:
            query: Natural-language query text.
            top_k: Maximum number of results (defaults to ``config.search_top_k``).
            content_type: Restrict to one content type (``"code"``, ``"url"``...).
            min_score: Minimum cosine similarity to keep a hit; defaults to
                ``config.min_score``. Raise it for precision, lower it for recall.
            since: Only consider entries created at or after this instant.
            until: Only consider entries created strictly before this instant.

        Returns:
            Results ordered by descending score. Empty when the database has no
            embeddings yet, the query is blank, or nothing clears the threshold.
            Diagnostics for the empty cases are left in :attr:`last_best_score`
            and :attr:`last_index_size`.
        """
        top_k = top_k if top_k is not None else self.config.search_top_k
        threshold = self.config.min_score if min_score is None else min_score
        self.last_threshold = threshold

        query = (query or "").strip()
        if not query or top_k <= 0:
            self.last_best_score = None
            self.last_index_size = 0
            return []

        records, index = self._index(content_type, since, until)
        self.last_index_size = index.size
        if index.size == 0:
            # Nothing to compare against - do not even load the model.
            self.last_best_score = None
            return []

        query_vector = self.embedder.embed_texts([query])[0]
        hits = index.search(query_vector, top_k)
        self.last_best_score = hits[0].score if hits else None

        by_id = {record.id: record for record in records}
        results: list[SearchResult] = []
        for hit in hits:
            if hit.score < threshold:
                continue
            record = by_id.get(hit.key)
            if record is not None:
                results.append(SearchResult(record=record, score=hit.score))
        return results


def search_semantic(
    db: Database,
    query: str,
    top_k: int = 5,
    *,
    embedder: Optional[Embedder] = None,
    config: Optional[Config] = None,
    content_type: Optional[str] = None,
    min_score: Optional[float] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> list[SearchResult]:
    """Convenience wrapper: semantic search in one call.

    Args:
        db: Database to search.
        query: Natural-language query text.
        top_k: Maximum number of results.
        embedder: Optional embedder (resolved from config when omitted).
        config: Optional configuration.
        content_type: Optional content-type filter.
        min_score: Optional minimum cosine similarity override.
        since: Optional lower time bound (inclusive).
        until: Optional upper time bound (exclusive).

    Returns:
        Ranked :class:`SearchResult` objects, best first.
    """
    searcher = SemanticSearcher(db, embedder=embedder, config=config)
    return searcher.search(
        query,
        top_k,
        content_type=content_type,
        min_score=min_score,
        since=since,
        until=until,
    )
