"""Semantic (meaning-based) search over stored embeddings.

The public entry point is :meth:`SemanticSearcher.search` (and the module-level
:func:`search_semantic` convenience wrapper). It embeds the query with the same
local model used for ingestion, then ranks stored vectors by cosine similarity
through a long-lived, in-memory :class:`~clipmind.search.embedding_index.EmbeddingIndex`.

The index is built once (from a single SQLite snapshot) and then kept in sync
with the database incrementally: the searcher registers itself as a
:class:`~clipmind.storage.db.DatabaseListener` and folds each committed insert,
embedding, delete, purge and clear into the matrix. Time and type filters are
boolean masks over the row-aligned metadata arrays, so a filtered query is a
masked matmul - never a re-read of the table.

Query planning (time phrases, hybrid weighting) is Phase 3 and deliberately
absent here: this module only answers \"which stored entries mean the most
similar thing?\".
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ..config import Config
from ..embeddings.encoder import (
    Embedder,
    EmbeddingModelMismatch,
    embed_queries,
    get_embedder,
)
from ..storage.db import ClipRecord, Database, DatabaseChange
from .embedding_index import EmbeddingIndex

logger = logging.getLogger(__name__)


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
        """Return a short age string such as ``\"3m ago\"``."""
        return self.record.relative_time(now)


class SemanticSearcher:
    """Rank stored entries by semantic similarity to a query.

    One searcher owns one :class:`EmbeddingIndex`. The index is loaded lazily on
    the first query and then updated in place as the database changes, so the
    per-query cost is a masked matmul plus one model forward pass - not a full
    re-read of every embedding BLOB.

    Instances are safe to share between threads. The ``last_*`` diagnostics
    describe the most recent call.
    """

    def __init__(
        self,
        db: Database,
        *,
        embedder: Optional[Embedder] = None,
        config: Optional[Config] = None,
    ) -> None:
        """Create a searcher.

        Args:
            db: Database holding the entries and their embeddings. The searcher
                registers as a listener so it sees committed changes.
            embedder: Embedder for the query text; resolved lazily from config
                when omitted (so constructing a searcher never loads the model).
            config: Configuration (default top-k, minimum score).
        """
        self.db = db
        self.config = config or Config()
        self._embedder = embedder
        self._index = EmbeddingIndex()
        self._index_loaded = False
        self._index_generation = -1
        self._index_lock = threading.RLock()

        self.last_best_score: Optional[float] = None
        self.last_index_size: int = 0
        self.last_threshold: float = self.config.min_score

        db.add_listener(self)

    # ------------------------------------------------------------------ embedding

    @property
    def embedder(self) -> Embedder:
        """Return the embedder, creating the configured one on first access."""
        if self._embedder is None:
            self._embedder = get_embedder(self.config)
        return self._embedder

    # ------------------------------------------------------------------ index

    @property
    def index(self) -> EmbeddingIndex:
        """Return the underlying long-lived index (ensuring it is loaded)."""
        self._ensure_index()
        return self._index

    def index_size(
        self,
        content_type: Optional[str] = None,
        *,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> int:
        """Return how many embedded entries the given filter covers."""
        self._ensure_index()
        return self._index.count(
            content_type=content_type, since=since, until=until
        )

    def invalidate(self) -> None:
        """Drop the cached index so the next query reloads it from the database."""
        with self._index_lock:
            self._index.clear()
            self._index_loaded = False
            self._index_generation = -1

    def on_database_change(self, change: DatabaseChange) -> None:
        """Fold one committed change into the in-memory index.

        Called by the database while the writer holds its lock; this method only
        touches the index (never the database) to avoid lock inversion.
        """
        with self._index_lock:
            if not self._index_loaded:
                # A snapshot taken later will include this change; before loading
                # there is nothing to update.
                return
            self._index.apply(change)
            self._index_generation = change.generation

    def _check_embedding_model(self) -> None:
        """Refuse to search when the stored vectors came from another model.

        A database that predates model tagging has embeddings but no recorded
        model name; it is adopted silently under the configured model so the
        upgrade needs no manual step. Once a name is recorded, changing the
        configured model is a hard error until ``reindex --force`` rebuilds the
        vectors (they are not comparable across models, and the dimension can
        differ too).
        """
        stored = self.db.get_embedding_model()
        configured = self.config.embedding_model
        if stored is None:
            if self.db.indexed_fingerprint()[0] > 0:
                self.db.set_embedding_model(configured)
            return
        if stored != configured:
            raise EmbeddingModelMismatch(
                f"stored embeddings were created with model {stored!r}, but the "
                f"configured model is {configured!r}. Run `clipmind reindex --force` "
                "to rebuild them for the new model."
            )

    def _ensure_index(self) -> None:
        """Load the index on first use and refresh it if a change was missed."""
        with self._index_lock:
            if self._index_loaded and self.db.change_counter == self._index_generation:
                return
            first_load = not self._index_loaded
        if first_load:
            self._check_embedding_model()
        # Take the snapshot outside the index lock: the database lock must never
        # be acquired while holding it (listeners acquire it in the other order).
        generation, rows = self.db.index_snapshot()
        with self._index_lock:
            if self._index_loaded and self._index_generation >= generation:
                return  # an incremental update already covered this snapshot
            self._index.load(rows)
            self._index_loaded = True
            self._index_generation = generation
            logger.debug("loaded embedding index: %d vectors", self._index.size)

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
            content_type: Restrict to one content type (``\"code\"``, ``\"url\"``...).
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

        self._ensure_index()
        index_size = self._index.count(
            content_type=content_type, since=since, until=until
        )
        self.last_index_size = index_size
        if index_size == 0:
            # Nothing to compare against - do not even load the model.
            self.last_best_score = None
            return []

        # The model call is deliberately outside the index lock: a slow forward
        # pass must not block the embedding worker's incremental updates.
        query_vector = embed_queries(self.embedder, [query])[0]
        if query_vector.shape[0] != self._index.dimension:
            raise EmbeddingModelMismatch(
                f"stored embeddings have {self._index.dimension} dimensions but the "
                f"configured model {self.config.embedding_model!r} produces "
                f"{query_vector.shape[0]}. Run `clipmind reindex --force` to rebuild "
                "them for the new model."
            )
        hits = self._index.search(
            query_vector,
            top_k,
            content_type=content_type,
            since=since,
            until=until,
        )
        self.last_best_score = hits[0].score if hits else None

        results: list[SearchResult] = []
        for hit in hits:
            if hit.score < threshold:
                continue
            record = self.db.get_by_id(hit.key)
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
