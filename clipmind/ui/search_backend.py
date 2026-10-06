"""Adapter between the popup view-model and the existing search/storage layers.

The view-model knows nothing about ``hybrid_search`` or SQLite; it only calls
the small :class:`~clipmind.ui.view_model.SearchBackend` protocol. This module
implements that protocol:

* an empty query lists the most recent items (never a ranked search),
* a non-empty query runs the Phase 3 hybrid + time-aware search,
* a dismissed time chip re-runs the query with time parsing switched off,
* a locked database or an unavailable model becomes a ``SearchOutcome.error``
  instead of an exception, and every query logs its latency at debug level.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Optional, Sequence

from ..config import Config
from ..embeddings.encoder import Embedder, EmbeddingModelUnavailable
from ..search.hybrid import HybridOutcome, HybridResult, HybridSearcher, hybrid_search
from ..storage.db import Database
from .view_model import ResultItem, SearchOutcome, SearchRequest, result_items_from_records

logger = logging.getLogger(__name__)


def _chip_label(phrase: Optional[str]) -> Optional[str]:
    """Return a display label for a matched time phrase (e.g. ``"Yesterday"``).

    Multi-word phrases are title-cased when they are all words, but left as-is
    (minus the first letter) when they contain numbers, so ``"2 hours ago"``
    does not become ``"2 Hours Ago"``.
    """
    if not phrase:
        return None
    cleaned = phrase.strip()
    if not cleaned:
        return None
    if all(part.isalpha() for part in cleaned.split()):
        return cleaned.title()
    return cleaned[0].upper() + cleaned[1:]


def _items_from_results(results: Sequence[HybridResult]) -> tuple[ResultItem, ...]:
    """Convert hybrid results into view-model rows."""
    return tuple(
        ResultItem(
            id=result.id,
            content=result.content,
            content_type=result.content_type,
            created_at=result.created_at,
            source_app=result.source_app,
            final_score=result.final_score,
            semantic_score=result.semantic_score,
            keyword_score=result.keyword_score,
            content_hash=result.record.content_hash,
            embedding=result.record.embedding,
        )
        for result in results
    )


class HybridSearchBackend:
    """Runs the real search/storage operations behind the popup.

    Args:
        db: Open database (shared with the watcher; the database is thread-safe).
        embedder: Embedder to reuse so the model is loaded once. ``None`` lets
            the search layer resolve the configured embedder on first use, which
            happens on the search worker thread - never the UI thread.
        config: Configuration (weights, threshold, list limit).
        clipboard: Optional object with ``copy(text)`` used for copy-back. When
            omitted, ``copy_to_clipboard`` is a no-op (handy for ``--check``).
    """

    def __init__(
        self,
        db: Database,
        *,
        embedder: Optional[Embedder] = None,
        config: Optional[Config] = None,
        clipboard=None,
    ) -> None:
        """Store the dependencies."""
        self._db = db
        self._embedder = embedder
        self._config = config or Config()
        self._clipboard = clipboard

    def run_search(self, request: SearchRequest) -> SearchOutcome:
        """Run one search and never raise for an expected failure.

        Args:
            request: The query, content-type filter and result cap.

        Returns:
            A :class:`SearchOutcome`; ``error`` is set (instead of raising) when
            the model is unavailable or any other failure occurs.
        """
        started = time.monotonic()
        try:
            outcome = self._run(request)
        except EmbeddingModelUnavailable as exc:
            outcome = SearchOutcome(error=str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("search failed")
            outcome = SearchOutcome(error=f"search failed: {exc}")
        elapsed_ms = (time.monotonic() - started) * 1000.0
        logger.debug(
            "search %r (type=%s, %d result(s)) took %.1f ms",
            request.query,
            request.content_type or "all",
            len(outcome.items),
            elapsed_ms,
        )
        return replace(outcome, elapsed_ms=elapsed_ms)

    def _run(self, request: SearchRequest) -> SearchOutcome:
        """Dispatch to the recent-list or hybrid path."""
        query = (request.query or "").strip()
        if not query:
            records = self._db.list_recent(
                limit=request.top_k, content_type=request.content_type
            )
            return SearchOutcome(items=result_items_from_records(records))

        if request.ignore_time:
            # The user dismissed the time chip: treat the phrase as plain text.
            hybrid = HybridSearcher(
                self._db, embedder=self._embedder, config=self._config
            ).search(
                query,
                top_k=request.top_k,
                content_type=request.content_type,
            )
            return self._to_outcome(hybrid, time_label=None)

        outcome = hybrid_search(
            self._db,
            query,
            top_k=request.top_k,
            embedder=self._embedder,
            config=self._config,
            content_type=request.content_type,
        )
        return self._to_outcome(outcome)

    def _to_outcome(
        self, outcome: HybridOutcome, *, time_label: Optional[str] = "auto"
    ) -> SearchOutcome:
        """Convert a hybrid outcome, deriving the time chip label/note."""
        query = outcome.query
        if time_label == "auto":
            derived = _chip_label(query.matched_phrase) if query.has_range else None
        else:
            derived = time_label
        note = query.note if query.ambiguous else None
        return SearchOutcome(
            items=_items_from_results(outcome.results),
            time_label=derived,
            time_note=note,
        )

    def copy_to_clipboard(self, text: str) -> None:
        """Write ``text`` to the clipboard (suppressing re-capture)."""
        if self._clipboard is None:
            logger.debug("no clipboard writer configured; copy is a no-op")
            return
        self._clipboard.copy(text)

    def delete_item(self, item_id: int) -> bool:
        """Delete one entry; return False on any failure."""
        try:
            return bool(self._db.delete_by_id(item_id))
        except Exception:  # pragma: no cover - defensive
            logger.exception("delete of item %s failed", item_id)
            return False

    def restore_item(self, item: ResultItem) -> bool:
        """Re-insert a deleted entry with its original timestamp and embedding."""
        try:
            result = self._db.insert_clip(
                content=item.content,
                content_type=item.content_type,
                source_app=item.source_app,
                created_at=item.created_at,
                content_hash=item.content_hash,
                embedding=item.embedding,
            )
            return result.inserted
        except Exception:  # pragma: no cover - defensive
            logger.exception("restore of item %s failed", item.id)
            return False


__all__ = ["HybridSearchBackend", "_chip_label"]
