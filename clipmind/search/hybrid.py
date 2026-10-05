"""Hybrid search: fuse semantic similarity with keyword (BM25) relevance.

Final score is a weighted sum::

    final = w_sem * semantic + w_kw * (keyword * coverage)

with ``w_sem = 0.7`` and ``w_kw = 0.3`` by default (configurable). ``coverage``
is the fraction of query tokens present in the entry, so a partial keyword match
cannot masquerade as an exact one.

Two behaviours matter and are covered by tests:

* An item found by only one retriever is still scored - a keyword-only hit gets
  ``semantic = 0`` and vice versa.
* A *strong exact* keyword hit bypasses the minimum-score threshold, so
  arbitrary strings like ``ERR_4021`` are found even though they embed poorly
  (their semantic similarity is near zero by construction).

Time filtering is applied as a **pre-filter**: the date range is passed down to
both retrievers, so ranking never sees rows outside the window. Time-phrase
extraction itself lives in :mod:`clipmind.search.timeparse`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from ..config import Config
from ..embeddings.encoder import Embedder
from ..storage.db import ClipRecord, Database
from .keyword import KeywordHit, KeywordSearcher
from .semantic import SemanticSearcher
from .timeparse import TimeQuery, parse_time_query

logger = logging.getLogger(__name__)

#: Coverage required for a keyword hit to count as "exact" (all tokens found).
EXACT_COVERAGE = 1.0

#: Minimum normalised keyword score for the exact-hit threshold bypass.
EXACT_MIN_KEYWORD_SCORE = 0.5


@dataclass(frozen=True, slots=True)
class HybridResult:
    """One fused search result with its score breakdown.

    Attributes:
        record: The stored entry.
        final_score: ``w_sem * semantic + w_kw * keyword``.
        semantic_score: Cosine similarity (``0.0`` when found only by keyword).
        keyword_score: Normalised, coverage-weighted BM25 (``0.0`` when found
            only by semantic).
        keyword_coverage: Fraction of query tokens present (``0.0`` if n/a).
        bm25: Raw SQLite BM25 value, or ``None`` for semantic-only hits.
        semantic_rank: 1-based rank from the semantic retriever, if any.
        keyword_rank: 1-based rank from the keyword retriever, if any.
    """

    record: ClipRecord
    final_score: float
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    keyword_coverage: float = 0.0
    bm25: Optional[float] = None
    semantic_rank: Optional[int] = None
    keyword_rank: Optional[int] = None

    @property
    def id(self) -> int:
        """Database id of the matched entry."""
        return self.record.id

    @property
    def content(self) -> str:
        """Full stored content."""
        return self.record.content

    @property
    def content_type(self) -> str:
        """Detected content type."""
        return self.record.content_type

    @property
    def created_at(self) -> datetime:
        """UTC capture timestamp."""
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

    @property
    def exact_keyword(self) -> bool:
        """True when every query token was found in the entry."""
        return self.keyword_coverage >= EXACT_COVERAGE


@dataclass(frozen=True, slots=True)
class HybridOutcome:
    """Everything a caller (CLI/UI) needs to explain a hybrid query.

    Attributes:
        query: The parsed time query (cleaned text + optional window).
        results: Ranked results.
        threshold: The effective minimum score.
        index_size: Number of embedded entries the semantic retriever considered.
        best_final: Best ``final_score`` seen before thresholding, if any.
        used_semantic: Whether the semantic retriever ran.
        used_keyword: Whether the keyword retriever ran.
        time_only: True when the query was just a time phrase, so the range was
            listed newest-first instead of ranked.
        notes: Human-readable explanations for ``--explain``.
    """

    query: TimeQuery
    results: list[HybridResult] = field(default_factory=list)
    threshold: float = 0.0
    index_size: int = 0
    best_final: Optional[float] = None
    used_semantic: bool = True
    used_keyword: bool = True
    time_only: bool = False
    notes: list[str] = field(default_factory=list)


class HybridSearcher:
    """Fuse semantic and keyword retrieval over one database."""

    def __init__(
        self,
        db: Database,
        *,
        embedder: Optional[Embedder] = None,
        config: Optional[Config] = None,
    ) -> None:
        """Create a searcher.

        Args:
            db: Database to search.
            embedder: Optional embedder shared with the semantic searcher.
            config: Optional configuration (weights, threshold, pool size).
        """
        self.db = db
        self.config = config or Config()
        self.semantic = SemanticSearcher(db, embedder=embedder, config=self.config)
        self.keyword = KeywordSearcher(db, config=self.config)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        *,
        content_type: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        min_score: Optional[float] = None,
        semantic_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None,
        semantic_only: bool = False,
        keyword_only: bool = False,
    ) -> HybridOutcome:
        """Rank stored entries for ``query`` within an optional time window.

        Args:
            query: Cleaned query text (time phrase already removed).
            top_k: Maximum results (defaults to ``config.search_top_k``).
            content_type: Optional content-type filter.
            since: Only consider entries created at or after this instant.
            until: Only consider entries created strictly before this instant.
            min_score: Minimum ``final_score``; defaults to ``config.min_score``.
            semantic_weight: Override ``config.semantic_weight``.
            keyword_weight: Override ``config.keyword_weight``.
            semantic_only: Disable keyword retrieval.
            keyword_only: Disable semantic retrieval.

        Returns:
            A :class:`HybridOutcome`.
        """
        top_k = top_k if top_k is not None else self.config.search_top_k
        threshold = self.config.min_score if min_score is None else min_score
        if semantic_only:
            w_sem, w_kw = 1.0, 0.0
        elif keyword_only:
            w_sem, w_kw = 0.0, 1.0
        else:
            w_sem = self.config.semantic_weight if semantic_weight is None else semantic_weight
            w_kw = self.config.keyword_weight if keyword_weight is None else keyword_weight

        notes: list[str] = []
        query = (query or "").strip()
        time_only = not query and (since is not None or until is not None)

        # A query that is only a time phrase: list the window newest-first.
        if time_only:
            records = self.db.list_recent(
                limit=top_k, content_type=content_type, since=since, until=until
            )
            results = [HybridResult(record=record, final_score=0.0) for record in records]
            notes.append("query was only a time phrase; listing the range newest first")
            return HybridOutcome(
                query=_as_time_placeholder(since, until),
                results=results,
                threshold=threshold,
                index_size=len(records),
                best_final=None,
                used_semantic=False,
                used_keyword=False,
                time_only=True,
                notes=notes,
            )

        if not query or top_k <= 0:
            return HybridOutcome(
                query=_as_time_placeholder(since, until),
                threshold=threshold,
                used_semantic=w_sem > 0,
                used_keyword=w_kw > 0,
                notes=notes,
            )

        pool = max(top_k, self.config.candidate_pool)

        # -- gather candidates (min_score=0 keeps both halves un-thresholded) --
        semantic_hits = []
        if w_sem > 0:
            semantic_hits = self.semantic.search(
                query,
                top_k=pool,
                content_type=content_type,
                min_score=-1.0,
                since=since,
                until=until,
            )
        index_size = self.semantic.last_index_size if w_sem > 0 else 0

        keyword_hits = []
        if w_kw > 0:
            keyword_hits = self.keyword.search(
                query, limit=pool, content_type=content_type, since=since, until=until
            )

        # -- merge by id -------------------------------------------------------
        merged: dict[int, dict[str, object]] = {}

        def slot(record: ClipRecord) -> dict[str, object]:
            return merged.setdefault(
                record.id,
                {"record": record, "sem": 0.0, "kw": 0.0, "cov": 0.0, "bm25": None,
                 "sem_rank": None, "kw_rank": None},
            )

        for rank, hit in enumerate(semantic_hits, start=1):
            entry = slot(hit.record)
            entry["sem"] = hit.score
            entry["sem_rank"] = rank
        for rank, hit in enumerate(keyword_hits, start=1):
            entry = slot(hit.record)
            entry["kw"] = hit.score * hit.coverage
            entry["cov"] = hit.coverage
            entry["bm25"] = hit.bm25
            entry["kw_rank"] = rank

        # -- fuse --------------------------------------------------------------
        scored: list[HybridResult] = []
        for entry in merged.values():
            sem = float(entry["sem"])
            kw = float(entry["kw"])
            final = w_sem * sem + w_kw * kw
            scored.append(
                HybridResult(
                    record=entry["record"],  # type: ignore[arg-type]
                    final_score=final,
                    semantic_score=sem,
                    keyword_score=kw,
                    keyword_coverage=float(entry["cov"]),
                    bm25=entry["bm25"],  # type: ignore[arg-type]
                    semantic_rank=entry["sem_rank"],  # type: ignore[arg-type]
                    keyword_rank=entry["kw_rank"],  # type: ignore[arg-type]
                )
            )

        scored.sort(
            key=lambda r: (r.final_score, r.created_at, r.id), reverse=True
        )
        best_final = scored[0].final_score if scored else None

        # Keep results that clear the threshold, plus strong exact keyword hits.
        kept: list[HybridResult] = []
        for result in scored:
            if result.final_score >= threshold:
                kept.append(result)
            elif result.exact_keyword and result.keyword_score >= EXACT_MIN_KEYWORD_SCORE:
                kept.append(result)
        if any(r.final_score < threshold for r in kept):
            notes.append(
                "one or more results were below the threshold but kept as exact "
                "keyword matches"
            )

        if semantic_hits and keyword_hits:
            notes.append(
                f"fused {len(semantic_hits)} semantic and {len(keyword_hits)} "
                f"keyword candidates (weights {w_sem:.2f}/{w_kw:.2f})"
            )

        return HybridOutcome(
            query=_as_time_placeholder(since, until),
            results=kept[:top_k],
            threshold=threshold,
            index_size=index_size,
            best_final=best_final,
            used_semantic=w_sem > 0,
            used_keyword=w_kw > 0,
            notes=notes,
        )


def _as_time_placeholder(
    since: Optional[datetime], until: Optional[datetime]
) -> TimeQuery:
    """Return a lightweight :class:`TimeQuery` carrying just the bounds.

    Used internally when the caller already stripped the time phrase but the
    searcher still wants to report the window on its outcome.
    """
    return TimeQuery(raw="", cleaned_query="", start=since, end=until)


def hybrid_search(
    db: Database,
    query: str,
    top_k: int = 5,
    *,
    embedder: Optional[Embedder] = None,
    config: Optional[Config] = None,
    content_type: Optional[str] = None,
    min_score: Optional[float] = None,
    semantic_weight: Optional[float] = None,
    keyword_weight: Optional[float] = None,
    semantic_only: bool = False,
    keyword_only: bool = False,
    now: Optional[datetime] = None,
) -> HybridOutcome:
    """Parse ``query`` for a time phrase, then run a hybrid search.

    This is the end-to-end entry point used by the CLI and the UI: it splits the
    time phrase off the query (see :func:`~clipmind.search.timeparse.parse_time_query`),
    applies the resulting window as a pre-filter, and fuses the two rankings.

    Args:
        db: Database to search.
        query: Raw user query, possibly containing a time phrase.
        top_k: Maximum results.
        embedder: Optional embedder (resolved from config when omitted).
        config: Optional configuration.
        content_type: Optional content-type filter.
        min_score: Minimum final score.
        semantic_weight: Override the semantic weight.
        keyword_weight: Override the keyword weight.
        semantic_only: Disable keyword retrieval.
        keyword_only: Disable semantic retrieval.
        now: Reference instant for time parsing (injectable for tests).

    Returns:
        A :class:`HybridOutcome` whose ``query`` is the fully parsed
        :class:`~clipmind.search.timeparse.TimeQuery`.
    """
    config = config or Config()
    time_query = parse_time_query(query, now=now, config=config)
    searcher = HybridSearcher(db, embedder=embedder, config=config)
    outcome = searcher.search(
        time_query.cleaned_query,
        top_k,
        content_type=content_type,
        since=time_query.start,
        until=time_query.end,
        min_score=min_score,
        semantic_weight=semantic_weight,
        keyword_weight=keyword_weight,
        semantic_only=semantic_only,
        keyword_only=keyword_only,
    )
    return HybridOutcome(
        query=time_query,
        results=outcome.results,
        threshold=outcome.threshold,
        index_size=outcome.index_size,
        best_final=outcome.best_final,
        used_semantic=outcome.used_semantic,
        used_keyword=outcome.used_keyword,
        time_only=outcome.time_only,
        notes=outcome.notes,
    )
