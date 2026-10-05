"""Keyword search over the FTS5 index, with BM25 ranking.

This is the exact-token half of the hybrid search. It exists because semantic
similarity is deliberately bad at remembering arbitrary strings: a query like
``"ERR_4021"`` or a URL has no "meaning" to embed, but FTS5 finds it instantly.

The critical job here is **sanitising**: raw user text is never handed to the
FTS5 ``MATCH`` parser. A query such as ``SELECT "id" FROM orders;`` or
``C++ (v2)`` would otherwise be read as FTS5 syntax and raise
``OperationalError: fts5: syntax error``. Instead, the query is reduced to
alphanumeric tokens and each is wrapped in double quotes, so the worst case is
an empty result, never an exception.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ..config import Config
from ..storage.db import ClipRecord, Database

#: Maximum number of tokens taken from a query, to bound MATCH cost.
MAX_QUERY_TOKENS = 32

#: Characters that survive sanitising. Everything else is a separator, which
#: mirrors SQLite's ``unicode61`` tokenizer closely enough to be quoted safely.
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class KeywordHit:
    """One FTS5 match.

    Attributes:
        record: The stored entry that matched.
        bm25: Raw ``bm25()`` value from SQLite (more negative is better).
        score: BM25 normalised to ``0..1`` across this query's result set.
        coverage: Fraction of query tokens present in the entry's text
            (``0..1``); ``1.0`` means every token was found.
    """

    record: ClipRecord
    bm25: float
    score: float
    coverage: float
    matched_terms: int
    total_terms: int

    @property
    def id(self) -> int:
        """Database id of the matched entry."""
        return self.record.id


def tokenize_query(query: str) -> list[str]:
    """Return the deduplicated, lowercased alphanumeric tokens of ``query``.

    The tokenisation matches how :func:`sanitize_fts_query` builds the MATCH
    expression, so the tokens can also be used to measure how much of the query
    an entry actually contains.
    """
    seen: set[str] = set()
    tokens: list[str] = []
    for token in _TOKEN_RE.findall(query or ""):
        lowered = token.lower()
        if lowered not in seen:
            seen.add(lowered)
            tokens.append(lowered)
        if len(tokens) >= MAX_QUERY_TOKENS:
            break
    return tokens


def sanitize_fts_query(query: str) -> str:
    """Translate arbitrary user text into a safe FTS5 ``MATCH`` expression.

    Args:
        query: Raw user query, possibly containing quotes, SQL, URLs or
            operator characters.

    Returns:
        An OR-joined expression of quoted tokens (``'"err" OR "4021"'``), or an
        empty string when the query has no usable tokens. An empty string must
        be treated as "no results" by the caller; it is never passed to SQLite.
    """
    tokens = tokenize_query(query)
    if not tokens:
        return ""
    return " OR ".join(f'"{token}"' for token in tokens)


def _normalise_bm25(raw_scores: list[float]) -> list[float]:
    """Map BM25 scores (more negative is better) onto ``0..1`` per query.

    The best hit becomes ``1.0`` and the worst ``0.0``. When every hit has the
    same raw score (a single result, or a tie) all are reported as ``1.0``:
    relative to nothing else, that hit is the best available.
    """
    if not raw_scores:
        return []
    best = min(raw_scores)
    worst = max(raw_scores)
    span = worst - best
    if span <= 0:
        return [1.0 for _ in raw_scores]
    return [(worst - raw) / span for raw in raw_scores]


class KeywordSearcher:
    """Run sanitised FTS5 searches and normalise their BM25 scores."""

    def __init__(self, db: Database, *, config: Optional[Config] = None) -> None:
        """Create a searcher.

        Args:
            db: Database holding the FTS5 index.
            config: Optional configuration (currently unused, kept for symmetry
                with :class:`~clipmind.search.semantic.SemanticSearcher`).
        """
        self.db = db
        self.config = config or Config()

    def search(
        self,
        query: str,
        limit: int = 20,
        *,
        content_type: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> list[KeywordHit]:
        """Find entries whose text contains the query's tokens.

        Args:
            query: Raw user query (sanitised internally).
            limit: Maximum number of hits.
            content_type: Optional content-type filter.
            since: Only include entries created at or after this instant.
            until: Only include entries created strictly before this instant.

        Returns:
            Hits ordered by descending normalised score. Empty when the query
            has no usable tokens.
        """
        if limit <= 0:
            return []
        expression = sanitize_fts_query(query)
        if not expression:
            return []

        rows = self.db.search_keywords(
            expression, limit, content_type=content_type, since=since, until=until
        )
        if not rows:
            return []

        tokens = tokenize_query(query)
        raw_scores = [raw for _record, raw in rows]
        normalised = _normalise_bm25(raw_scores)

        hits: list[KeywordHit] = []
        for (record, raw), score in zip(rows, normalised):
            document_tokens = set(_TOKEN_RE.findall(record.content.lower()))
            matched = sum(1 for token in tokens if token in document_tokens)
            coverage = matched / len(tokens) if tokens else 0.0
            hits.append(
                KeywordHit(
                    record=record,
                    bm25=raw,
                    score=score,
                    coverage=coverage,
                    matched_terms=matched,
                    total_terms=len(tokens),
                )
            )
        return hits


def keyword_search(
    db: Database,
    query: str,
    limit: int = 20,
    *,
    config: Optional[Config] = None,
    content_type: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> list[KeywordHit]:
    """Convenience wrapper: keyword search in one call."""
    searcher = KeywordSearcher(db, config=config)
    return searcher.search(
        query, limit, content_type=content_type, since=since, until=until
    )
