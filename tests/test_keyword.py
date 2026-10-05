"""Tests for :mod:`clipmind.search.keyword` (FTS5 + BM25, no model).

The important guarantees are: arbitrary user text (SQL, URLs, quotes, operators)
never reaches the FTS5 parser unquoted, and BM25 is normalised to ``0..1`` per
query so it can be fused with a cosine similarity.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from clipmind.search.keyword import (
    MAX_QUERY_TOKENS,
    KeywordHit,
    KeywordSearcher,
    keyword_search,
    sanitize_fts_query,
    tokenize_query,
)
from clipmind.storage.db import Database

UTC = timezone.utc


@pytest.fixture()
def db() -> Database:
    database = Database(db_path=":memory:")
    try:
        yield database
    finally:
        database.close()


DOCS = [
    ("SELECT id FROM orders WHERE status = open", "code"),
    ("ERR_4021 database connection timeout", "code"),
    ("https://react.dev/reference/react/hooks", "url"),
    ("meeting notes about quarterly budget", "text"),
]


def _seed(db: Database, docs=DOCS) -> None:
    for content, content_type in docs:
        assert db.insert_clip(content, content_type=content_type).inserted


# ------------------------------------------------------------- sanitising


def test_sanitize_quotes_every_token_and_or_joins() -> None:
    assert sanitize_fts_query("ERR_4021") == '"err" OR "4021"'


def test_sanitize_sql_never_leaves_operators_bare() -> None:
    assert (
        sanitize_fts_query('SELECT "id" FROM orders;')
        == '"select" OR "id" OR "from" OR "orders"'
    )


def test_sanitize_url() -> None:
    assert (
        sanitize_fts_query("https://react.dev/reference/react/hooks")
        == '"https" OR "react" OR "dev" OR "reference" OR "hooks"'
    )


def test_sanitize_quotes_and_operators() -> None:
    assert sanitize_fts_query('C++ (v2) "quoted"') == '"c" OR "v2" OR "quoted"'


@pytest.mark.parametrize(
    "query",
    ["", "   ", "!!!", "()[]{}", "\"'\\", "***", "-- ; --"],
)
def test_sanitize_returns_empty_for_unusable_queries(query: str) -> None:
    assert sanitize_fts_query(query) == ""


def test_sanitize_caps_the_number_of_tokens() -> None:
    query = " ".join(f"token{i}" for i in range(MAX_QUERY_TOKENS + 20))
    assert len(tokenize_query(query)) == MAX_QUERY_TOKENS
    assert sanitize_fts_query(query).count(" OR ") == MAX_QUERY_TOKENS - 1


def test_tokenize_deduplicates_preserving_order() -> None:
    assert tokenize_query("React react REACT hooks") == ["react", "hooks"]


# --------------------------------------------------------------- searching


def test_exact_token_is_found_with_full_coverage(db: Database) -> None:
    _seed(db)
    hits = KeywordSearcher(db).search("ERR_4021")

    assert len(hits) == 1
    hit = hits[0]
    assert isinstance(hit, KeywordHit)
    assert "ERR_4021" in hit.record.content
    assert hit.score == 1.0
    assert hit.coverage == 1.0
    assert hit.matched_terms == 2 and hit.total_terms == 2
    assert hit.bm25 < 0  # SQLite bm25: more negative is better


def test_dangerous_queries_do_not_raise(db: Database) -> None:
    _seed(db)
    searcher = KeywordSearcher(db)
    # None of these may raise an FTS5 syntax error.
    for query in (
        'SELECT * FROM orders; DROP TABLE items; --',
        "https://example.com/a?b=1&c=2#frag",
        '"unbalanced quote',
        "C++ AND (v2 OR v3) NOT",
        "{}[]()^~*",
    ):
        assert isinstance(searcher.search(query), list)


def test_sql_snippet_returns_the_matching_entry(db: Database) -> None:
    _seed(db)
    hits = KeywordSearcher(db).search('SELECT "id" FROM orders;')
    assert hits and "orders" in hits[0].record.content


def test_partial_coverage_is_reported(db: Database) -> None:
    _seed(db)
    hits = KeywordSearcher(db).search("ERR_4021 react")
    by_content = {hit.record.content: hit for hit in hits}

    err = next(hit for content, hit in by_content.items() if "ERR_4021" in content)
    url = next(hit for content, hit in by_content.items() if "react.dev" in content)
    # Query tokens are err, 4021 and react: the ERR entry has the first two.
    assert err.coverage == pytest.approx(2 / 3) and err.matched_terms == 2
    assert url.coverage == pytest.approx(1 / 3) and url.matched_terms == 1


def test_scores_are_normalised_to_zero_one(db: Database) -> None:
    _seed(db)
    hits = KeywordSearcher(db).search("orders budget")
    scores = [hit.score for hit in hits]
    assert len(hits) == 2
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert max(scores) == 1.0
    assert scores == sorted(scores, reverse=True)


def test_single_hit_scores_as_one(db: Database) -> None:
    _seed(db)
    hits = KeywordSearcher(db).search("budget")
    assert len(hits) == 1
    assert hits[0].score == 1.0


def test_empty_result_for_unknown_tokens(db: Database) -> None:
    _seed(db)
    assert KeywordSearcher(db).search("nonexistent zzz") == []


def test_content_type_filter(db: Database) -> None:
    _seed(db)
    hits = KeywordSearcher(db).search("orders react meeting", content_type="code")
    assert [hit.record.content_type for hit in hits] == ["code"]
    assert all("meeting" not in hit.record.content for hit in hits)


def test_limit_is_respected(db: Database) -> None:
    _seed(db)
    assert len(KeywordSearcher(db).search("orders ERR_4021 react", limit=2)) == 2


def test_limit_zero_returns_nothing(db: Database) -> None:
    _seed(db)
    assert KeywordSearcher(db).search("orders", limit=0) == []


# ----------------------------------------------------------- date filtering


def test_date_range_filters_before_ranking(db: Database) -> None:
    base = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
    assert db.insert_clip(
        "database backup finished", content_type="text", created_at=base
    ).inserted
    assert db.insert_clip(
        "database migration planned", content_type="text",
        created_at=base + timedelta(days=1),
    ).inserted

    searcher = KeywordSearcher(db)
    all_hits = searcher.search("database")
    assert len(all_hits) == 2

    only_day_two = searcher.search(
        "database", since=base + timedelta(days=1), until=base + timedelta(days=2)
    )
    assert [hit.record.content for hit in only_day_two] == ["database migration planned"]

    excluded = searcher.search("database", since=base + timedelta(days=10))
    assert excluded == []


def test_keyword_search_wrapper(db: Database) -> None:
    _seed(db)
    hits = keyword_search(db, "ERR_4021")
    assert hits and hits[0].coverage == 1.0
