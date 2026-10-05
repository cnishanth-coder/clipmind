"""Tests for :mod:`clipmind.search.semantic` (fake embedder, no model load)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from clipmind.config import Config
from clipmind.search.semantic import SearchResult, SemanticSearcher, search_semantic
from clipmind.storage.db import Database
from fakes import FakeEmbedder, embed_into_db


def _config(**kwargs) -> Config:
    return Config(db_path=":memory:", **kwargs)


@pytest.fixture()
def db() -> Database:
    database = Database(db_path=":memory:")
    try:
        yield database
    finally:
        database.close()


@pytest.fixture()
def searcher_factory(db):
    """Return a helper building a searcher around the shared fake embedder."""

    def build(**config_kwargs):
        config = _config(**config_kwargs)
        embedder = FakeEmbedder()
        return SemanticSearcher(db, embedder=embedder, config=config), embedder

    return build


DOCS = [
    ("SELECT id FROM orders WHERE status = open;", "code"),
    ("https://react.dev/reference/react/hooks", "url"),
    ("meeting at 5pm with Rahul", "text"),
]


def test_returns_search_results_with_required_fields(db: Database, searcher_factory) -> None:
    embed_into_db(db, DOCS)
    searcher, _embedder = searcher_factory()

    results = searcher.search("orders status")

    assert results and isinstance(results[0], SearchResult)
    top = results[0]
    assert isinstance(top.id, int)
    assert isinstance(top.score, float)
    assert isinstance(top.content, str)
    assert isinstance(top.content_type, str)
    assert isinstance(top.created_at, datetime)
    assert top.content_type in {"code", "url", "text"}
    assert 0.0 <= top.score <= 1.0


def test_ranking_is_by_similarity(db: Database, searcher_factory) -> None:
    embed_into_db(db, DOCS)
    searcher, _embedder = searcher_factory()

    results = searcher.search("react hooks reference")

    assert results[0].content == "https://react.dev/reference/react/hooks"
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_short_query_ranks_its_best_document_first(db: Database, searcher_factory) -> None:
    embed_into_db(db, [("alpha beta gamma", "text"), ("delta epsilon zeta", "text")])
    searcher, _embedder = searcher_factory()

    results = searcher.search("epsilon", min_score=0.0)
    assert results[0].content == "delta epsilon zeta"


def test_min_score_filters_junk(db: Database, searcher_factory) -> None:
    embed_into_db(db, DOCS)
    searcher, _embedder = searcher_factory()

    permissive = searcher.search("order status", min_score=0.0)
    strict = searcher.search("order status", min_score=0.99)

    assert len(permissive) >= 1
    assert strict == []
    # Diagnostics explain why the result set was empty.
    assert searcher.last_best_score is not None
    assert searcher.last_best_score < 0.99
    assert searcher.last_threshold == 0.99


def test_default_min_score_comes_from_config(db: Database, searcher_factory) -> None:
    embed_into_db(db, DOCS)
    searcher, _embedder = searcher_factory(min_score=0.99)
    assert searcher.search("orders") == []
    assert searcher.last_threshold == 0.99


def test_top_k_limits_results(db: Database, searcher_factory) -> None:
    embed_into_db(db, DOCS)
    searcher, _embedder = searcher_factory()
    assert len(searcher.search("order react meeting", top_k=1, min_score=0.0)) == 1
    assert len(searcher.search("order react meeting", top_k=3, min_score=0.0)) == 3


def test_top_k_defaults_to_config(db: Database, searcher_factory) -> None:
    embed_into_db(db, DOCS)
    searcher, _embedder = searcher_factory(search_top_k=2)
    assert len(searcher.search("order react meeting", min_score=0.0)) == 2


def test_content_type_filter_restricts_candidates(db: Database, searcher_factory) -> None:
    embed_into_db(db, DOCS)
    searcher, _embedder = searcher_factory()

    code_only = searcher.search("order react meeting", content_type="code", min_score=0.0)
    assert [r.content_type for r in code_only] == ["code"]
    assert searcher.last_index_size == 1

    url_only = searcher.search("order react meeting", content_type="url", min_score=0.0)
    assert [r.content_type for r in url_only] == ["url"]


def test_empty_database_returns_nothing_and_never_loads_the_model(
    db: Database, searcher_factory
) -> None:
    searcher, embedder = searcher_factory()

    assert searcher.search("anything at all") == []
    assert searcher.last_index_size == 0
    assert searcher.last_best_score is None
    assert embedder.call_count == 0  # no model work for an empty index


def test_unindexed_rows_are_not_searchable(db: Database, searcher_factory) -> None:
    db.insert_clip("never embedded because reindex has not run", content_type="text")
    searcher, embedder = searcher_factory()

    assert searcher.search("never embedded") == []
    assert searcher.last_index_size == 0
    assert embedder.call_count == 0


def test_blank_query_returns_nothing_without_embedding(db: Database, searcher_factory) -> None:
    embed_into_db(db, DOCS)
    searcher, embedder = searcher_factory()

    assert searcher.search("   ") == []
    assert searcher.search("") == []
    assert searcher.search("anything", top_k=0) == []
    assert embedder.call_count == 0


def test_index_is_cached_and_invalidated_by_new_rows(db: Database, searcher_factory) -> None:
    embed_into_db(db, DOCS)
    searcher, _embedder = searcher_factory()

    searcher.search("orders", min_score=0.0)
    first_size = searcher.last_index_size
    searcher.search("orders", min_score=0.0)
    assert searcher.last_index_size == first_size

    embed_into_db(db, [("brand new searchable entry", "text")])
    searcher.search("searchable", min_score=0.0)
    assert searcher.last_index_size == first_size + 1


def test_search_semantic_convenience_function(db: Database) -> None:
    embed_into_db(db, DOCS)
    results = search_semantic(
        db, "react hooks", top_k=2, embedder=FakeEmbedder(), config=_config(), min_score=0.0
    )
    assert results[0].content == "https://react.dev/reference/react/hooks"


def test_index_size_helper(db: Database, searcher_factory) -> None:
    embed_into_db(db, DOCS)
    searcher, _embedder = searcher_factory()
    assert searcher.index_size() == 3
    assert searcher.index_size("code") == 1


def test_relative_time_and_preview_helpers(db: Database, searcher_factory) -> None:
    embed_into_db(db, DOCS)
    searcher, _embedder = searcher_factory()
    result = searcher.search("react hooks", min_score=0.0)[0]

    assert isinstance(result.relative_time(datetime.now(timezone.utc)), str)
    # preview() collapses whitespace and ellipsises to at most max_len chars.
    assert result.preview(10) == "https://r\u2026"
    assert result.preview(200) == "https://react.dev/reference/react/hooks"
