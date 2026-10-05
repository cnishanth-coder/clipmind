"""Tests for :mod:`clipmind.search.hybrid` (fake embedder, no model).

These cover the Phase 3 fusion contract: weighted sum of the two scores, merge
by id, one-sided hits still scored, the strong-exact-keyword threshold bypass,
and - crucially - that a parsed date range is applied as a *pre-filter* before
ranking.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from clipmind.config import Config
from clipmind.embeddings.codec import encode_vector
from clipmind.search.hybrid import (
    EXACT_MIN_KEYWORD_SCORE,
    HybridOutcome,
    HybridResult,
    HybridSearcher,
    hybrid_search,
)
from clipmind.storage.db import Database
from fakes import FakeEmbedder

TZ = timezone(timedelta(hours=-4))
NOW = datetime(2026, 10, 7, 14, 30, tzinfo=TZ)


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
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


def _add(db: Database, embedder: FakeEmbedder, content: str, *, content_type="text",
         created_at=None) -> int:
    """Insert ``content`` and store its embedding; return the row id."""
    result = db.insert_clip(
        content, content_type=content_type, created_at=created_at, dedup=False
    )
    assert result.inserted and result.record is not None
    db.set_embedding(result.record.id, encode_vector(embedder.embed_one(content)))
    return result.record.id


def _add_unindexed(db: Database, content: str, *, content_type="text",
                   created_at=None) -> int:
    """Insert ``content`` but leave it unembedded (keyword-only candidate)."""
    result = db.insert_clip(
        content, content_type=content_type, created_at=created_at, dedup=False
    )
    assert result.inserted and result.record is not None
    return result.record.id


def _by_content(outcome: HybridOutcome) -> dict[str, HybridResult]:
    return {result.content: result for result in outcome.results}


# ------------------------------------------------------------------- weighting


def test_final_score_is_the_weighted_sum(db, embedder) -> None:
    _add(db, embedder, "react hooks are useful for stateful logic")
    searcher = HybridSearcher(db, embedder=embedder, config=_config())

    outcome = searcher.search("react hooks", min_score=0.0)

    top = outcome.results[0]
    assert top.semantic_score > 0.0 and top.keyword_score > 0.0
    assert top.final_score == pytest.approx(
        0.7 * top.semantic_score + 0.3 * top.keyword_score
    )


def test_weights_are_configurable(db, embedder) -> None:
    _add(db, embedder, "react hooks are useful for stateful logic")
    config = _config(semantic_weight=0.5, keyword_weight=0.5)
    searcher = HybridSearcher(db, embedder=embedder, config=config)

    top = searcher.search("react hooks", min_score=0.0).results[0]
    assert top.final_score == pytest.approx(
        0.5 * top.semantic_score + 0.5 * top.keyword_score
    )


def test_explicit_weight_overrides_beats_config(db, embedder) -> None:
    _add(db, embedder, "react hooks are useful for stateful logic")
    searcher = HybridSearcher(db, embedder=embedder, config=_config())

    top = searcher.search(
        "react hooks", min_score=0.0, semantic_weight=0.0, keyword_weight=1.0
    ).results[0]
    assert top.final_score == pytest.approx(top.keyword_score)


# ----------------------------------------------------------------------- merge


def test_results_are_merged_by_id(db, embedder) -> None:
    _add(db, embedder, "react hooks are useful for stateful logic")
    searcher = HybridSearcher(db, embedder=embedder, config=_config())

    outcome = searcher.search("react hooks", min_score=0.0)
    ids = [result.id for result in outcome.results]
    assert len(ids) == len(set(ids))
    # The single entry matched both retrievers, but appears once.
    assert outcome.results[0].semantic_rank == 1
    assert outcome.results[0].keyword_rank == 1


def test_a_keyword_only_hit_is_still_scored(db, embedder) -> None:
    _add(db, embedder, "react hooks are useful for stateful logic")
    _add_unindexed(db, "ERR_4021 database connection timeout")
    searcher = HybridSearcher(db, embedder=embedder, config=_config())

    outcome = searcher.search("ERR_4021 connection", min_score=0.0)
    err = _by_content(outcome)["ERR_4021 database connection timeout"]

    assert err.semantic_rank is None
    assert err.semantic_score == 0.0
    assert err.keyword_score > 0.0
    assert err.final_score == pytest.approx(0.3 * err.keyword_score)


def test_a_semantic_only_hit_is_still_scored(db, embedder) -> None:
    _add(db, embedder, "react hooks are useful for stateful logic")
    _add_unindexed(db, "ERR_4021 database connection timeout")
    searcher = HybridSearcher(db, embedder=embedder, config=_config())

    outcome = searcher.search("ERR_4021 connection", min_score=0.0)
    react = _by_content(outcome)["react hooks are useful for stateful logic"]

    assert react.keyword_rank is None
    assert react.keyword_score == 0.0
    assert react.semantic_score > 0.0
    assert react.final_score == pytest.approx(0.7 * react.semantic_score)


# ------------------------------------------------------ min-score / exact hits


def test_strong_exact_keyword_bypasses_the_threshold(db, embedder) -> None:
    _add_unindexed(db, "ERR_4021 database connection timeout")
    searcher = HybridSearcher(db, embedder=embedder, config=_config())

    outcome = searcher.search("ERR_4021", min_score=0.9)

    assert outcome.results
    top = outcome.results[0]
    assert top.content == "ERR_4021 database connection timeout"
    assert top.exact_keyword
    assert top.keyword_score >= EXACT_MIN_KEYWORD_SCORE
    assert top.final_score < 0.9
    assert any("exact keyword" in note for note in outcome.notes)


def test_partial_keyword_hit_is_not_rescued(db, embedder) -> None:
    _add_unindexed(db, "ERR_4021")
    searcher = HybridSearcher(db, embedder=embedder, config=_config())

    # Only one of three query tokens is present, so this is not an exact hit
    # and it must not sneak under the threshold.
    outcome = searcher.search("ERR_4021 connection failed", min_score=0.9)
    assert outcome.results == []


def test_threshold_filters_weak_but_keeps_best(db, embedder) -> None:
    _add(db, embedder, "react hooks are useful for stateful logic")
    searcher = HybridSearcher(db, embedder=embedder, config=_config())

    # The odd token keeps keyword coverage below 1.0, so the exact-hit bypass
    # does not apply and a very high threshold really does drop the result.
    assert searcher.search("react hooks zzznope", min_score=0.99).results == []
    assert searcher.search("react hooks zzznope", min_score=0.0).results


# --------------------------------------------------------------- single half


def test_semantic_only_ignores_keyword(db, embedder) -> None:
    _add(db, embedder, "react hooks are useful for stateful logic")
    searcher = HybridSearcher(db, embedder=embedder, config=_config())

    outcome = searcher.search("react hooks", min_score=0.0, semantic_only=True)
    assert outcome.used_semantic and not outcome.used_keyword
    assert outcome.results
    for result in outcome.results:
        assert result.keyword_score == 0.0
        assert result.final_score == pytest.approx(result.semantic_score)


def test_keyword_only_ignores_semantics(db, embedder) -> None:
    _add_unindexed(db, "ERR_4021 database connection timeout")
    searcher = HybridSearcher(db, embedder=embedder, config=_config())

    outcome = searcher.search("ERR_4021", min_score=0.0, keyword_only=True)
    assert outcome.used_keyword and not outcome.used_semantic
    assert outcome.results
    for result in outcome.results:
        assert result.semantic_score == 0.0
        assert result.final_score == pytest.approx(result.keyword_score)


# ----------------------------------------------------------------- date filter


def test_date_range_is_a_pre_filter(db, embedder) -> None:
    _add(
        db, embedder, "react hooks article about stateful logic",
        created_at=datetime(2026, 10, 6, 10, 0, tzinfo=TZ),  # yesterday
    )
    _add(
        db, embedder, "react hooks article about older stateful logic",
        created_at=datetime(2026, 10, 1, 10, 0, tzinfo=TZ),  # last week
    )
    config = _config()

    outcome = hybrid_search(
        db, "react hooks article yesterday", now=NOW, embedder=embedder, config=config
    )

    assert outcome.query.cleaned_query == "react hooks article"
    assert outcome.query.has_range
    assert len(outcome.results) == 1
    assert outcome.results[0].content == "react hooks article about stateful logic"


def test_hybrid_search_returns_the_parsed_time_query(db, embedder) -> None:
    _add(db, embedder, "react hooks article", created_at=datetime(2026, 10, 6, 10, tzinfo=TZ))
    outcome = hybrid_search(
        db, "react hooks yesterday", now=NOW, embedder=embedder, config=_config()
    )
    assert outcome.query.raw == "react hooks yesterday"
    assert outcome.query.matched_phrase == "yesterday"


def test_ambiguous_phrase_runs_a_normal_search(db, embedder) -> None:
    _add(db, embedder, "monday.com link to the project board")
    outcome = hybrid_search(
        db, "monday.com link", now=NOW, embedder=embedder, config=_config(), min_score=0.0
    )
    assert outcome.query.ambiguous and not outcome.query.has_range
    assert not outcome.time_only
    assert _by_content(outcome)["monday.com link to the project board"].keyword_coverage == 1.0


# ------------------------------------------------------------------ time only


def test_time_only_query_lists_the_range_newest_first(db, embedder) -> None:
    _add(
        db, embedder, "older note", created_at=datetime(2026, 10, 6, 8, 0, tzinfo=TZ),
    )
    _add(
        db, embedder, "newer note", created_at=datetime(2026, 10, 6, 20, 0, tzinfo=TZ),
    )
    _add(
        db, embedder, "today note", created_at=datetime(2026, 10, 7, 9, 0, tzinfo=TZ),
    )

    outcome = hybrid_search(
        db, "yesterday", top_k=10, now=NOW, embedder=embedder, config=_config()
    )

    assert outcome.time_only
    assert not outcome.used_semantic and not outcome.used_keyword
    assert outcome.query.cleaned_query == ""
    assert [result.content for result in outcome.results] == ["newer note", "older note"]
    assert any("time phrase" in note for note in outcome.notes)


def test_empty_cleaned_query_without_a_range_returns_nothing(db, embedder) -> None:
    _add(db, embedder, "anything")
    outcome = hybrid_search(
        db, "   ", now=NOW, embedder=embedder, config=_config()
    )
    assert outcome.results == []
    assert not outcome.time_only


# --------------------------------------------------------------- other filters


def test_content_type_filter(db, embedder) -> None:
    _add(db, embedder, "react hooks article", content_type="url")
    _add(db, embedder, "react hooks in a code sample", content_type="code")

    outcome = hybrid_search(
        db, "react hooks", now=NOW, embedder=embedder, config=_config(),
        content_type="url", min_score=0.0,
    )
    assert outcome.results
    assert all(result.content_type == "url" for result in outcome.results)


def test_result_helper_properties(db, embedder) -> None:
    _add(db, embedder, "react hooks article", created_at=NOW - timedelta(minutes=5))
    outcome = hybrid_search(
        db, "react hooks", now=NOW, embedder=embedder, config=_config(), min_score=0.0
    )
    result = outcome.results[0]
    assert isinstance(result, HybridResult)
    assert result.content == "react hooks article"
    assert result.created_at.tzinfo is not None
    assert result.preview(5) == "reac\u2026"
    assert "ago" in result.relative_time(NOW)
