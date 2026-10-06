"""Tests for :mod:`clipmind.ui.search_backend` (offline, fake embedder)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from clipmind.config import Config
from clipmind.storage.db import Database
from clipmind.ui.search_backend import HybridSearchBackend, _chip_label
from clipmind.ui.view_model import SearchRequest
from fakes import FakeEmbedder, embed_into_db

TZ = timezone(timedelta(hours=-4))
NOW = datetime(2026, 10, 6, 12, 0, tzinfo=TZ)


@pytest.fixture()
def db() -> Database:
    database = Database(db_path=":memory:")
    try:
        yield database
    finally:
        database.close()


def _request(query: str, **kwargs) -> SearchRequest:
    return SearchRequest(query=query, content_type=None, generation=1, top_k=10, **kwargs)


def _config() -> Config:
    return Config(db_path=":memory:")


def test_empty_query_lists_the_most_recent(db: Database) -> None:
    embed_into_db(
        db,
        [("SELECT id FROM orders;", "code"), ("https://react.dev/learn", "url")],
    )
    backend = HybridSearchBackend(db, embedder=FakeEmbedder(), config=_config())
    outcome = backend.run_search(_request(""))
    assert [item.content_type for item in outcome.items] == ["url", "code"]
    assert outcome.error is None


def test_empty_query_honours_the_type_filter(db: Database) -> None:
    embed_into_db(
        db,
        [("SELECT id FROM orders;", "code"), ("https://react.dev/learn", "url")],
    )
    backend = HybridSearchBackend(db, embedder=FakeEmbedder(), config=_config())
    request = SearchRequest(query="", content_type="code", generation=1, top_k=10)
    outcome = backend.run_search(request)
    assert [item.content_type for item in outcome.items] == ["code"]


def test_ranked_search_returns_items(db: Database) -> None:
    embed_into_db(
        db,
        [("react hooks article", "text"), ("gardening tips", "text")],
    )
    backend = HybridSearchBackend(db, embedder=FakeEmbedder(), config=_config())
    outcome = backend.run_search(_request("react hooks"))
    assert outcome.error is None
    assert any("react" in item.content for item in outcome.items)


def test_time_phrase_becomes_a_chip_label(db: Database) -> None:
    embed_into_db(db, [("react hooks yesterday", "text")])
    backend = HybridSearchBackend(db, embedder=FakeEmbedder(), config=_config())
    outcome = backend.run_search(_request("react hooks yesterday"))
    assert outcome.time_label == "Yesterday"


def test_ignore_time_treats_the_phrase_as_text(db: Database) -> None:
    embed_into_db(db, [("meeting notes", "text")])
    backend = HybridSearchBackend(db, embedder=FakeEmbedder(), config=_config())
    outcome = backend.run_search(_request("meeting notes yesterday", ignore_time=True))
    assert outcome.time_label is None
    assert outcome.error is None


def test_locked_database_returns_an_error_not_an_exception(db: Database) -> None:
    backend = HybridSearchBackend(db, embedder=FakeEmbedder(), config=_config())
    db.close()  # every subsequent query raises sqlite3.ProgrammingError
    outcome = backend.run_search(_request("anything"))
    assert outcome.error is not None
    assert outcome.items == ()


def test_delete_and_restore_round_trip(db: Database) -> None:
    first = db.insert_clip("SELECT 1;", content_type="code", created_at=NOW)
    assert first.inserted and first.record is not None
    backend = HybridSearchBackend(db, embedder=FakeEmbedder(), config=_config())
    rows = backend.run_search(_request("")).items
    item = rows[0]

    assert backend.delete_item(item.id) is True
    assert db.count() == 0
    assert backend.restore_item(item) is True
    rows = db.list_recent(limit=1)
    assert len(rows) == 1
    assert rows[0].content == "SELECT 1;"
    assert rows[0].content_hash == item.content_hash
    assert rows[0].created_at == NOW


@pytest.mark.parametrize(
    "phrase, expected",
    [
        ("yesterday", "Yesterday"),
        ("last monday", "Last Monday"),
        ("2 hours ago", "2 hours ago"),
        (None, None),
        ("   ", None),
    ],
)
def test_chip_label(phrase, expected) -> None:
    assert _chip_label(phrase) == expected
