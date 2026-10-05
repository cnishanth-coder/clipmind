"""Tests for :mod:`clipmind.storage.db`."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from clipmind.config import Config
from clipmind.storage.db import ClipRecord, Database, hash_content


@pytest.fixture()
def db() -> Database:
    database = Database(db_path=":memory:")
    try:
        yield database
    finally:
        database.close()


def test_schema_creates_fts_table(db: Database) -> None:
    tables = {
        row["name"]
        for row in db._conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
    }
    assert {"items", "items_fts", "meta"} <= tables


def test_insert_and_read_back(db: Database) -> None:
    result = db.insert_clip("hello world", content_type="text", source_app="Terminal")
    assert result.inserted is True
    assert result.record is not None
    assert result.record.content == "hello world"
    assert result.record.source_app == "Terminal"
    assert result.record.content_hash == hash_content("hello world")
    assert db.count() == 1


def test_exact_duplicates_are_skipped(db: Database) -> None:
    first = db.insert_clip("same text")
    second = db.insert_clip("same text")
    assert first.inserted and first.record is not None
    assert second.inserted is False
    assert second.reason == "duplicate"
    assert second.duplicate_of == first.record.id
    assert db.count() == 1


def test_unique_index_protects_even_when_dedup_lookup_is_disabled(db: Database) -> None:
    first = db.insert_clip("same text")
    second = db.insert_clip("same text", dedup=False)
    assert first.inserted
    assert second.inserted is False
    assert second.reason == "duplicate"
    assert db.count() == 1


def test_empty_content_is_not_stored_by_default(db: Database) -> None:
    result = db.insert_clip("")
    assert result.inserted is False
    assert result.reason == "empty"
    assert db.count() == 0


def test_oversize_truncate_and_skip(tmp_path) -> None:
    truncating = Database(db_path=tmp_path / "t.db", config=Config(max_chars=5))
    result = truncating.insert_clip("abcdefghij")
    assert result.inserted is True
    assert result.record is not None
    assert result.record.content == "abcde"
    assert result.record.content_hash == hash_content("abcde")
    truncating.close()

    skipping = Database(
        db_path=tmp_path / "s.db",
        config=Config(max_chars=5, oversize_policy="skip"),
    )
    skipped = skipping.insert_clip("abcdefghij")
    assert skipped.inserted is False
    assert skipped.reason == "too_long"
    assert skipping.count() == 0
    skipping.close()


def test_list_recent_newest_first_and_type_filter(db: Database) -> None:
    base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    db.insert_clip("older", content_type="text", created_at=base)
    db.insert_clip("newer", content_type="code", created_at=base + timedelta(minutes=1))

    records = db.list_recent()
    assert [r.content for r in records] == ["newer", "older"]
    assert [r.content for r in db.list_recent(content_type="code")] == ["newer"]
    assert [r.content for r in db.list_recent(order="asc")] == ["older", "newer"]


def test_list_recent_since_until_filter(db: Database) -> None:
    base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    db.insert_clip("a", created_at=base)
    db.insert_clip("b", created_at=base + timedelta(hours=1))
    db.insert_clip("c", created_at=base + timedelta(hours=2))

    since_results = db.list_recent(since=base + timedelta(hours=1))
    assert [r.content for r in since_results] == ["c", "b"]
    until_results = db.list_recent(until=base + timedelta(hours=1))
    assert [r.content for r in until_results] == ["a"]


def test_fts_keyword_search(db: Database) -> None:
    db.insert_clip("SELECT id FROM orders WHERE status = 'open'", content_type="code")
    db.insert_clip("react hooks article about useEffect", content_type="text")
    hits = db.search_keywords("orders", limit=5)
    assert hits
    assert "orders" in hits[0][0].content
    assert db.search_keywords("nonexistenttoken") == []


def test_fts_stays_in_sync_after_update_and_delete(db: Database) -> None:
    result = db.insert_clip("findme now")
    assert result.record is not None
    db.set_embedding(result.record.id, b"\x00\x01")
    assert db.search_keywords("findme")

    assert db.delete_by_id(result.record.id) is True
    assert db.search_keywords("findme") == []


def test_delete_all(db: Database) -> None:
    db.insert_clip("one")
    db.insert_clip("two")
    assert db.delete_all() == 2
    assert db.count() == 0
    assert db.search_keywords("one") == []


def test_type_counts(db: Database) -> None:
    db.insert_clip("https://a.dev", content_type="url")
    db.insert_clip("def f(): pass", content_type="code")
    db.insert_clip("more code", content_type="code")
    assert db.type_counts() == {"code": 2, "url": 1}


def test_iter_missing_embeddings(db: Database) -> None:
    a = db.insert_clip("a")
    db.insert_clip("b")
    assert a.record is not None
    db.set_embedding(a.record.id, b"blob")
    missing = list(db.iter_missing_embeddings())
    assert [r.content for r in missing] == ["b"]


def test_relative_time_and_preview() -> None:
    now = datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc)
    record = ClipRecord(
        id=1,
        content="line one\nline two",
        content_type="text",
        source_app=None,
        created_at=now - timedelta(minutes=5),
        content_hash="x",
    )
    assert record.preview() == "line one line two"
    assert record.relative_time(now) == "5m ago"
    assert record.char_count == 17


def test_context_manager_closes_connection(tmp_path) -> None:
    with Database(db_path=tmp_path / "ctx.db") as database:
        database.insert_clip("inside")
        assert database.count() == 1
