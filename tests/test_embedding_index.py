"""Tests for :mod:`clipmind.search.embedding_index` and the incremental
:class:`~clipmind.search.semantic.SemanticSearcher` index (fake embedder, no model).

These prove the property the benchmark relies on: the vector matrix is built
once and then kept correct as rows are inserted, embedded, deleted, purged,
cleared and reindexed - without any per-query re-read of the embedding table.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from clipmind.config import Config
from clipmind.embeddings.codec import encode_vector
from clipmind.embeddings.service import embed_batch, reindex_pending
from clipmind.search.embedding_index import EmbeddingIndex
from clipmind.search.semantic import SemanticSearcher
from clipmind.storage.db import Database, DatabaseChange, coerce_timestamp
from fakes import FakeEmbedder, embed_into_db

TZ = timezone(timedelta(hours=-4))
BASE = datetime(2026, 10, 1, 12, 0, tzinfo=TZ)


def _config(**kwargs) -> Config:
    return Config(db_path=":memory:", **kwargs)


@pytest.fixture()
def db() -> Database:
    database = Database(db_path=":memory:")
    try:
        yield database
    finally:
        database.close()


# --------------------------------------------------------------------- unit index


def _row(item_id: int, content: str, content_type: str, created_at: datetime, blob: bytes):
    return (item_id, content_type, coerce_timestamp(created_at), blob)


def test_load_builds_a_row_aligned_matrix() -> None:
    embedder = FakeEmbedder()
    index = EmbeddingIndex()
    index.load(
        [
            _row(1, "alpha beta", "text", BASE, encode_vector(embedder.embed_one("alpha beta"))),
            _row(2, "gamma delta", "code", BASE, encode_vector(embedder.embed_one("gamma delta"))),
        ]
    )

    assert index.loaded
    assert index.size == 2
    assert index.dimension == embedder.dimension
    # Rows are stored L2-normalised.
    assert np.allclose(np.linalg.norm(index._vectors, axis=1), 1.0, atol=1e-5)


def test_select_masks_by_type_and_date() -> None:
    embedder = FakeEmbedder()
    index = EmbeddingIndex()
    index.load(
        [
            _row(1, "a", "code", BASE, encode_vector(embedder.embed_one("a"))),
            _row(2, "b", "url", BASE + timedelta(days=1), encode_vector(embedder.embed_one("b"))),
            _row(3, "c", "code", BASE + timedelta(days=2), encode_vector(embedder.embed_one("c"))),
        ]
    )

    assert list(index.select(content_type="code")) == [True, False, True]
    assert list(index.select(since=BASE + timedelta(hours=1))) == [False, True, True]
    assert list(index.select(until=BASE + timedelta(hours=1))) == [True, False, False]
    assert list(index.select(content_type="code", since=BASE + timedelta(days=1))) == [
        False,
        False,
        True,
    ]
    assert index.count(content_type="code") == 2
    # Filtering must match the SQL date semantics (>= since, < until).
    assert index.count(since=BASE + timedelta(days=1)) == 2
    assert index.count(until=BASE + timedelta(days=1)) == 1


def test_apply_insert_embedding_delete_clear() -> None:
    embedder = FakeEmbedder()
    index = EmbeddingIndex()
    index.load([])

    index.apply(
        DatabaseChange(
            generation=1,
            kind="insert",
            item_id=10,
            content_type="text",
            created_at=coerce_timestamp(BASE),
            embedding=encode_vector(embedder.embed_one("hello world")),
        )
    )
    assert index.size == 1
    assert index.count(content_type="text") == 1

    # An 'embedding' change for the same id replaces in place rather than adding.
    index.apply(
        DatabaseChange(
            generation=2,
            kind="embedding",
            item_id=10,
            content_type="code",
            created_at=coerce_timestamp(BASE),
            embedding=encode_vector(embedder.embed_one("SELECT 1;")),
        )
    )
    assert index.size == 1
    assert index.count(content_type="code") == 1

    index.apply(DatabaseChange(generation=3, kind="delete", item_id=10))
    assert index.size == 0

    index.apply(
        DatabaseChange(
            generation=4,
            kind="insert",
            item_id=11,
            content_type="text",
            created_at=coerce_timestamp(BASE),
            embedding=encode_vector(embedder.embed_one("alpha")),
        )
    )
    index.apply(
        DatabaseChange(
            generation=5,
            kind="insert",
            item_id=12,
            content_type="text",
            created_at=coerce_timestamp(BASE),
            embedding=encode_vector(embedder.embed_one("beta")),
        )
    )
    index.apply(DatabaseChange(generation=6, kind="delete_many", item_ids=(11,)))
    assert index.size == 1
    assert index.count() == 1

    index.apply(DatabaseChange(generation=7, kind="clear"))
    assert index.size == 0


def test_apply_before_load_is_a_noop() -> None:
    embedder = FakeEmbedder()
    index = EmbeddingIndex()
    index.apply(
        DatabaseChange(
            generation=1,
            kind="insert",
            item_id=1,
            content_type="text",
            created_at=coerce_timestamp(BASE),
            embedding=encode_vector(embedder.embed_one("hello")),
        )
    )
    assert index.size == 0


def test_search_orders_by_similarity_within_the_mask() -> None:
    embedder = FakeEmbedder()
    index = EmbeddingIndex()
    index.load(
        [
            _row(1, "alpha beta gamma", "text", BASE, encode_vector(embedder.embed_one("alpha beta gamma"))),
            _row(2, "delta epsilon", "code", BASE, encode_vector(embedder.embed_one("delta epsilon"))),
        ]
    )
    query = embedder.embed_one("alpha beta")
    hits = index.search(query, 5)
    assert [hit.key for hit in hits][0] == 1
    # The type mask excludes the other row entirely.
    code_hits = index.search(query, 5, content_type="code")
    assert [hit.key for hit in code_hits] == [2]


# -------------------------------------------------- incremental searcher updates


def _searcher(db: Database, **kwargs) -> tuple[SemanticSearcher, FakeEmbedder]:
    embedder = FakeEmbedder()
    return SemanticSearcher(db, embedder=embedder, config=_config(**kwargs)), embedder


def test_new_embedding_is_visible_without_invalidating(db: Database) -> None:
    embed_into_db(db, [("react hooks article", "text")])
    searcher, _embedder = _searcher(db)

    first = searcher.search("react hooks", min_score=0.0)
    assert searcher.last_index_size == 1
    assert [r.content for r in first] == ["react hooks article"]

    # Insert + embed while the searcher is alive: the index must grow in place.
    embed_into_db(db, [("react hooks reference", "text")])
    second = searcher.search("react hooks", min_score=0.0)
    assert searcher.last_index_size == 2
    assert {r.content for r in second} >= {"react hooks article", "react hooks reference"}


def test_insert_with_embedding_is_indexed(db: Database) -> None:
    searcher, embedder = _searcher(db)
    searcher.search("warm up the index", min_score=0.0)  # load empty index
    assert searcher.last_index_size == 0

    vector = embedder.embed_one("a directly inserted embedding")
    result = db.insert_clip(
        "a directly inserted embedding",
        content_type="text",
        created_at=BASE,
        embedding=encode_vector(vector),
    )
    assert result.inserted

    results = searcher.search("inserted embedding", min_score=0.0)
    assert searcher.last_index_size == 1
    assert [r.content for r in results] == ["a directly inserted embedding"]


def test_delete_by_id_removes_from_the_index(db: Database) -> None:
    embed_into_db(db, [("alpha entry", "text"), ("beta entry", "text")])
    searcher, _embedder = _searcher(db)
    searcher.search("entry", min_score=0.0)
    assert searcher.last_index_size == 2

    stored = {r.content: r.id for r in db.list_recent(limit=10)}
    assert db.delete_by_id(stored["alpha entry"]) is True

    searcher.search("entry", min_score=0.0)
    assert searcher.last_index_size == 1
    assert all(r.content != "alpha entry" for r in searcher.search("alpha", min_score=0.0))


def test_purge_removes_only_the_old_rows(db: Database) -> None:
    old = BASE - timedelta(days=40)
    recent = BASE - timedelta(days=1)
    embed_into_db(db, [("ancient note", "text"), ("recent note", "text")])
    ids = {r.content: r.id for r in db.list_recent(limit=10)}
    db._conn.execute(
        "UPDATE items SET created_at = ? WHERE id = ?", (coerce_timestamp(old), ids["ancient note"])
    )
    db._conn.execute(
        "UPDATE items SET created_at = ? WHERE id = ?", (coerce_timestamp(recent), ids["recent note"])
    )
    db._conn.commit()

    searcher, _embedder = _searcher(db)
    assert searcher.index_size() == 2

    removed = db.delete_older_than(BASE - timedelta(days=10))
    assert removed == 1

    assert searcher.index_size() == 1
    contents = [r.content for r in searcher.search("note", min_score=0.0, top_k=10)]
    assert contents == ["recent note"]


def test_clear_empties_the_index(db: Database) -> None:
    embed_into_db(db, [("alpha entry", "text"), ("beta entry", "text")])
    searcher, _embedder = _searcher(db)
    assert searcher.index_size() == 2

    assert db.delete_all() == 2
    assert searcher.index_size() == 0
    assert searcher.search("entry", min_score=0.0) == []


def test_reindex_updates_the_live_index(db: Database) -> None:
    db.insert_clip("reindex me please", content_type="text", created_at=BASE)
    searcher, _embedder = _searcher(db)
    assert searcher.index_size() == 0

    report = reindex_pending(db, embedder=FakeEmbedder(), config=_config())
    assert report.embedded == 1

    assert searcher.index_size() == 1
    results = searcher.search("reindex me", min_score=0.0)
    assert [r.content for r in results] == ["reindex me please"]


def test_embed_batch_updates_the_live_index(db: Database) -> None:
    from clipmind.embeddings.service import EmbedItem

    result = db.insert_clip("batch embedded content", content_type="text", created_at=BASE)
    searcher, _embedder = _searcher(db)
    searcher.search("warm", min_score=0.0)

    embedded, failed = embed_batch(
        db,
        FakeEmbedder(),
        [EmbedItem(result.record.id, result.record.content, result.record.content_type)],
    )
    assert (embedded, failed) == (1, 0)
    assert searcher.index_size() == 1


def test_invalidate_forces_a_reload(db: Database) -> None:
    embed_into_db(db, [("alpha entry", "text")])
    searcher, _embedder = _searcher(db)
    assert searcher.index_size() == 1
    searcher.invalidate()
    assert searcher.index_size() == 1


# ---------------------------------------------------------- filter correctness


def test_filters_match_sql_list_indexed(db: Database) -> None:
    entries = [
        ("code one", "code", BASE),
        ("url one", "url", BASE + timedelta(days=1)),
        ("code two", "code", BASE + timedelta(days=2)),
        ("text one", "text", BASE + timedelta(days=3)),
    ]
    embedder = FakeEmbedder()
    for content, content_type, created_at in entries:
        result = db.insert_clip(
            content, content_type=content_type, created_at=created_at, dedup=False
        )
        db.set_embedding(result.record.id, encode_vector(embedder.embed_one(content)))

    searcher = SemanticSearcher(db, embedder=embedder, config=_config())

    cases = [
        {"content_type": "code"},
        {"content_type": "url"},
        {"since": BASE + timedelta(hours=12)},
        {"until": BASE + timedelta(days=2)},
        {"content_type": "code", "since": BASE + timedelta(days=1)},
        {"content_type": "code", "until": BASE + timedelta(days=1)},
    ]
    for case in cases:
        expected = {
            record.id
            for record in db.list_indexed(
                case.get("content_type"),
                since=case.get("since"),
                until=case.get("until"),
            )
        }
        assert searcher.index_size(**case) == len(expected)
        results = searcher.search("one two three", top_k=10, min_score=-1.0, **case)
        assert {r.id for r in results} == expected


def test_content_type_filter_reports_filtered_size(db: Database) -> None:
    embed_into_db(db, [("code snippet", "code"), ("text snippet", "text")])
    searcher, _embedder = _searcher(db)
    searcher.search("snippet", content_type="code", min_score=0.0)
    assert searcher.last_index_size == 1


def test_model_mismatch_is_rejected(db: Database) -> None:
    from clipmind.embeddings.encoder import EmbeddingModelMismatch

    embed_into_db(db, [("alpha entry", "text")])
    db.set_embedding_model("all-MiniLM-L6-v2")
    searcher = SemanticSearcher(
        db,
        embedder=FakeEmbedder(),
        config=_config(embedding_model="BAAI/bge-small-en-v1.5"),
    )
    with pytest.raises(EmbeddingModelMismatch):
        searcher.search("alpha")


def test_legacy_database_adopts_the_configured_model(db: Database) -> None:
    embed_into_db(db, [("alpha entry", "text")])
    assert db.get_embedding_model() is None
    searcher, _embedder = _searcher(db)
    searcher.search("alpha", min_score=0.0)
    assert db.get_embedding_model() == Config().embedding_model


def test_stale_generation_falls_back_to_a_reload(db: Database) -> None:
    embed_into_db(db, [("alpha entry", "text")])
    searcher, _embedder = _searcher(db)
    assert searcher.index_size() == 1

    # Simulate a change whose event never reached the index.
    result = db.insert_clip("beta entry", content_type="text", created_at=BASE, dedup=False)
    db.set_embedding(result.record.id, encode_vector(FakeEmbedder().embed_one("beta entry")))
    with searcher._index_lock:
        searcher._index_generation = -1  # pretend the events were missed

    assert searcher.index_size() == 2
