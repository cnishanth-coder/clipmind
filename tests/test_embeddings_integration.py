"""Integration test against the real default embedding model.

Marked ``slow`` and skipped unless pytest is run with ``--runslow``, because the
first execution downloads the weights (~90 MB for ``all-MiniLM-L6-v2``) from
Hugging Face (afterwards the model is read from the local cache and the test is
fully offline).

It asserts the behaviour that actually matters to a user: remembering *what a
thing was about* finds it, and an unrelated query returns nothing rather than
junk.
"""

from __future__ import annotations

import pytest

from clipmind.config import Config
from clipmind.embeddings.encoder import (
    DEFAULT_EMBEDDING_DIM,
    SentenceTransformerEmbedder,
)
from clipmind.embeddings.service import reindex_pending
from clipmind.search.semantic import SemanticSearcher
from clipmind.storage.db import Database

pytestmark = pytest.mark.slow

SQL = "SELECT id FROM orders;"
URL = "https://react.dev/reference/react/hooks"
MEETING = "meeting at 5pm with Rahul"


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    """Load the real model once for this module, skipping if it is unavailable."""
    pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")
    model = SentenceTransformerEmbedder()
    try:
        vectors = model.embed_texts(["warm up"])
    except Exception as exc:  # noqa: BLE001 - any load/download failure means "skip"
        pytest.skip(f"embedding model unavailable (not downloaded?): {exc}")
    assert vectors.shape == (1, DEFAULT_EMBEDDING_DIM)
    return model


@pytest.fixture()
def db(tmp_path) -> Database:
    """A database holding the three sample entries, with no embeddings."""
    database = Database(db_path=tmp_path / "integration.db")
    for content, content_type in ((SQL, "code"), (URL, "url"), (MEETING, "text")):
        result = database.insert_clip(content, content_type=content_type)
        assert result.inserted
    try:
        yield database
    finally:
        database.close()


def test_reindex_then_semantic_search_ranks_expected_documents(
    db: Database, embedder: SentenceTransformerEmbedder
) -> None:
    report = reindex_pending(db, embedder=embedder)
    assert report.embedded == 3
    assert report.failed == 0
    assert db.count_missing_embeddings() == 0

    searcher = SemanticSearcher(db, embedder=embedder, config=Config(db_path=":memory:"))

    memory_of_sql = searcher.search("database query")
    assert memory_of_sql, "expected the SQL entry to clear the min-score threshold"
    assert memory_of_sql[0].content == SQL

    memory_of_react = searcher.search("react hooks article")
    assert memory_of_react, "expected the React link to clear the min-score threshold"
    assert memory_of_react[0].content == URL


def test_unrelated_query_returns_nothing_above_threshold(
    db: Database, embedder: SentenceTransformerEmbedder
) -> None:
    reindex_pending(db, embedder=embedder)
    searcher = SemanticSearcher(db, embedder=embedder, config=Config(db_path=":memory:"))

    results = searcher.search("a random unrelated topic about gardening")

    assert results == []
    assert searcher.last_best_score is not None
    assert searcher.last_best_score < searcher.last_threshold


def test_embeddings_are_reused_and_dimension_matches_the_model(
    db: Database, embedder: SentenceTransformerEmbedder
) -> None:
    reindex_pending(db, embedder=embedder)
    after_first = [record.embedding for record in db.list_recent()]

    # A second sweep must not touch rows that already have embeddings.
    second = reindex_pending(db, embedder=embedder)
    assert second.missing == 0
    assert [record.embedding for record in db.list_recent()] == after_first

    assert all(
        blob is not None and len(blob) == DEFAULT_EMBEDDING_DIM * 4
        for blob in after_first
    )
