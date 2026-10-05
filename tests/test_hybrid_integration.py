"""End-to-end hybrid search against the real ``all-MiniLM-L6-v2`` model.

Marked ``slow`` and skipped unless pytest is run with ``--runslow``. It builds a
small, realistic history across several timestamps and checks the three
behaviours Phase 3 exists for:

* a time phrase selects only the right day (``"react hooks article yesterday"``);
* an exact arbitrary string is found by the keyword retriever (``"ERR_4021"``);
* a purely meaning-based query still finds the SQL entry (``"database query"``).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from clipmind.config import Config
from clipmind.embeddings.encoder import SentenceTransformerEmbedder
from clipmind.embeddings.service import reindex_pending
from clipmind.search.hybrid import hybrid_search
from clipmind.storage.db import Database

pytestmark = pytest.mark.slow

UTC = timezone.utc

#: Fixed reference instant so the parsed "yesterday" window is deterministic.
NOW = datetime(2026, 10, 7, 14, 30, tzinfo=UTC)

REACT_URL = "https://react.dev/reference/react/hooks"
BUDGET = "quarterly budget meeting notes"
SQL = "SELECT id FROM orders WHERE status = 'open';"
ERR = "ERR_4021 database connection timeout"
ICELAND = "Iceland trip packing checklist"
EMAIL = "from: rahul@example.com subject: deploy notes"
SHORT_NUMBER = "424242"
CODE = "def parse_time_query(query): return query"

#: ``(content, content_type, captured_at)`` - eight varied, timestamped entries.
ENTRIES = [
    (REACT_URL, "url", datetime(2026, 10, 6, 10, 0, tzinfo=UTC)),   # yesterday
    (BUDGET, "text", datetime(2026, 10, 6, 15, 0, tzinfo=UTC)),     # yesterday
    (SQL, "code", datetime(2026, 10, 4, 9, 0, tzinfo=UTC)),         # 3 days ago
    (ERR, "text", datetime(2026, 10, 5, 11, 0, tzinfo=UTC)),        # 2 days ago
    (ICELAND, "text", datetime(2026, 10, 2, 8, 0, tzinfo=UTC)),     # 5 days ago
    (EMAIL, "email", datetime(2026, 10, 7, 9, 0, tzinfo=UTC)),      # today
    (SHORT_NUMBER, "number", datetime(2026, 10, 7, 8, 0, tzinfo=UTC)),
    (CODE, "code", NOW - timedelta(days=6)),
]


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    """Load the real model once for this module, skipping if unavailable."""
    pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")
    model = SentenceTransformerEmbedder()
    try:
        model.embed_texts(["warm up"])
    except Exception as exc:  # noqa: BLE001 - any load/download failure means "skip"
        pytest.skip(f"embedding model unavailable (not downloaded?): {exc}")
    return model


@pytest.fixture()
def populated(tmp_path, embedder) -> tuple[Database, Config]:
    """A database with all eight entries embedded where eligible."""
    config = Config(db_path=tmp_path / "hybrid-integration.db")
    db = Database(config=config)
    for content, content_type, created_at in ENTRIES:
        assert db.insert_clip(content, content_type=content_type, created_at=created_at).inserted
    report = reindex_pending(db, embedder=embedder, config=config)
    assert report.failed == 0
    try:
        yield db, config
    finally:
        db.close()


def test_time_phrase_restricts_results_to_yesterday(populated, embedder) -> None:
    db, config = populated
    outcome = hybrid_search(
        db, "react hooks article yesterday", now=NOW, embedder=embedder, config=config
    )

    assert outcome.query.cleaned_query == "react hooks article"
    assert outcome.query.has_range
    assert outcome.results, "expected the React link from yesterday to be found"

    # Every hit must fall inside yesterday's window...
    yesterday = datetime(2026, 10, 6, tzinfo=UTC)
    assert all(yesterday <= r.created_at < yesterday + timedelta(days=1) for r in outcome.results)
    # ...and the React link must be the best hit.
    assert "react.dev" in outcome.results[0].content
    # The older SQL/code entries must not leak through the filter.
    assert all("SELECT id FROM orders" not in r.content for r in outcome.results)


def test_exact_string_is_found_by_the_keyword_retriever(populated, embedder) -> None:
    db, config = populated
    outcome = hybrid_search(db, "ERR_4021", now=NOW, embedder=embedder, config=config)

    assert outcome.used_keyword
    assert outcome.results
    assert "ERR_4021" in outcome.results[0].content
    exact = next(r for r in outcome.results if "ERR_4021" in r.content)
    assert exact.keyword_coverage == 1.0
    assert exact.keyword_rank == 1


def test_meaning_based_query_still_finds_the_sql_entry(populated, embedder) -> None:
    db, config = populated
    outcome = hybrid_search(db, "database query", now=NOW, embedder=embedder, config=config)

    assert outcome.used_semantic
    assert any("SELECT id FROM orders" in r.content for r in outcome.results)
