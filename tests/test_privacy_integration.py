"""End-to-end privacy through the capture pipeline (fake embedder, no model).

The key assertion is that a masked secret is absent from *everything* that
persists or gets computed: the stored row, the FTS index, and the text handed to
the embedding model.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

from clipmind.capture.monitor import ClipboardMonitor
from clipmind.config import Config
from clipmind.embeddings.service import EmbeddingService
from clipmind.privacy.detectors import detect_secrets
from clipmind.search.keyword import sanitize_fts_query
from clipmind.storage.db import Database
from fakes import FakeEmbedder

UTC = timezone.utc
AWS_KEY = "AKIAIOSFODNN7EXAMPLE"


def _config(**kwargs) -> Config:
    return Config(db_path=":memory:", **kwargs)


@pytest.fixture()
def db() -> Database:
    database = Database(db_path=":memory:")
    try:
        yield database
    finally:
        database.close()


class SequenceReader:
    """Clipboard reader returning a fixed sequence of values."""

    def __init__(self, values: list[Optional[str]]) -> None:
        self._values = list(values)
        self._index = 0
        self.reads = 0

    def paste(self) -> Optional[str]:
        self.reads += 1
        if self._index < len(self._values):
            value = self._values[self._index]
        else:
            value = self._values[-1] if self._values else None
        self._index += 1
        return value


# ----------------------------------------------------------- mask end to end


def test_masked_secret_never_reaches_db_fts_or_embeddings(db: Database) -> None:
    config = _config()
    embedder = FakeEmbedder()
    service = EmbeddingService(db, embedder=embedder, config=config).start()
    monitor = ClipboardMonitor(
        db, config, embedding_service=service, detect_source_app=False
    )

    result = monitor.capture(f"my AWS key is {AWS_KEY} keep this")
    service.stop()

    assert result is not None and result.inserted
    record = db.get_by_id(result.record.id)
    assert record is not None

    # 1. The stored row holds the masked text, not the secret.
    assert AWS_KEY not in record.content
    assert "[REDACTED:aws_access_key]" in record.content

    # 2. No row anywhere in the table contains the secret.
    assert all(AWS_KEY not in row.content for row in db.list_recent())

    # 3. The FTS index cannot find the secret token.
    assert db.search_keywords(sanitize_fts_query(AWS_KEY)) == []

    # 4. The embedding model only ever saw the masked text.
    assert embedder.embedded_texts
    assert all(AWS_KEY not in text for text in embedder.embedded_texts)
    assert any("[REDACTED:aws_access_key]" in text for text in embedder.embedded_texts)


def test_content_hash_is_computed_over_masked_text(db: Database) -> None:
    from clipmind.storage.db import hash_content

    config = _config()
    monitor = ClipboardMonitor(db, config, detect_source_app=False)
    result = monitor.capture(f"key {AWS_KEY}")
    assert result is not None and result.inserted and result.record is not None
    stored = db.get_by_id(result.record.id)
    assert stored is not None
    assert stored.content_hash == hash_content(stored.content)
    assert stored.content_hash != hash_content(f"key {AWS_KEY}")


# ----------------------------------------------------------- skip end to end


def test_skipped_secret_is_not_stored(db: Database) -> None:
    monitor = ClipboardMonitor(db, _config(), detect_source_app=False)
    result = monitor.capture("password=hunter2")
    assert result is not None
    assert result.inserted is False
    assert result.reason == "secret"
    assert db.count() == 0


def test_skipped_private_key_is_not_stored(db: Database) -> None:
    monitor = ClipboardMonitor(db, _config(), detect_source_app=False)
    secret = "-----BEGIN EC PRIVATE KEY-----\nABCDEF\n-----END EC PRIVATE KEY-----"
    result = monitor.capture(secret)
    assert result is not None and result.reason == "secret"
    assert db.count() == 0


def test_privacy_disabled_stores_the_secret(db: Database) -> None:
    monitor = ClipboardMonitor(db, _config(privacy_enabled=False), detect_source_app=False)
    result = monitor.capture("password=hunter2")
    assert result is not None and result.inserted
    assert db.count() == 1


# --------------------------------------------------------------- ignore list


def test_ignored_source_app_is_not_captured(db: Database) -> None:
    monitor = ClipboardMonitor(db, _config(), detect_source_app=False)

    blocked = monitor.capture("master password", source_app="1Password 8")
    assert blocked is not None and blocked.reason == "ignored"
    assert db.count() == 0

    allowed = monitor.capture("public note", source_app="Safari")
    assert allowed is not None and allowed.inserted
    assert db.count() == 1

    # Unknown source app degrades gracefully: capture continues.
    unknown = monitor.capture("another note", source_app=None)
    assert unknown is not None and unknown.inserted
    assert db.count() == 2


# ------------------------------------------------------------- pause / resume


def test_paused_monitor_does_not_capture_and_resume_restores_it(db: Database) -> None:
    reader = SequenceReader(["first", "second", "third"])
    monitor = ClipboardMonitor(db, _config(), reader=reader, detect_source_app=False)

    db.set_paused(True)
    assert monitor.poll_once() is None
    assert db.count() == 0
    assert reader.reads == 0  # the clipboard is not even read while paused
    assert monitor.stats["paused"] == 1

    db.set_paused(False)  # live resume, no restart
    monitor.poll_once()  # seeds the baseline ("first")
    assert db.count() == 0
    monitor.poll_once()  # "second" is new
    assert db.count() == 1
    assert db.list_recent()[0].content == "second"


def test_pause_flag_is_persisted(db: Database) -> None:
    assert db.is_paused() is False
    db.set_paused(True)
    assert db.is_paused() is True
    # A second Database instance on the same file would also see it; the
    # in-memory flag is read straight from the meta table each poll.
    assert db.get_meta("paused") == "1"


# --------------------------------------------------------------- retention


def test_retention_purge_removes_old_rows_and_updates_fts(db: Database) -> None:
    old = datetime(2026, 1, 1, tzinfo=UTC)
    recent = datetime.now(UTC) - timedelta(days=1)
    db.insert_clip("ancient secret-free note", content_type="text", created_at=old)
    db.insert_clip("recent note about orders", content_type="text", created_at=recent)

    config = _config(retention_days=30)
    monitor = ClipboardMonitor(db, config, detect_source_app=False)

    removed = monitor.purge_expired(force=True)
    assert removed == 1
    remaining = db.list_recent()
    assert [r.content for r in remaining] == ["recent note about orders"]
    # The purged text is gone from the FTS index too ("ancient" appears only
    # in the deleted row, so an OR-joined match cannot hit the survivor).
    assert db.search_keywords(sanitize_fts_query("ancient")) == []


def test_retention_disabled_does_nothing(db: Database) -> None:
    db.insert_clip("old note", content_type="text", created_at=datetime(2020, 1, 1, tzinfo=UTC))
    monitor = ClipboardMonitor(db, _config(), detect_source_app=False)
    assert monitor.purge_expired(force=True) == 0
    assert db.count() == 1


def test_retention_runs_at_most_daily_without_force(db: Database) -> None:
    db.insert_clip("old note", content_type="text", created_at=datetime(2020, 1, 1, tzinfo=UTC))
    config = _config(retention_days=1)
    monitor = ClipboardMonitor(db, config, detect_source_app=False)

    assert monitor.purge_expired(force=True) == 1
    # A second sweep moments later is skipped by the 24h guard.
    db.insert_clip("another old note", content_type="text", created_at=datetime(2020, 1, 2, tzinfo=UTC))
    assert monitor.purge_expired() == 0
