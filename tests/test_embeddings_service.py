"""Tests for :mod:`clipmind.embeddings.service` and the embedding policy."""

from __future__ import annotations

import time

import pytest

from clipmind.capture.monitor import ClipboardMonitor
from clipmind.config import Config
from clipmind.embeddings.codec import decode_vector
from clipmind.embeddings.encoder import (
    EmbeddingModelUnavailable,
    SentenceTransformerEmbedder,
    is_embeddable,
)
from clipmind.embeddings.service import (
    EmbeddingService,
    embed_batch,
    reindex_pending,
)
from clipmind.storage.db import Database
from fakes import BrokenEmbedder, FakeEmbedder


def _config(**kwargs) -> Config:
    return Config(db_path=":memory:", **kwargs)


@pytest.fixture()
def db() -> Database:
    database = Database(db_path=":memory:")
    try:
        yield database
    finally:
        database.close()


class _Reader:
    """Clipboard stand-in returning a fixed sequence of values."""

    def __init__(self, values):
        self.values = list(values)
        self.index = 0

    def paste(self):
        if self.index < len(self.values):
            value = self.values[self.index]
            self.index += 1
            return value
        return self.values[-1] if self.values else None


# ------------------------------------------------------------------ eligibility


@pytest.mark.parametrize(
    ("text", "content_type", "expected"),
    [
        ("SELECT id FROM orders", "code", True),
        ("a normal sentence", "text", True),
        ("42", "number", False),
        ("1234.5", "number", False),
        ("ab", "text", False),
        ("  a ", "text", False),  # whitespace-only padding still counts as short
        ("abc", "text", True),
        (None, "text", False),
        ("", "text", False),
    ],
)
def test_is_embeddable(text, content_type, expected) -> None:
    assert is_embeddable(text, content_type, _config()) is expected


def test_is_embeddable_honours_custom_skip_types() -> None:
    config = _config(embedding_skip_types=frozenset({"url"}))
    assert is_embeddable("https://react.dev", "url", config) is False
    assert is_embeddable("12345678", "number", config) is True


def test_is_embeddable_honours_min_chars() -> None:
    config = _config(embedding_min_chars=10)
    assert is_embeddable("short", "text", config) is False
    assert is_embeddable("long enough here", "text", config) is True


# ------------------------------------------------------------------ reindex


def test_reindex_embeds_only_rows_with_null_embedding(db: Database) -> None:
    from clipmind.embeddings.codec import encode_vector

    first = db.insert_clip("SELECT id FROM orders;", content_type="code")
    db.insert_clip("https://react.dev/reference/react/hooks", content_type="url")
    assert first.record is not None

    already = encode_vector(FakeEmbedder().embed_one("SELECT id FROM orders;"))
    db.set_embedding(first.record.id, already)

    report = reindex_pending(db, embedder=FakeEmbedder())

    assert report.missing == 1
    assert report.eligible == 1
    assert report.embedded == 1
    assert report.failed == 0
    # The pre-existing embedding is byte-identical: untouched, not rewritten.
    assert db.get_by_id(first.record.id).embedding == already


def test_reindex_skips_numbers_and_very_short_entries(db: Database) -> None:
    db.insert_clip("42", content_type="number")
    db.insert_clip("ab", content_type="text")
    db.insert_clip("a real sentence worth embedding", content_type="text")

    report = reindex_pending(db, embedder=FakeEmbedder())

    assert report.missing == 3
    assert report.eligible == 1
    assert report.skipped == 2
    assert report.embedded == 1
    assert db.count_missing_embeddings() == 2


def test_reindex_is_idempotent(db: Database) -> None:
    db.insert_clip("first entry to embed", content_type="text")
    db.insert_clip("second entry to embed", content_type="text")
    embedder = FakeEmbedder()

    first = reindex_pending(db, embedder=embedder)
    second = reindex_pending(db, embedder=embedder)

    assert first.embedded == 2
    assert second.missing == 0
    assert second.embedded == 0
    assert embedder.call_count == 1  # second pass embeds nothing, so no call


def test_reindex_reports_progress(db: Database) -> None:
    for i in range(5):
        db.insert_clip(f"entry number {i} with enough text", content_type="text")

    seen: list[tuple[int, int]] = []
    report = reindex_pending(
        db,
        embedder=FakeEmbedder(),
        batch_size=2,
        progress=lambda done, total: seen.append((done, total)),
    )

    assert report.embedded == 5
    assert seen == [(2, 5), (4, 5), (5, 5)]
    assert all(total == 5 for _done, total in seen)


def test_reindex_progress_fires_once_when_nothing_to_do(db: Database) -> None:
    seen: list[tuple[int, int]] = []
    reindex_pending(db, embedder=FakeEmbedder(), progress=lambda d, t: seen.append((d, t)))
    assert seen == [(0, 0)]


def test_reindex_counts_failures_without_raising(db: Database) -> None:
    db.insert_clip("some content to embed", content_type="text")
    db.insert_clip("more content to embed", content_type="text")

    report = reindex_pending(db, embedder=BrokenEmbedder())

    assert report.eligible == 2
    assert report.embedded == 0
    assert report.failed == 2
    assert db.count_missing_embeddings() == 2


def test_reindex_propagates_a_missing_model(db: Database, monkeypatch) -> None:
    """A missing model is an environment error, so it must not be counted as a
    per-entry failure and silently reported as 'failed N'."""
    import sys

    db.insert_clip("an eligible entry that needs embedding", content_type="text")
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    with pytest.raises(EmbeddingModelUnavailable):
        reindex_pending(db, embedder=SentenceTransformerEmbedder(), config=_config())


def test_worker_survives_a_missing_model(db: Database, monkeypatch) -> None:
    """The capture daemon must keep running when the model cannot be loaded."""
    import sys

    clip = db.insert_clip("daemon must survive this", content_type="text")
    assert clip.record is not None
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    service = EmbeddingService(db, embedder=SentenceTransformerEmbedder(), config=_config())
    service.start()
    try:
        service.submit(clip.record.id, clip.record.content, "text")
        assert service.flush(timeout=10) is True  # drained, not wedged
        assert service.stats["failed"] == 1
        assert "not installed" in (service.last_error or "")
    finally:
        service.stop()

    # Still usable afterwards: processing continues with a working embedder.
    assert db.get_by_id(clip.record.id).embedding is None


def test_embed_batch_writes_vectors_of_expected_dimension(db: Database) -> None:
    from clipmind.embeddings.service import EmbedItem

    result = db.insert_clip("embed me please", content_type="text")
    assert result.record is not None
    embedded, failed = embed_batch(
        db, FakeEmbedder(dimension=32), [EmbedItem(result.record.id, result.record.content, "text")]
    )
    assert (embedded, failed) == (1, 0)
    stored = decode_vector(db.get_by_id(result.record.id).embedding)
    assert stored.shape == (32,)


# ------------------------------------------------------------------ worker


def test_submit_skips_ineligible_entries(db: Database) -> None:
    service = EmbeddingService(db, embedder=FakeEmbedder())
    assert service.submit(1, "42", "number") is False
    assert service.submit(2, "ab", "text") is False
    assert service.submit(3, "a proper sentence", "text") is True
    assert service.stats["skipped"] == 2
    assert service.stats["queued"] == 1


def test_worker_writes_embeddings_on_flush(db: Database) -> None:
    result = db.insert_clip("background embedding target", content_type="text")
    assert result.record is not None

    with EmbeddingService(db, embedder=FakeEmbedder()) as service:
        assert service.submit(result.record.id, result.record.content, "text") is True
        assert service.flush(timeout=10) is True
        assert service.stats["embedded"] == 1

    assert db.get_by_id(result.record.id).embedding is not None


def test_flush_drains_the_queue(db: Database) -> None:
    service = EmbeddingService(db, embedder=FakeEmbedder(), config=_config(embedding_batch_size=2))
    service.start()
    try:
        for i in range(5):
            clip = db.insert_clip(f"queued entry {i} long enough", content_type="text")
            assert service.submit(clip.record.id, clip.record.content, "text") is True
        # The worker drains concurrently, so only the post-flush state is stable.
        assert service.stats["queued"] == 5
        assert service.flush(timeout=10) is True
        assert service.pending == 0
        assert service.stats["embedded"] == 5
        assert db.count_missing_embeddings() == 0
    finally:
        service.stop()


def test_worker_records_failures_without_killing_the_service(db: Database) -> None:
    clip = db.insert_clip("this will fail to embed", content_type="text")
    service = EmbeddingService(db, embedder=BrokenEmbedder())
    service.start()
    try:
        service.submit(clip.record.id, clip.record.content, "text")
        assert service.flush(timeout=10) is True
        assert service.stats["failed"] == 1
        assert service.stats["embedded"] == 0
        assert service.last_error is not None
    finally:
        service.stop()


def test_stop_is_idempotent(db: Database) -> None:
    service = EmbeddingService(db, embedder=FakeEmbedder())
    service.start()
    service.stop()
    service.stop()  # must not raise


def test_monitor_poll_does_not_block_on_the_embedder(db: Database) -> None:
    """The whole point of the worker: capture never waits for the model."""
    config = _config(poll_interval=0.0, embedding_batch_size=1)
    slow = FakeEmbedder(delay=0.5)
    service = EmbeddingService(db, embedder=slow, config=config)
    monitor = ClipboardMonitor(
        db,
        config,
        reader=_Reader(["a slow first entry", "another slow entry"]),
        embedding_service=service,
        detect_source_app=False,
        capture_initial=True,
    )
    service.start()
    try:
        started = time.perf_counter()
        assert monitor.poll_once() is not None
        first_elapsed = time.perf_counter() - started
        assert monitor.poll_once() is not None
        second_elapsed = time.perf_counter() - started

        # Two 0.5s embeddings are still in flight; polling returned immediately.
        assert first_elapsed < 0.25
        assert second_elapsed < 0.25

        assert service.flush(timeout=10) is True
        assert service.stats["embedded"] == 2
        assert all(record.embedding is not None for record in db.list_recent())
    finally:
        service.stop()


def test_monitor_without_embedding_service_never_embeds(db: Database) -> None:
    monitor = ClipboardMonitor(
        db,
        _config(poll_interval=0.0),
        reader=_Reader(["plain entry"]),
        detect_source_app=False,
        capture_initial=True,
    )
    assert monitor.poll_once() is not None
    assert db.list_recent()[0].embedding is None


def test_unchanged_clipboard_value_is_not_embedded_again(db: Database) -> None:
    config = _config(poll_interval=0.0)
    embedder = FakeEmbedder()
    service = EmbeddingService(db, embedder=embedder, config=config)
    monitor = ClipboardMonitor(
        db,
        config,
        reader=_Reader(["same text", "same text"]),
        embedding_service=service,
        detect_source_app=False,
        capture_initial=True,
    )
    service.start()
    try:
        monitor.poll_once()
        monitor.poll_once()  # unchanged value -> no insert, no embed
        assert service.flush(timeout=10) is True
        assert service.stats["queued"] == 1
        assert embedder.call_count == 1
    finally:
        service.stop()
