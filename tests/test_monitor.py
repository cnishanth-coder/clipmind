"""Tests for :mod:`clipmind.capture.monitor`."""

from __future__ import annotations

from typing import Optional

import pytest

from clipmind.capture.monitor import ClipboardMonitor
from clipmind.config import Config
from clipmind.storage.db import Database


class FakeReader:
    """Deterministic clipboard stand-in."""

    def __init__(self, values: list[Optional[str]]) -> None:
        self.values = list(values)
        self.index = 0

    def paste(self) -> Optional[str]:
        if self.index < len(self.values):
            value = self.values[self.index]
            self.index += 1
            return value
        return self.values[-1] if self.values else None


@pytest.fixture()
def db() -> Database:
    database = Database(db_path=":memory:")
    try:
        yield database
    finally:
        database.close()


def _monitor(db: Database, values: list[Optional[str]], **kwargs) -> ClipboardMonitor:
    config = Config(db_path=":memory:", poll_interval=0.0)
    return ClipboardMonitor(
        db,
        config,
        reader=FakeReader(values),
        detect_source_app=False,
        **kwargs,
    )


def test_capture_stores_entry(db: Database) -> None:
    monitor = _monitor(db, [])
    result = monitor.capture("https://example.com", source_app="Safari")
    assert result is not None and result.inserted
    assert result.record is not None
    assert result.record.content_type == "url"
    assert result.record.source_app == "Safari"


def test_capture_empty_returns_none(db: Database) -> None:
    monitor = _monitor(db, [])
    assert monitor.capture("   ") is None
    assert db.count() == 0


def test_initial_clipboard_is_ignored_by_default(db: Database) -> None:
    monitor = _monitor(db, ["preexisting", "preexisting", "new value"])
    assert monitor.poll_once() is None  # seeded, not stored
    assert monitor.poll_once() is None  # unchanged
    result = monitor.poll_once()
    assert result is not None and result.inserted
    assert db.count() == 1


def test_capture_initial_flag_stores_existing_value(db: Database) -> None:
    monitor = _monitor(db, ["already here"], capture_initial=True)
    result = monitor.poll_once()
    assert result is not None and result.inserted
    assert db.list_recent()[0].content == "already here"


def test_repeated_values_are_deduplicated(db: Database) -> None:
    monitor = _monitor(db, ["a", "b", "a"], capture_initial=True)
    assert monitor.poll_once() is not None  # a
    assert monitor.poll_once() is not None  # b
    third = monitor.poll_once()  # a again -> duplicate in DB
    assert third is not None
    assert third.inserted is False
    assert third.reason == "duplicate"
    assert db.count() == 2


def test_stats_track_counts(db: Database) -> None:
    monitor = _monitor(db, ["a", "a"], capture_initial=True)
    monitor.poll_once()
    monitor.poll_once()  # unchanged, ignored
    stats = monitor.stats
    assert stats["polls"] == 2
    assert stats["new"] == 1


def test_run_with_max_iterations(db: Database) -> None:
    monitor = _monitor(db, ["one", "two", "three"], capture_initial=True)
    monitor.run(max_iterations=3)
    assert db.count() == 3
    assert monitor.stats["polls"] == 3


def test_reader_errors_do_not_crash_the_loop(db: Database) -> None:
    class BrokenReader:
        def paste(self):
            raise RuntimeError("clipboard unavailable")

    config = Config(db_path=":memory:", poll_interval=0.0)
    monitor = ClipboardMonitor(db, config, reader=BrokenReader(), detect_source_app=False)
    monitor.poll_once()
    assert monitor.stats["polls"] == 1


def test_source_app_env_override(monkeypatch, db: Database) -> None:
    from clipmind.capture.source_app import frontmost_app_name

    monkeypatch.setenv("CLIPMIND_SOURCE_APP", "1Password")
    assert frontmost_app_name() == "1Password"
