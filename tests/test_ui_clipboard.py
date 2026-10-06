"""Copy-back suppression: copying an entry back must not re-capture it.

Two guards are tested here: the ``SuppressingClipboard`` port updates the
monitor's baseline immediately, and the database rejects the content as a
duplicate (without touching the original timestamp) if a poll ever slips
through.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

from clipmind.capture.monitor import ClipboardMonitor
from clipmind.config import Config
from clipmind.storage.db import Database
from clipmind.ui.clipboard import SuppressingClipboard

TZ = timezone(timedelta(hours=-4))
EARLIER = datetime(2020, 1, 1, 9, 0, tzinfo=TZ)


class FakeReader:
    """Clipboard stand-in that always returns the same value."""

    def __init__(self, value: Optional[str]) -> None:
        self.value = value
        self.reads = 0

    def paste(self) -> Optional[str]:
        self.reads += 1
        return self.value


@pytest.fixture()
def db() -> Database:
    database = Database(db_path=":memory:")
    try:
        yield database
    finally:
        database.close()


def test_suppressing_clipboard_writes_then_remembers() -> None:
    events: list[tuple[str, str]] = []
    clipboard = SuppressingClipboard(
        write=lambda text: events.append(("write", text)),
        remember=lambda text: events.append(("remember", text)),
    )
    clipboard.copy("hello")
    assert events == [("write", "hello"), ("remember", "hello")]
    assert clipboard.last_copied == "hello"


def test_remembering_a_value_stops_the_next_poll_capturing_it(db: Database) -> None:
    config = Config(db_path=":memory:", poll_interval=0.0)
    reader = FakeReader("copied back")
    monitor = ClipboardMonitor(db, config, reader=reader, detect_source_app=False)

    # Simulate the popup copying this value back to the clipboard.
    monitor.remember_clipboard("copied back")
    assert monitor.poll_once() is None
    assert db.count() == 0


def test_copy_back_of_a_stored_item_does_not_bump_the_duplicate(db: Database) -> None:
    config = Config(db_path=":memory:", poll_interval=0.0)
    first = db.insert_clip("already stored", created_at=EARLIER)
    assert first.inserted and first.record is not None
    original_created_at = first.record.created_at

    reader = FakeReader("already stored")
    monitor = ClipboardMonitor(db, config, reader=reader, detect_source_app=False)
    monitor.remember_clipboard("already stored")
    assert monitor.poll_once() is None

    # Even if the baseline had not been set, dedup would reject the value and
    # leave the original row (and its timestamp) untouched.
    result = db.insert_clip("already stored")
    assert result.inserted is False
    assert result.reason == "duplicate"
    assert db.count() == 1
    assert db.get_by_id(first.record.id).created_at == original_created_at
