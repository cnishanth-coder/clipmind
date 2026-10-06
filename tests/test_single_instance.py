"""Tests for the ``clipmind app`` single-instance lock."""

from __future__ import annotations

import os
from pathlib import Path

from clipmind.ui.single_instance import SingleInstanceLock, default_lock_path


def test_second_acquire_reports_the_holder(tmp_path: Path) -> None:
    path = tmp_path / "app.lock"
    first = SingleInstanceLock(path)
    assert first.acquire() is None
    assert path.exists()
    assert path.read_text(encoding="utf-8").strip() == str(os.getpid())

    second = SingleInstanceLock(path)
    pid = second.acquire()
    assert pid == os.getpid()

    first.release()
    # Once released, a fresh lock can be taken again.
    assert second.acquire() is None
    second.release()


def test_release_is_idempotent(tmp_path: Path) -> None:
    lock = SingleInstanceLock(tmp_path / "app.lock")
    lock.acquire()
    lock.release()
    lock.release()  # must not raise


def test_lock_creates_missing_directories(tmp_path: Path) -> None:
    lock = SingleInstanceLock(tmp_path / "nested" / "dir" / "app.lock")
    assert lock.acquire() is None
    lock.release()


def test_default_lock_path() -> None:
    assert default_lock_path("/tmp/data") == Path("/tmp/data/app.lock")
