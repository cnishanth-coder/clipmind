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


def test_second_acquire_reports_zero_when_pid_is_unreadable(
    tmp_path: Path, monkeypatch
) -> None:
    """A failing pid read must still report "already running", without a pid.

    This is the Windows failure mode: a locked byte range cannot be read by
    another handle, so reading the holder's pid may raise ``PermissionError``.
    The second launch must degrade to an unknown pid (``0``) rather than crash.
    """
    path = tmp_path / "app.lock"
    first = SingleInstanceLock(path)
    assert first.acquire() is None
    try:
        def boom(*_args, **_kwargs):
            raise PermissionError("simulated locked byte range")

        monkeypatch.setattr(SingleInstanceLock, "_read_pid", boom)
        second = SingleInstanceLock(path)
        assert second.acquire() == 0
    finally:
        first.release()


def test_read_pid_returns_zero_for_unparsable_content(tmp_path: Path) -> None:
    """Garbage in the lock file yields pid ``0`` instead of raising."""
    path = tmp_path / "app.lock"
    path.write_text("not-a-pid\n", encoding="utf-8")
    with open(path, encoding="utf-8") as handle:
        assert SingleInstanceLock._read_pid(handle) == 0


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
