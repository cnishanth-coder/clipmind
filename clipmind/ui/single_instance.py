"""Single-instance guard for the ClipMind app.

``clipmind app`` must be one long-running process: a second start would fight
over the clipboard, the tray icon and the hotkey. A lock file in the data
directory is used instead of an OS mutex so it works the same way on macOS,
Linux and Windows and leaves behind a readable pid for the error message.

The lock is advisory and released automatically by the OS when the process
exits, so a crashed ClipMind never blocks the next start (no stale-lock
cleanup needed).
"""

from __future__ import annotations

import os
from pathlib import Path

#: Filename used inside the data directory.
LOCK_FILENAME = "app.lock"


class SingleInstanceLock:
    """Advisory, non-blocking file lock used to enforce one ClipMind process.

    Example:
        >>> lock = SingleInstanceLock(Path("/tmp/clipmind-app.lock"))
        >>> other = lock.acquire()
        >>> assert other is None  # we got it
        >>> lock.release()
    """

    def __init__(self, path: Path | str) -> None:
        """Remember the lock file path (created on :meth:`acquire`)."""
        self.path = Path(path)
        self._handle = None

    def acquire(self) -> int | None:
        """Try to take the lock without blocking.

        Returns:
            ``None`` when the lock was acquired, otherwise the pid of the
            process that already holds it (``0`` when the pid cannot be read).
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.path, "a+", encoding="utf-8")
        try:
            self._lock_handle(handle)
        except OSError:
            pid = self._read_pid(handle)
            handle.close()
            return pid
        # We own the lock: record our pid for the next caller's message.
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(f"{os.getpid()}\n")
            handle.flush()
        except OSError:  # pragma: no cover - best effort
            pass
        self._handle = handle
        return None

    def release(self) -> None:
        """Release the lock and remove the lock file (safe to call twice)."""
        if self._handle is None:
            return
        try:
            self._unlock_handle(self._handle)
        except OSError:  # pragma: no cover - best effort
            pass
        try:
            self._handle.close()
        finally:
            self._handle = None
        try:
            self.path.unlink()
        except OSError:  # pragma: no cover - another instance may have removed it
            pass

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _read_pid(handle) -> int:
        """Read the pid written by the current holder, or ``0``."""
        try:
            handle.seek(0)
            text = handle.read().strip()
            return int(text) if text else 0
        except (OSError, ValueError):
            return 0

    @staticmethod
    def _lock_handle(handle) -> None:
        """Take an exclusive, non-blocking lock on ``handle``."""
        if os.name == "nt":  # pragma: no cover - exercised only on Windows
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _unlock_handle(handle) -> None:
        """Release the lock taken by :meth:`_lock_handle`."""
        if os.name == "nt":  # pragma: no cover - exercised only on Windows
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def default_lock_path(data_dir: Path | str) -> Path:
    """Return the lock-file path inside ``data_dir``."""
    return Path(data_dir) / LOCK_FILENAME


__all__ = ["LOCK_FILENAME", "SingleInstanceLock", "default_lock_path"]
