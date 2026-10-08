"""Clean shutdown on Ctrl+C / ``SIGTERM``.

Two layers:

* unit tests drive :class:`~clipmind.ui.app.SignalShutdown` and assert the
  shutdown callback runs on the Qt thread (via the poll timer) and that the
  previous handlers are restored;
* a subprocess test runs the real ``clipmind app`` process, sends ``SIGINT``
  and asserts it exits 0, prints no traceback and leaves no instance lock
  behind - the exact bugs reported for Ctrl+C.

Platform note: delivering a *real* ``SIGINT``/``SIGTERM`` is POSIX-only. On
Windows ``os.kill``/``Popen.send_signal`` do not deliver catchable signals (they
call ``TerminateProcess``), so those specific tests are skipped there; the
handler/poll wiring is still covered by the platform-independent tests.

Skipped cleanly when PySide6 / pytest-qt are not installed, like the other UI
tests. ``QT_QPA_PLATFORM=offscreen`` comes from ``conftest.py`` and is inherited
by the subprocess through the environment.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from clipmind.ui.app import SHUTDOWN_SIGNALS, SignalShutdown  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

#: Delivering a real signal is POSIX-only; on Windows these calls kill the
#: process outright instead of invoking a Python handler.
needs_posix_signals = pytest.mark.skipif(
    sys.platform == "win32",
    reason="os.kill/send_signal do not deliver catchable signals on Windows",
)


def test_the_handler_only_records_and_the_timer_runs_the_callback(qtbot) -> None:
    """A recorded request is executed on the Qt thread, never in signal context."""
    calls: list[str] = []
    shutdown = SignalShutdown(lambda: calls.append("shutdown"))
    shutdown.install()
    try:
        shutdown._handle(signal.SIGINT, None)
        assert shutdown.requested == "SIGINT"
        assert calls == []  # nothing runs inside the handler itself
        qtbot.waitUntil(lambda: bool(calls), timeout=5000)
    finally:
        shutdown.uninstall()

    assert calls == ["shutdown"]


@needs_posix_signals
def test_sigint_runs_the_shutdown_callback_and_restores_handlers(qtbot) -> None:
    calls: list[str] = []
    shutdown = SignalShutdown(lambda: calls.append("shutdown"))
    previous = {signum: signal.getsignal(signum) for signum in SHUTDOWN_SIGNALS}
    shutdown.install()
    try:
        # Our handler is in place (bound methods compare equal for the same
        # function and instance).
        assert signal.getsignal(signal.SIGINT) == shutdown._handle
        os.kill(os.getpid(), signal.SIGINT)
        qtbot.waitUntil(lambda: bool(calls), timeout=5000)
    finally:
        shutdown.uninstall()

    assert calls == ["shutdown"]
    # Every signal we touched must be handed back to its previous handler.
    for signum, handler in previous.items():
        assert signal.getsignal(signum) is handler


@needs_posix_signals
def test_sigterm_is_also_handled(qtbot) -> None:
    calls: list[str] = []
    shutdown = SignalShutdown(lambda: calls.append("shutdown"))
    previous = signal.getsignal(signal.SIGTERM)
    shutdown.install()
    try:
        assert signal.getsignal(signal.SIGTERM) == shutdown._handle
        os.kill(os.getpid(), signal.SIGTERM)
        qtbot.waitUntil(lambda: bool(calls), timeout=5000)
    finally:
        shutdown.uninstall()

    assert calls == ["shutdown"]
    assert signal.getsignal(signal.SIGTERM) is previous


def test_install_twice_is_a_no_op_and_uninstall_is_safe() -> None:
    shutdown = SignalShutdown(lambda: None)
    previous = signal.getsignal(signal.SIGINT)
    try:
        first = shutdown.install()
        assert first is shutdown
        shutdown.install()  # idempotent
        assert signal.getsignal(signal.SIGINT) == shutdown._handle
    finally:
        shutdown.uninstall()
    assert signal.getsignal(signal.SIGINT) is previous
    shutdown.uninstall()  # second call must not raise
    assert signal.getsignal(signal.SIGINT) is previous


def _spawn_app(db_path: Path) -> tuple[subprocess.Popen, list[str], threading.Event]:
    """Start ``clipmind app`` and wait for its capture loop to come up.

    Returns:
        ``(process, stderr_lines, ready_event)``. ``ready_event`` is set once the
        monitor logs that it started, which happens *after* the shutdown signal
        handlers are installed.
    """
    env = {
        **os.environ,
        "CLIPMIND_DB": str(db_path),
        "QT_QPA_PLATFORM": "offscreen",
    }
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "clipmind",
            "app",
            "--no-hotkey",
            "--no-tray",
            "--no-embed",
        ],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    lines: list[str] = []
    ready = threading.Event()

    def _read() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            lines.append(line)
            if "capture started" in line:
                ready.set()

    threading.Thread(target=_read, name="clipmind-app-stderr", daemon=True).start()
    return proc, lines, ready


@needs_posix_signals
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_clipmind_app_exits_cleanly_on_a_signal(tmp_path: Path, signum: int) -> None:
    db_path = tmp_path / "data" / "clipmind.db"
    lock_path = db_path.parent / "app.lock"
    proc, lines, ready = _spawn_app(db_path)
    try:
        assert ready.wait(timeout=60), (
            "clipmind app did not start in time; stderr:\n" + "".join(lines)
        )
        assert lock_path.exists(), "instance lock should be held while running"
        proc.send_signal(signum)
        returncode = proc.wait(timeout=30)
    finally:
        if proc.poll() is None:  # pragma: no cover - only on a hang
            proc.kill()
            proc.wait(timeout=10)

    stderr = "".join(lines)
    assert returncode == 0, f"exit code {returncode}; stderr:\n{stderr}"
    assert "Traceback" not in stderr, stderr
    assert "KeyboardInterrupt" not in stderr, stderr
    assert not lock_path.exists(), "the instance lock must be released on shutdown"
