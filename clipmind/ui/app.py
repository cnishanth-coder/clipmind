"""The ``clipmind app`` process.

One long-running process owns everything:

* the clipboard watcher (the existing capture pipeline, incl. privacy filters),
* the embedding service, with the model loaded **once** and kept warm,
* the tray / menu-bar icon,
* the global hotkey listener,
* the popup, shown by the hotkey or the tray.

Threading model: the watcher and the embedding worker use plain threads; the
hotkey callback runs on pynput's thread and only emits a Qt signal; search runs
on the controller's worker thread. Every path from a worker back to the UI goes
through a Qt signal, so widgets are only ever touched on the main thread.

``clipmind app --check`` builds the whole object graph, prints a one-screen
summary and exits, which is how the smoke test verifies the wiring headlessly.
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Optional

from PySide6.QtCore import QObject, QUrl, Qt, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QMessageBox, QSystemTrayIcon

from ..config import (
    Config,
    config_file_path,
    default_config_toml,
)
from ..capture.monitor import ClipboardMonitor
from ..embeddings.encoder import EmbeddingModelUnavailable, get_embedder
from ..embeddings.service import EmbeddingService
from ..storage.db import Database
from .clipboard import SuppressingClipboard, make_pyperclip_writer
from .hotkey import (
    HotkeyError,
    HotkeyListener,
    input_monitoring_granted,
    parse_hotkey,
    permission_help,
)
from .popup import PopupWindow, ViewBridge
from .search_backend import HybridSearchBackend
from .single_instance import SingleInstanceLock, default_lock_path
from .tray import TrayIcon
from .view_model import SearchController, UiStatus

logger = logging.getLogger(__name__)

#: ``meta`` key recording that the permission dialog has been shown once.
_PERMISSION_PROMPTED_KEY = "ui_permission_prompted"


class HotkeyBridge(QObject):
    """Qt boundary for the pynput listener thread."""

    activated = Signal()


class ClipMindApplication(QObject):
    """Owns the popup, tray, watcher, embedder and hotkey for one process."""

    def __init__(
        self,
        config: Config,
        *,
        enable_hotkey: bool = True,
        enable_tray: bool = True,
        enable_embedding: bool = True,
    ) -> None:
        """Build the object graph (no threads or timers are started yet)."""
        super().__init__()
        self.config = config
        self.enable_hotkey = enable_hotkey
        self.enable_tray = enable_tray
        self.enable_embedding = enable_embedding

        self.db = Database(config=config)
        self.embedder = get_embedder(config) if enable_embedding else None
        self.embedding_service: Optional[EmbeddingService] = (
            EmbeddingService(db=self.db, embedder=self.embedder, config=config)
            if enable_embedding
            else None
        )
        self.monitor = ClipboardMonitor(
            self.db, config, embedding_service=self.embedding_service
        )

        self._clipboard = SuppressingClipboard(
            make_pyperclip_writer(), self.monitor.remember_clipboard
        )
        self.backend = HybridSearchBackend(
            self.db,
            embedder=self.embedder,
            config=config,
            clipboard=self._clipboard,
        )
        self.bridge = ViewBridge()
        self.controller = SearchController(
            self.backend,
            self.bridge,
            debounce_ms=config.ui_debounce_ms,
            top_k=config.ui_list_limit,
            explain=config.ui_explain,
        )
        self.popup = PopupWindow(self.controller, self.bridge, theme=config.theme)

        self.hotkey_bridge = HotkeyBridge()
        self.hotkey_bridge.activated.connect(
            self.toggle_popup, Qt.ConnectionType.QueuedConnection
        )
        self.hotkey: Optional[HotkeyListener] = None

        self.tray: Optional[TrayIcon] = None
        self.lock: Optional[SingleInstanceLock] = None

        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._model_state = (
            UiStatus.LOADING if enable_embedding else UiStatus.READY
        )
        self._model_message: Optional[str] = None

    # ------------------------------------------------------------------ start

    def start(self) -> None:
        """Start every background piece (the lock must already be held)."""
        self.hide_dock_icon()

        if self.embedding_service is not None:
            self.embedding_service.start()
            self._start_thread(self._warm_model, "clipmind-model")

        # The popup starts in the "loading" state; capture keeps working anyway.
        self.controller.set_status(self._model_state, self._model_message)

        monitor_thread = threading.Thread(
            target=self.monitor.run, args=(self._stop_event,), name="clipmind-watch",
            daemon=True,
        )
        monitor_thread.start()
        self._threads.append(monitor_thread)

        if self.enable_tray:
            self._start_tray()
        if self.enable_hotkey:
            self._start_hotkey()

    def _start_thread(self, target, name: str) -> None:
        """Start a daemon thread and remember it for a clean shutdown."""
        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()
        self._threads.append(thread)

    # -------------------------------------------------------------------- tray

    def _start_tray(self) -> None:
        """Create and show the tray icon when a system tray is available."""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            logger.warning("no system tray available; the tray icon is disabled")
            return
        self.tray = TrayIcon(
            is_paused=self.db.is_paused,
            on_toggle_pause=self.toggle_pause,
            on_open=self.open_popup,
            on_clear_history=self.clear_history,
            on_open_config=self.open_config_file,
            on_quit=self.quit,
            theme=self.config.theme,
        )
        self.tray.show()

    # ------------------------------------------------------------------ hotkey

    def _start_hotkey(self) -> None:
        """Register the global hotkey and prompt for permission when needed."""
        try:
            spec = parse_hotkey(self.config.hotkey)
        except HotkeyError as exc:
            logger.error("invalid hotkey: %s", exc)
            self._show_once(
                "ui_hotkey_error_prompted",
                "ClipMind hotkey",
                f"The configured hotkey {self.config.hotkey!r} is invalid:\n\n"
                f"{exc}\n\nFix it in {config_file_path()}.",
            )
            return

        self.hotkey = HotkeyListener(spec, self.hotkey_bridge.activated.emit)
        started = self.hotkey.start()

        # Prompt when we know the hotkey cannot work: either macOS reports the
        # permission is missing, or pynput refused to register.
        granted = input_monitoring_granted()
        if granted is False or not started:
            self._show_once(
                _PERMISSION_PROMPTED_KEY,
                "ClipMind global hotkey",
                permission_help("ClipMind"),
            )

    def _show_once(self, meta_key: str, title: str, message: str) -> None:
        """Show an informational dialog at most once per database."""
        if self.db.get_meta(meta_key) == "1":
            return
        self.db.set_meta(meta_key, "1")
        QMessageBox.information(None, title, message)

    # ------------------------------------------------------------------- popup

    def open_popup(self) -> None:
        """Show the popup (used by the tray and the hotkey)."""
        self.popup.open_popup()

    def toggle_popup(self) -> None:
        """Toggle popup visibility (the hotkey action)."""
        if self.popup.isVisible():
            self.popup.hide()
        else:
            self.popup.open_popup()

    # ------------------------------------------------------------ tray actions

    def toggle_pause(self) -> bool:
        """Flip the persisted pause flag; return the new state."""
        new_state = not self.db.is_paused()
        self.db.set_paused(new_state)
        logger.info("capture %s", "paused" if new_state else "resumed")
        return new_state

    def clear_history(self) -> int:
        """Delete every stored entry and vacuum the database."""
        removed = self.db.delete_all()
        self.db.vacuum()
        logger.info("cleared %d entr(y/ies)", removed)
        return removed

    def open_config_file(self) -> None:
        """Open ``config.toml`` in the default editor, creating it if needed."""
        path = config_file_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text(default_config_toml(), encoding="utf-8")
        except OSError as exc:
            logger.error("could not create %s: %s", path, exc)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    # ------------------------------------------------------------------ model

    def _warm_model(self) -> None:
        """Load the embedding model once, off the UI thread."""
        assert self.embedder is not None
        try:
            self.embedder.embed_texts(["warmup"])  # triggers the one-time load
        except EmbeddingModelUnavailable as exc:
            self._model_state = UiStatus.ERROR
            self._model_message = str(exc)
            logger.error("embedding model unavailable: %s", exc)
        except Exception as exc:  # pragma: no cover - defensive
            self._model_state = UiStatus.ERROR
            self._model_message = f"embedding model failed: {exc}"
            logger.exception("warming the embedding model failed")
        else:
            self._model_state = UiStatus.READY
            self._model_message = None
            logger.info("embedding model ready")
        self.controller.set_status(self._model_state, self._model_message)

    # ------------------------------------------------------------------ macOS

    def hide_dock_icon(self) -> None:
        """Hide the Dock icon on macOS so ClipMind is a menu-bar app.

        Packaged builds get this from ``LSUIElement`` in ``Info.plist``; when run
        from a script it is set through AppKit. Failure is not fatal.
        """
        if not self.config.hide_dock_icon or sys.platform != "darwin":
            return
        try:  # pragma: no cover - macOS only
            from AppKit import (  # type: ignore[import-not-found]
                NSApplication,
                NSApplicationActivationPolicyAccessory,
            )

            NSApplication.sharedApplication().setActivationPolicy_(
                NSApplicationActivationPolicyAccessory
            )
        except Exception:
            logger.debug("could not hide the Dock icon", exc_info=True)

    # ------------------------------------------------------------------ shutdown

    def quit(self) -> None:
        """Stop everything and quit the Qt event loop."""
        logger.info("shutting down ClipMind")
        if self.hotkey is not None:
            self.hotkey.stop()
        if self.tray is not None:
            self.tray.stop()
            self.tray.hide()
        self._stop_event.set()
        self.controller.close()
        for thread in self._threads:
            thread.join(timeout=3.0)
        if self.embedding_service is not None:
            self.embedding_service.stop(timeout=10.0)
        try:
            self.popup.hide()
            self.popup.deleteLater()
        except Exception:  # pragma: no cover - defensive
            pass
        self.db.close()
        if self.lock is not None:
            self.lock.release()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    # ------------------------------------------------------------------- status

    def status_summary(self) -> list[str]:
        """Return a short human-readable summary of the running configuration."""
        try:
            spec = parse_hotkey(self.config.hotkey)
            hotkey = f"{spec.label} ({'enabled' if self.enable_hotkey else 'disabled'})"
        except HotkeyError as exc:
            hotkey = f"invalid: {exc}"
        granted = input_monitoring_granted()
        if granted is None:
            permission = "unknown (install pyobjc to check)"
        else:
            permission = "granted" if granted else "NOT granted"
        return [
            f"database      : {self.db.path}",
            f"items         : {self.db.count()}",
            f"theme         : {self.config.theme}",
            f"embedding     : {'enabled' if self.enable_embedding else 'disabled'}",
            f"tray          : {'enabled' if self.enable_tray else 'disabled'}"
            f" (available: {QSystemTrayIcon.isSystemTrayAvailable()})",
            f"hotkey        : {hotkey}",
            f"input monitor : {permission}",
        ]


class AlreadyRunning(RuntimeError):
    """Raised when another ClipMind process already holds the instance lock."""

    def __init__(self, pid: int) -> None:
        """Record the pid of the running instance."""
        self.pid = pid
        super().__init__(
            f"ClipMind is already running (pid {pid})." if pid
            else "ClipMind is already running."
        )


def run_app(
    config: Config,
    *,
    enable_hotkey: bool = True,
    enable_tray: bool = True,
    enable_embedding: bool = True,
    check: bool = False,
) -> int:
    """Create the Qt application, start ClipMind and (unless ``check``) run it.

    Args:
        config: Resolved configuration.
        enable_hotkey: Register the global hotkey.
        enable_tray: Create the tray icon.
        enable_embedding: Start the embedding service and warm the model.
        check: Build everything, print a summary and exit 0 without running the
            event loop or any background thread.

    Returns:
        A process exit code. ``2`` means another instance is already running.
    """
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("ClipMind")

    instance = ClipMindApplication(
        config,
        enable_hotkey=enable_hotkey and not check,
        enable_tray=enable_tray,
        enable_embedding=enable_embedding,
    )
    instance.hide_dock_icon()

    lock = SingleInstanceLock(default_lock_path(instance.db.path.parent))
    already = lock.acquire()
    if already is not None:
        print(str(AlreadyRunning(already)), file=sys.stderr)
        instance.db.close()
        return 2
    instance.lock = lock

    if check:
        print("ClipMind app check")
        for line in instance.status_summary():
            print(f"  {line}")
        instance.popup.deleteLater()
        instance.controller.close()
        instance.db.close()
        lock.release()
        return 0

    # Normal run: keep the lock, start the background pieces, enter the loop.
    instance.start()
    try:
        return int(app.exec())
    finally:
        instance.db.close()
        lock.release()


__all__ = ["AlreadyRunning", "ClipMindApplication", "HotkeyBridge", "run_app"]
