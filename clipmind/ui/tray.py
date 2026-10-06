"""System tray (menu-bar) icon.

The tray owns no business logic: it calls back into the app for every action.
The Pause/Resume item and the icon reflect the *persisted* pause flag, refreshed
by a timer, so a ``clipmind pause`` run in a terminal is visible here straight
away.

On macOS this is a menu-bar item. Hiding the Dock icon is handled in
``clipmind.ui.app`` (``LSUIElement`` behaviour); see the README.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QMenu, QMessageBox, QSystemTrayIcon

from .theme import palette, resolve_theme

#: How often the tray re-reads the persisted pause flag (milliseconds).
REFRESH_INTERVAL_MS = 2000


def make_icon(paused: bool, theme: str = "light") -> QIcon:
    """Draw the tray icon: a clipboard, greyed and slashed when paused.

    Args:
        paused: Whether capture is currently paused.
        theme: ``"light"`` or ``"dark"``; controls the stroke colour so the
            icon stays legible in the menu bar.

    Returns:
        A freshly painted :class:`QIcon`.
    """
    c = palette(theme)
    size = 44
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    stroke = QColor(c["muted"]) if paused else QColor(c["text"])
    pen = QPen(stroke)
    pen.setWidth(3)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRoundedRect(QRectF(11, 9, 22, 29), 5, 5)
    painter.drawRoundedRect(QRectF(16, 4, 12, 9), 3, 3)

    if paused:
        slash = QPen(QColor(c["muted"]))
        slash.setWidth(4)
        slash.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(slash)
        painter.drawLine(QPointF(7, 37), QPointF(37, 7))

    painter.end()
    return QIcon(pixmap)


class TrayIcon(QSystemTrayIcon):
    """Menu-bar icon with Open/Pause/Clear/Config/Quit actions.

    Args:
        is_paused: Returns whether capture is paused (read from the database).
        on_toggle_pause: Toggles the persisted pause flag; returns the new state.
        on_open: Opens the popup.
        on_clear_history: Clears all entries; returns the number removed.
        on_open_config: Opens ``config.toml``.
        on_quit: Quits the application.
        theme: ``"system"``, ``"light"`` or ``"dark"``.
        parent: Optional parent object.
    """

    def __init__(
        self,
        *,
        is_paused: Callable[[], bool],
        on_toggle_pause: Callable[[], bool],
        on_open: Callable[[], None],
        on_clear_history: Callable[[], int],
        on_open_config: Callable[[], None],
        on_quit: Callable[[], None],
        theme: str = "system",
        parent: Optional[object] = None,
    ) -> None:
        """Build the menu, icon and refresh timer."""
        super().__init__(parent)
        self._is_paused = is_paused
        self._on_toggle_pause = on_toggle_pause
        self._on_clear_history = on_clear_history
        self._theme_pref = theme
        self._theme = resolve_theme(theme)

        self.setToolTip("ClipMind")
        menu = QMenu()

        self.open_action = QAction("Open ClipMind", menu)
        self.open_action.triggered.connect(lambda _=False: on_open())
        menu.addAction(self.open_action)

        menu.addSeparator()

        self.pause_action = QAction("Pause capture", menu)
        self.pause_action.triggered.connect(lambda _=False: self._toggle_pause())
        menu.addAction(self.pause_action)

        self.clear_action = QAction("Clear history…", menu)
        self.clear_action.triggered.connect(lambda _=False: self._clear_history())
        menu.addAction(self.clear_action)

        self.config_action = QAction("Open config file", menu)
        self.config_action.triggered.connect(lambda _=False: on_open_config())
        menu.addAction(self.config_action)

        menu.addSeparator()

        self.quit_action = QAction("Quit ClipMind", menu)
        self.quit_action.triggered.connect(lambda _=False: on_quit())
        menu.addAction(self.quit_action)

        menu.aboutToShow.connect(self._refresh_state)
        self.setContextMenu(menu)
        self._menu = menu

        # Activate on left-click on platforms that support it.
        self.activated.connect(self._on_activated)

        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_INTERVAL_MS)
        self._timer.timeout.connect(self._refresh_state)
        self._timer.start()

        self._refresh_state()

    # ------------------------------------------------------------------ actions

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        """Open the popup on a left click / double click."""
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self.open_action.trigger()

    def _toggle_pause(self) -> None:
        """Toggle the persisted flag and refresh the label/icon."""
        self._on_toggle_pause()
        self._refresh_state()

    def _clear_history(self) -> None:
        """Confirm, then clear every stored entry."""
        answer = QMessageBox.question(
            None,
            "Clear ClipMind history",
            "Delete every stored clipboard entry? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Yes:
            removed = self._on_clear_history()
            self.showMessage(
                "ClipMind",
                f"Cleared {removed} entr{'y' if removed == 1 else 'ies'}.",
                QSystemTrayIcon.MessageIcon.Information,
                3000,
            )

    # ------------------------------------------------------------------- state

    def _refresh_state(self) -> None:
        """Reflect the persisted pause flag in the label and icon."""
        paused = bool(self._is_paused())
        self.pause_action.setText("Resume capture" if paused else "Pause capture")
        self.setIcon(make_icon(paused, self._theme))

    def set_theme(self, theme: str) -> None:
        """Update the resolved theme and repaint the icon."""
        self._theme_pref = theme
        self._theme = resolve_theme(theme)
        self._refresh_state()

    def stop(self) -> None:
        """Stop the refresh timer (call before hiding)."""
        self._timer.stop()


__all__ = ["REFRESH_INTERVAL_MS", "TrayIcon", "make_icon"]
