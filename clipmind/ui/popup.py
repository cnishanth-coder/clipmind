"""Spotlight-style popup window.

The window is deliberately thin: it renders :class:`~clipmind.ui.view_model.SearchView`
snapshots and forwards keystrokes to the controller. No search, filtering or
selection logic lives here.

``ViewBridge`` is the thread boundary: the controller may publish a snapshot
from its search worker, and ``on_view`` simply emits a Qt signal, which Qt
delivers to this window's thread. Widgets are only ever touched on the main
thread.
"""

from __future__ import annotations

import sys
from typing import Dict, Optional

from PySide6.QtCore import QEvent, QObject, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QCursor,
    QFont,
    QGuiApplication,
    QKeyEvent,
    QPainter,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from .theme import MONOSPACE, palette, resolve_theme, stylesheet
from .view_model import ResultItem, SearchController, SearchView, UiStatus

#: Row height in pixels.
ROW_HEIGHT = 40

#: Preferred width of the popup.
POPUP_WIDTH = 680

#: Vertical position of the popup, as a fraction of the available height.
TOP_FRACTION = 0.22


class ViewBridge(QObject):
    """QObject adapter that turns controller snapshots into a Qt signal."""

    viewReady = Signal(object)

    def on_view(self, view: SearchView) -> None:
        """Publish ``view`` (safe to call from any thread)."""
        self.viewReady.emit(view)


class ResultDelegate(QStyledItemDelegate):
    """Paints one result row: badge, preview, age and (optionally) score."""

    def __init__(self, parent: Optional[QObject] = None) -> None:
        """Create the delegate with the light palette."""
        super().__init__(parent)
        self.colors: Dict[str, str] = palette("light")
        self.explain = False

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        """Draw the row from its :class:`ResultItem` payload."""
        row: Optional[ResultItem] = index.data(Qt.ItemDataRole.UserRole)
        if not isinstance(row, ResultItem):
            super().paint(painter, option, index)
            return

        c = self.colors
        rect = option.rect
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        body = rect.adjusted(4, 2, -4, -2)

        if option.state & QStyle.StateFlag.State_Selected:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(c["selected"]))
            painter.drawRoundedRect(QRectF(body), 8, 8)

        pad = 10
        x = body.left() + pad
        center_y = body.center().y()

        # -- type badge ----------------------------------------------------
        badge_font = QFont(option.font)
        badge_font.setPointSizeF(max(9.0, option.font.pointSizeF() - 2))
        painter.setFont(badge_font)
        badge_w = 46
        badge_h = 18
        badge_rect = QRectF(x, center_y - badge_h / 2, badge_w, badge_h)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(c["badge_bg"]))
        painter.drawRoundedRect(badge_rect, 7, 7)
        painter.setPen(QColor(c["badge_text"]))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, row.badge)
        x += badge_w + 10

        # -- right-hand columns -------------------------------------------
        right = body.right() - pad
        age_font = QFont(option.font)
        age_font.setPointSizeF(max(9.0, option.font.pointSizeF() - 2))
        painter.setFont(age_font)
        age_w = 74
        painter.setPen(QColor(c["muted"]))
        painter.drawText(
            QRectF(right - age_w, body.top(), age_w, body.height()),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            row.relative_time(),
        )
        right -= age_w + 8

        if self.explain:
            score_font = QFont(option.font)
            score_font.setPointSizeF(max(9.0, option.font.pointSizeF() - 2))
            score_font.setFamily(MONOSPACE)
            painter.setFont(score_font)
            score_w = 96
            painter.setPen(QColor(c["muted"]))
            painter.drawText(
                QRectF(right - score_w, body.top(), score_w, body.height()),
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                f"{row.final_score:.2f}/{row.semantic_score:.2f}/{row.keyword_score:.2f}",
            )
            right -= score_w + 8

        # -- preview -------------------------------------------------------
        preview_font = QFont(option.font)
        if row.is_code:
            preview_font.setFamily(MONOSPACE)
        painter.setFont(preview_font)
        painter.setPen(QColor(c["text"]))
        available = max(0, int(right - x))
        text = painter.fontMetrics().elidedText(
            row.preview(), Qt.TextElideMode.ElideRight, available
        )
        painter.drawText(
            QRectF(x, body.top(), available, body.height()),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            text,
        )
        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:
        """Return a fixed row height."""
        width = option.rect.width() or POPUP_WIDTH
        return QSize(width, ROW_HEIGHT)


class PopupWindow(QWidget):
    """Frameless, always-on-top search popup.

    Args:
        controller: The display-free controller driving every decision.
        bridge: The :class:`ViewBridge` the controller publishes to.
        theme: ``"system"``, ``"light"`` or ``"dark"``.
        parent: Optional parent widget.
    """

    def __init__(
        self,
        controller: SearchController,
        bridge: ViewBridge,
        *,
        theme: str = "system",
        parent: Optional[QWidget] = None,
    ) -> None:
        """Build the widget tree and connect the bridge."""
        super().__init__(parent)
        self._controller = controller
        self._theme_pref = theme
        self._theme = resolve_theme(theme)
        self._view = SearchView()
        self._ids: tuple[int, ...] = ()
        self._syncing = False
        self._close_on_focus_loss = True

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setWindowTitle("ClipMind")
        self.setFixedWidth(POPUP_WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)

        self._build_ui()
        self._resize_to_rows(0)  # a sane geometry before the first result
        self._apply_theme()

        bridge.viewReady.connect(self._apply_view)
        self.list.currentRowChanged.connect(self._on_row_changed)
        self.list.itemActivated.connect(lambda _item: self._activate())

        self._maybe_watch_color_scheme()

    # ------------------------------------------------------------------- build

    def _build_ui(self) -> None:
        """Create the search box, chips, list and footer."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)  # room for the drop shadow

        self.root = QFrame(self)
        self.root.setObjectName("PopupRoot")
        shadow = QGraphicsDropShadowEffect(self.root)
        shadow.setBlurRadius(36)
        shadow.setOffset(0, 10)
        shadow.setColor(QColor(0, 0, 0, 120))
        self.root.setGraphicsEffect(shadow)
        outer.addWidget(self.root)

        inner = QVBoxLayout(self.root)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(0)

        self.search = QLineEdit(self.root)
        self.search.setObjectName("SearchBox")
        self.search.setPlaceholderText("Search your clipboard…")
        self.search.setClearButtonEnabled(True)
        self.search.installEventFilter(self)
        self.search.textChanged.connect(self._controller.set_query)
        inner.addWidget(self.search)

        chip_row = QHBoxLayout()
        chip_row.setContentsMargins(16, 0, 16, 0)
        chip_row.setSpacing(8)

        self.filter_btn = QPushButton("All", self.root)
        self.filter_btn.setObjectName("Chip")
        self.filter_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.filter_btn.setToolTip("Cycle the type filter (Tab)")
        self.filter_btn.clicked.connect(self._controller.cycle_filter)
        chip_row.addWidget(self.filter_btn)

        self.time_btn = QPushButton("", self.root)
        self.time_btn.setObjectName("Chip")
        self.time_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.time_btn.setToolTip("Detected time filter — click to dismiss")
        self.time_btn.clicked.connect(self._controller.dismiss_time_filter)
        self.time_btn.hide()
        chip_row.addWidget(self.time_btn)

        self.explain_btn = QPushButton("Explain", self.root)
        self.explain_btn.setObjectName("Chip")
        self.explain_btn.setCheckable(True)
        self.explain_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.explain_btn.setToolTip("Show the semantic/keyword score per row")
        self.explain_btn.clicked.connect(self._controller.toggle_explain)
        chip_row.addWidget(self.explain_btn)

        chip_row.addStretch(1)
        inner.addLayout(chip_row)

        self.status = QLabel("", self.root)
        self.status.setObjectName("Status")
        inner.addWidget(self.status)

        self.list = QListWidget(self.root)
        self.list.setObjectName("ResultList")
        self.list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.delegate = ResultDelegate(self.list)
        self.list.setItemDelegate(self.delegate)
        inner.addWidget(self.list, 1)

        self.toast = QLabel("", self.root)
        self.toast.setObjectName("Toast")
        self.toast.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.toast.hide()
        inner.addWidget(self.toast, 0, Qt.AlignmentFlag.AlignHCenter)

        mod = "⌘" if sys.platform == "darwin" else "Ctrl+"
        self.hint = QLabel(
            f"↑↓ move · ⏎ copy · {mod}⌫ delete · {mod}Z undo · Tab filter · Esc close",
            self.root,
        )
        self.hint.setObjectName("Hint")
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.addWidget(self.hint)

    # ------------------------------------------------------------------ theming

    def _maybe_watch_color_scheme(self) -> None:
        """Re-theme when the OS switches light/dark, when Qt supports it."""
        hints = QApplication.instance().styleHints() if QApplication.instance() else None
        signal = getattr(hints, "colorSchemeChanged", None) if hints else None
        if signal is not None:
            try:
                signal.connect(self._on_color_scheme_changed)
            except Exception:  # pragma: no cover - platform specific
                pass

    def _on_color_scheme_changed(self, *_args: object) -> None:
        """Refresh the palette after a system light/dark switch."""
        self._theme = resolve_theme(self._theme_pref)
        self._apply_theme()
        self._apply_view(self._view)

    def _apply_theme(self) -> None:
        """Apply the current palette to the widget tree and the delegate."""
        self.setStyleSheet(stylesheet(self._theme))
        self.delegate.colors = palette(self._theme)

    # ----------------------------------------------------------------- rendering

    def _apply_view(self, view: SearchView) -> None:
        """Render a controller snapshot (main thread only)."""
        self._view = view

        ids = tuple(row.id for row in view.rows)
        if ids != self._ids:
            self._ids = ids
            self.list.clear()
            for row in view.rows:
                item = QListWidgetItem()
                item.setData(Qt.ItemDataRole.UserRole, row)
                item.setSizeHint(QSize(0, ROW_HEIGHT))
                self.list.addItem(item)

        self._syncing = True
        try:
            if view.selected_index is None:
                self.list.setCurrentRow(-1)
            else:
                self.list.setCurrentRow(view.selected_index)
        finally:
            self._syncing = False

        self.filter_btn.setText(view.filter_label)
        self.explain_btn.setChecked(view.explain)

        if view.time_label:
            self.time_btn.setText(f"⏱ {view.time_label}  ✕")
            self.time_btn.show()
        else:
            self.time_btn.hide()

        self.status.setText(self._status_text(view))
        self.status.setVisible(bool(self.status.text()))

        self.toast.setText("Deleted. Press ⌘Z to undo." if view.has_undo else "")
        self.toast.setVisible(view.has_undo)

        self._resize_to_rows(len(view.rows))

    def _status_text(self, view: SearchView) -> str:
        """Return the status line for ``view`` (empty when there is nothing to say)."""
        if view.status is UiStatus.LOADING:
            return "Loading model… search works once it is ready."
        if view.status is UiStatus.ERROR:
            return f"⚠ {view.status_message or 'Search unavailable.'}"
        if view.searching:
            return "Searching…"
        if not view.rows:
            if not view.query.strip():
                return "No clipboard items yet. Copy something while ClipMind runs."
            if view.time_note:
                return view.time_note
            return "No matches."
        return ""

    def _resize_to_rows(self, count: int) -> None:
        """Grow/shrink the popup with the number of visible rows."""
        visible = max(1, min(count, 8))
        chrome = 24 * 2 + 58 + 34 + 40 + 26  # margins + search + chips + footer
        height = chrome + (visible * ROW_HEIGHT) + (0 if count else 18)
        self.setFixedHeight(int(height))

    # --------------------------------------------------------------- interaction

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802 (Qt API)
        """Intercept popup keys before the search box consumes them."""
        if obj is self.search and event.type() == QEvent.Type.KeyPress:
            if self._handle_key(event):
                return True
        return super().eventFilter(obj, event)

    def _handle_key(self, event: QKeyEvent) -> bool:
        """Handle navigation keys; return True when the event was consumed."""
        key = event.key()
        mods = event.modifiers()
        ctrl = bool(mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier))

        if key == Qt.Key.Key_Escape:
            self.hide()
            return True
        if key == Qt.Key.Key_Up:
            self._controller.move_selection(-1)
            return True
        if key == Qt.Key.Key_Down:
            self._controller.move_selection(1)
            return True
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._activate()
            return True
        if key == Qt.Key.Key_Tab:
            self._controller.cycle_filter()
            return True
        if key == Qt.Key.Key_Backtab:
            self._controller.cycle_filter()
            return True
        if key == Qt.Key.Key_Delete or (key == Qt.Key.Key_Backspace and ctrl):
            self._controller.delete_selected()
            return True
        if key == Qt.Key.Key_Z and ctrl:
            self._controller.undo_delete()
            return True
        return False

    def _activate(self) -> None:
        """Copy the selected row back and close the popup."""
        if self._controller.activate_selected() is not None:
            self.hide()

    def _on_row_changed(self, row: int) -> None:
        """Mirror mouse/list selection back into the controller."""
        if self._syncing:
            return
        self._controller.select_index(row if row >= 0 else None)

    # ------------------------------------------------------------------- window

    def open_popup(self) -> None:
        """Reset to the most recent items, center, show and focus."""
        self.search.blockSignals(True)
        self.search.clear()
        self.search.blockSignals(False)
        self._controller.set_query("")
        self._controller.refresh()
        self._center()
        self.show()
        self.raise_()
        self.activateWindow()
        self.search.setFocus(Qt.FocusReason.OtherFocusReason)

    def _center(self) -> None:
        """Place the popup horizontally centered, near the top of the screen."""
        screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
        if screen is None:  # pragma: no cover - no screen (rare)
            return
        area = screen.availableGeometry()
        x = area.left() + (area.width() - self.width()) // 2
        y = area.top() + int(area.height() * TOP_FRACTION)
        self.move(max(area.left(), x), max(area.top(), y))

    def event(self, event: QEvent) -> bool:  # noqa: N802 (Qt API)
        """Close on focus loss."""
        if (
            event.type() == QEvent.Type.WindowDeactivate
            and self._close_on_focus_loss
            and self.isVisible()
        ):
            self.hide()
        return super().event(event)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        """Hide instead of destroying the window."""
        self.hide()
        event.ignore()


__all__ = ["POPUP_WIDTH", "ROW_HEIGHT", "PopupWindow", "ResultDelegate", "ViewBridge"]
