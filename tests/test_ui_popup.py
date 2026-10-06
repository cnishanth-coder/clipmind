"""Offscreen smoke tests for the PySide6 layer.

Skipped cleanly when PySide6/pytest-qt are not installed. ``QT_QPA_PLATFORM`` is
forced to ``offscreen`` in ``conftest.py``, so these run without a display and
never open a real window.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from clipmind.ui.popup import PopupWindow, ResultDelegate, ViewBridge  # noqa: E402
from clipmind.ui.theme import palette, resolve_theme, stylesheet  # noqa: E402
from clipmind.ui.tray import TrayIcon, make_icon  # noqa: E402
from clipmind.ui.view_model import (  # noqa: E402
    ResultItem,
    SearchController,
    SearchOutcome,
    SearchView,
    UiStatus,
)

TZ = timezone(timedelta(hours=-4))
NOW = datetime(2026, 10, 6, 12, 0, tzinfo=TZ)


def make_item(item_id: int, content: str, content_type: str = "text") -> ResultItem:
    """Build a result row for the popup."""
    return ResultItem(
        id=item_id,
        content=content,
        content_type=content_type,
        created_at=NOW - timedelta(minutes=item_id),
        source_app="Safari",
    )


class FakeBackend:
    """Backend that returns a fixed set of items."""

    def __init__(self, items) -> None:
        self.items = list(items)
        self.copied: list[str] = []

    def run_search(self, request) -> SearchOutcome:
        return SearchOutcome(items=tuple(self.items))

    def copy_to_clipboard(self, text: str) -> None:
        self.copied.append(text)

    def delete_item(self, item_id: int) -> bool:
        self.items = [item for item in self.items if item.id != item_id]
        return True

    def restore_item(self, item) -> bool:
        return True


def _make_popup(qtbot, items):
    backend = FakeBackend(items)
    bridge = ViewBridge()
    controller = SearchController(backend, bridge, debounce_ms=0, top_k=10)
    popup = PopupWindow(controller, bridge, theme="light")
    qtbot.addWidget(popup)
    return popup, controller, bridge, backend


def test_popup_renders_results_through_the_signal_bridge(qtbot) -> None:
    popup, controller, _bridge, _backend = _make_popup(
        qtbot, [make_item(1, "SELECT 1;", "code"), make_item(2, "https://x.dev", "url")]
    )
    controller.refresh()
    # The worker thread publishes through a Qt signal; wait for it to arrive.
    qtbot.waitUntil(lambda: popup.list.count() == 2, timeout=3000)
    assert popup.list.count() == 2
    assert popup.filter_btn.text() == "All"
    controller.close()


def test_popup_empty_state_text(qtbot) -> None:
    popup, _controller, _bridge, _backend = _make_popup(qtbot, [])
    popup._apply_view(SearchView(rows=(), query="", status=UiStatus.READY))
    assert "No clipboard items" in popup.status.text()
    assert popup.list.count() == 0


def test_popup_applies_selection_and_chips(qtbot) -> None:
    popup, controller, _bridge, _backend = _make_popup(
        qtbot, [make_item(1, "a"), make_item(2, "b")]
    )
    view = SearchView(
        rows=(make_item(1, "a"), make_item(2, "b")),
        query="yesterday react",
        filter="code",
        filter_label="Code",
        time_label="Yesterday",
        explain=True,
        selected_index=1,
    )
    popup._apply_view(view)
    assert popup.list.count() == 2
    assert popup.list.currentRow() == 1
    assert popup.filter_btn.text() == "Code"
    assert popup.time_btn.isHidden() is False
    assert "Yesterday" in popup.time_btn.text()
    assert popup.explain_btn.isChecked() is True
    controller.close()


def test_popup_loading_state_shows_the_model_message(qtbot) -> None:
    popup, controller, _bridge, _backend = _make_popup(qtbot, [])
    popup._apply_view(SearchView(status=UiStatus.LOADING, selected_index=None))
    assert "Loading model" in popup.status.text()
    controller.close()


def test_popup_enter_copies_and_hides(qtbot) -> None:
    popup, controller, _bridge, backend = _make_popup(qtbot, [make_item(1, "copy me")])
    popup.show()
    controller.refresh()
    qtbot.waitUntil(lambda: popup.list.count() == 1, timeout=3000)
    popup._activate()
    assert backend.copied == ["copy me"]
    assert popup.isVisible() is False
    controller.close()


def test_delegate_and_theme_build(qtbot) -> None:
    delegate = ResultDelegate()
    assert delegate.explain is False
    assert "PopupRoot" in stylesheet("dark")
    assert "QLineEdit#SearchBox" in stylesheet("light")


def test_tray_icon_reflects_pause_state(qtbot) -> None:
    state = {"paused": False}

    def toggle() -> bool:
        state["paused"] = not state["paused"]
        return state["paused"]

    tray = TrayIcon(
        is_paused=lambda: state["paused"],
        on_toggle_pause=toggle,
        on_open=lambda: None,
        on_clear_history=lambda: 0,
        on_open_config=lambda: None,
        on_quit=lambda: None,
        theme="light",
    )
    assert tray.pause_action.text() == "Pause capture"
    tray._toggle_pause()
    assert state["paused"] is True
    assert tray.pause_action.text() == "Resume capture"
    tray.stop()


def test_make_icon_is_not_null(qtbot) -> None:
    assert not make_icon(False, "light").isNull()
    assert not make_icon(True, "dark").isNull()


def test_theme_resolution(qtbot) -> None:
    assert resolve_theme("dark") == "dark"
    assert resolve_theme("light") == "light"
    assert resolve_theme("system") in ("light", "dark")
    assert palette("dark")["panel"] != palette("light")["panel"]


def test_hotkey_signal_toggles_the_popup(qtbot, tmp_path) -> None:
    """The hotkey thread only ever emits a signal; it must toggle the popup."""
    from clipmind.config import Config
    from clipmind.ui.app import ClipMindApplication

    instance = ClipMindApplication(
        Config(db_path=tmp_path / "app.sqlite"),
        enable_hotkey=False,
        enable_tray=False,
        enable_embedding=False,
    )
    qtbot.addWidget(instance.popup)
    assert instance.popup.isVisible() is False

    # Emitting the bridge signal is exactly what the pynput thread does.
    instance.hotkey_bridge.activated.emit()
    qtbot.waitUntil(lambda: instance.popup.isVisible(), timeout=3000)

    instance.hotkey_bridge.activated.emit()
    qtbot.waitUntil(lambda: instance.popup.isVisible() is False, timeout=3000)

    instance.controller.close()
    instance.db.close()
