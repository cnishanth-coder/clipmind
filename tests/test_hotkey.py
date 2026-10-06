"""Tests for hotkey parsing and the macOS permission helpers.

No Qt and no real key listener: the listener itself is only started in a real
session. These cover the pure logic that a bad ``config.toml`` would otherwise
break at startup.
"""

from __future__ import annotations

import pytest

from clipmind.ui.hotkey import (
    HotkeyError,
    HotkeyListener,
    input_monitoring_granted,
    parse_hotkey,
    permission_help,
)


@pytest.mark.parametrize(
    "raw, canonical, label",
    [
        ("cmd+shift+v", "<cmd>+<shift>+v", "Cmd+Shift+V"),
        ("ctrl+shift+v", "<ctrl>+<shift>+v", "Ctrl+Shift+V"),
        ("Ctrl+Alt+Space", "<ctrl>+<alt>+space", "Ctrl+Alt+Space"),
        ("cmd+shift+return", "<cmd>+<shift>+return", "Cmd+Shift+Return"),
        ("option+cmd+k", "<alt>+<cmd>+k", "Alt+Cmd+K"),
        ("super+shift+p", "<cmd>+<shift>+p", "Cmd+Shift+P"),
    ],
)
def test_parse_hotkey(raw: str, canonical: str, label: str) -> None:
    spec = parse_hotkey(raw)
    assert spec.canonical == canonical
    assert spec.label == label
    assert spec.raw == raw


def test_duplicate_modifiers_are_collapsed() -> None:
    assert parse_hotkey("cmd+cmd+shift+c").canonical == "<cmd>+<shift>+c"


@pytest.mark.parametrize("raw", ["", "   ", "v", "cmd+", "hyper+v"])
def test_invalid_hotkeys_are_rejected(raw: str) -> None:
    with pytest.raises(HotkeyError):
        parse_hotkey(raw)


def test_permission_help_names_the_app_and_the_setting() -> None:
    text = permission_help("ClipMind")
    assert "ClipMind" in text
    assert "Input Monitoring" in text
    assert "Privacy & Security" in text


def test_input_monitoring_granted_never_raises() -> None:
    # Returns True on non-macOS, a bool or None on macOS (pyobjc optional).
    assert input_monitoring_granted() in (True, False, None)


def test_listener_reports_unavailable_without_pynput(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("pynput"):
            raise ImportError("no pynput here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    listener = HotkeyListener(parse_hotkey("cmd+shift+v"), lambda: None)
    assert listener.start() is False
    assert listener.available is False
    assert listener.error is not None
    listener.stop()  # must be safe when never started
