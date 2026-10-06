"""Global hotkey listener.

The listener runs on :mod:`pynput`'s own thread and reports presses through a
plain callback. The Qt layer wraps that callback in a ``QObject`` signal, so the
only thing crossing from the pynput thread to the Qt main thread is a signal
emission - widgets are never touched off the main thread.

This module deliberately imports no Qt: the base test-suite must run without
PySide6 installed, and :func:`parse_hotkey` / the permission helpers are pure
functions that are worth testing on their own.

macOS note: watching for a key combination needs the **Input Monitoring**
permission (System Settings -> Privacy & Security -> Input Monitoring). Without
it, ``pynput`` starts but never receives events. :func:`input_monitoring_granted`
reports the answer when ``pyobjc-framework-Quartz`` is installed; the tray menu
stays a working fallback when the hotkey is unavailable.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)

#: Human-readable modifier names, keyed by the tokens users may type.
_MODIFIERS: dict[str, str] = {
    "cmd": "<cmd>",
    "command": "<cmd>",
    "super": "<cmd>",
    "win": "<cmd>",
    "meta": "<cmd>",
    "ctrl": "<ctrl>",
    "control": "<ctrl>",
    "alt": "<alt>",
    "option": "<alt>",
    "opt": "<alt>",
    "shift": "<shift>",
}

#: Pretty labels for the human-readable form.
_PRETTY = {
    "<cmd>": "Cmd",
    "<ctrl>": "Ctrl",
    "<alt>": "Alt",
    "<shift>": "Shift",
}


class HotkeyError(ValueError):
    """Raised when a configured hotkey string cannot be parsed."""


@dataclass(frozen=True, slots=True)
class HotkeySpec:
    """A parsed hotkey.

    Attributes:
        raw: The string as configured (e.g. ``"cmd+shift+v"``).
        canonical: The ``pynput`` hotkey format (e.g. ``"<cmd>+<shift>+v"``).
        label: Human-readable form for menus/dialogs (e.g. ``"Cmd+Shift+V"``).
    """

    raw: str
    canonical: str
    label: str


def parse_hotkey(raw: str) -> HotkeySpec:
    """Parse ``raw`` into a :class:`HotkeySpec`.

    Args:
        raw: A ``+``-separated hotkey such as ``"cmd+shift+v"`` or
            ``"ctrl+alt+space"``. At least one modifier is required.

    Returns:
        The parsed spec.

    Raises:
        HotkeyError: When the string is empty, has no modifier, or names an
            unknown modifier.
    """
    tokens = [part.strip().lower() for part in (raw or "").replace(" ", "+").split("+")]
    tokens = [token for token in tokens if token]
    if not tokens:
        raise HotkeyError("hotkey must not be empty (e.g. \"cmd+shift+v\")")
    *modifier_tokens, key = tokens
    if not modifier_tokens:
        raise HotkeyError(f"hotkey {raw!r} needs at least one modifier")

    parts: list[str] = []
    labels: list[str] = []
    seen: set[str] = set()
    for token in modifier_tokens:
        mapped = _MODIFIERS.get(token)
        if mapped is None:
            raise HotkeyError(
                f"unknown modifier {token!r} in hotkey {raw!r} "
                "(use cmd, ctrl, alt or shift)"
            )
        if mapped in seen:
            continue
        seen.add(mapped)
        parts.append(mapped)
        labels.append(_PRETTY[mapped])

    parts.append(key)
    # Single-character keys read better upper-cased; named keys title-cased.
    labels.append(key.upper() if len(key) == 1 else key.capitalize())
    return HotkeySpec(raw=raw, canonical="+".join(parts), label="+".join(labels))


def input_monitoring_granted() -> Optional[bool]:
    """Report the macOS Input Monitoring grant.

    Returns:
        ``None`` when it cannot be determined (not macOS, or no pyobjc),
        otherwise whether ClipMind may monitor input events.
    """
    if sys.platform != "darwin":  # pragma: no cover - platform specific
        return True
    try:  # pragma: no cover - requires pyobjc + macOS
        import Quartz  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover
        return None
    try:  # pragma: no cover
        return bool(Quartz.CGPreflightListenEventAccess())
    except Exception:  # pragma: no cover
        return None


def permission_help(app_name: str = "ClipMind") -> str:
    """Return the exact steps to grant the global-hotkey permission.

    Args:
        app_name: Name shown in the permission list; the app itself when
            packaged, or the terminal when ClipMind is started from a shell.
    """
    return (
        f"{app_name} needs the macOS **Input Monitoring** permission to watch "
        "for the global hotkey.\n\n"
        "1. Open  System Settings → Privacy & Security → Input Monitoring.\n"
        f"2. Turn on the switch next to “{app_name}”.\n"
        "   • If you run ClipMind from a terminal, the entry is your terminal "
        "app (e.g. Terminal or iTerm).\n"
        "   • If it is not listed, click “+” and add the ClipMind app (or your "
        "terminal).\n"
        f"3. Quit and reopen {app_name}.\n\n"
        "Until this is granted the hotkey will not fire. You can still open "
        "ClipMind from the menu-bar icon."
    )


class HotkeyListener:
    """Watches for one global hotkey on a background thread.

    Args:
        spec: The parsed hotkey to watch.
        on_activate: Called (on the pynput listener thread) when the hotkey
            fires. It must be cheap and thread-safe - in the app it only emits
            a Qt signal.
    """

    def __init__(self, spec: HotkeySpec, on_activate: Callable[[], None]) -> None:
        """Store the spec and callback; nothing is registered until :meth:`start`."""
        self.spec = spec
        self._on_activate = on_activate
        self._listener = None
        self._error: Optional[str] = None

    @property
    def available(self) -> bool:
        """True while the hotkey listener is running."""
        return self._listener is not None

    @property
    def error(self) -> Optional[str]:
        """Last registration error message, or ``None``."""
        return self._error

    def start(self) -> bool:
        """Register the hotkey.

        Returns:
            True when the listener started. False (with :attr:`error` set) when
            pynput is missing or registration failed - the caller then relies
            on the tray menu as the fallback.
        """
        if self._listener is not None:
            return True
        try:
            from pynput import keyboard
        except Exception as exc:
            self._error = f"pynput is not installed: {exc}"
            logger.warning("global hotkey disabled: %s", self._error)
            return False

        def _fire() -> None:
            try:
                self._on_activate()
            except Exception:  # pragma: no cover - callback must never crash the listener
                logger.exception("hotkey callback failed")

        try:
            self._listener = keyboard.GlobalHotKeys({self.spec.canonical: _fire})
            self._listener.start()
        except Exception as exc:  # pragma: no cover - depends on the OS hook
            self._error = f"could not register hotkey: {exc}"
            self._listener = None
            logger.warning("global hotkey disabled: %s", self._error)
            return False
        logger.info("global hotkey enabled: %s", self.spec.label)
        return True

    def stop(self) -> None:
        """Stop listening (safe to call when never started)."""
        listener, self._listener = self._listener, None
        if listener is None:
            return
        try:
            listener.stop()
        except Exception:  # pragma: no cover - best effort
            logger.debug("stopping the hotkey listener failed", exc_info=True)


__all__ = [
    "HotkeyError",
    "HotkeyListener",
    "HotkeySpec",
    "input_monitoring_granted",
    "parse_hotkey",
    "permission_help",
]
