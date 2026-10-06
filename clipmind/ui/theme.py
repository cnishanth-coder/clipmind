"""Light/dark theming for the popup and tray.

The palette tracks the operating system by default (``theme = "system"`` in
``config.toml``) and can be pinned to ``"light"`` or ``"dark"``. Colors are
kept in one place and turned into a Qt style sheet, so the widget code stays
free of hard-coded colours.
"""

from __future__ import annotations

from typing import Dict

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication

#: Named colour roles used by the style sheet.
PALETTES: Dict[str, Dict[str, str]] = {
    "light": {
        "bg": "#f5f5f7",
        "panel": "#ffffff",
        "border": "#d9d9de",
        "text": "#1c1c1e",
        "muted": "#86868b",
        "accent": "#0a84ff",
        "selected": "#e5efff",
        "hover": "#f0f4fb",
        "badge_bg": "#ececf0",
        "badge_text": "#515154",
        "chip_bg": "#e9e9ee",
        "code_bg": "#f4f4f6",
        "shadow": "rgba(0, 0, 0, 0.28)",
    },
    "dark": {
        "bg": "#1c1c1e",
        "panel": "#2c2c2e",
        "border": "#3a3a3c",
        "text": "#f2f2f7",
        "muted": "#98989d",
        "accent": "#0a84ff",
        "selected": "#0a3a66",
        "hover": "#33333a",
        "badge_bg": "#3a3a3c",
        "badge_text": "#d1d1d6",
        "chip_bg": "#3a3a3c",
        "code_bg": "#242426",
        "shadow": "rgba(0, 0, 0, 0.55)",
    },
}

#: Monospace family used for ``code`` previews.
MONOSPACE = '"SF Mono", Menlo, Consolas, "DejaVu Sans Mono", monospace'


def detect_system_theme() -> str:
    """Return ``"light"`` or ``"dark"`` from the application's colour scheme."""
    app = QGuiApplication.instance()
    if app is None:
        return "light"
    hints = getattr(app, "styleHints", lambda: None)()
    scheme: object = None
    if hints is not None and hasattr(hints, "colorScheme"):
        try:
            scheme = hints.colorScheme()
        except Exception:  # pragma: no cover - platform specific
            scheme = None
    if scheme == Qt.ColorScheme.Dark:
        return "dark"
    if scheme == Qt.ColorScheme.Light:
        return "light"
    # Older Qt / offscreen platforms: fall back to the window colour's lightness.
    try:
        if app.palette().window().color().lightness() < 128:
            return "dark"
    except Exception:  # pragma: no cover - defensive
        pass
    return "light"


def resolve_theme(preference: str) -> str:
    """Resolve a ``"system"``/``"light"``/``"dark"`` preference to a palette name."""
    if preference in PALETTES:
        return preference
    return detect_system_theme()


def palette(theme: str) -> Dict[str, str]:
    """Return the colour role mapping for ``theme`` (falling back to light)."""
    return PALETTES.get(theme, PALETTES["light"])


def stylesheet(theme: str) -> str:
    """Build the popup style sheet for ``theme``."""
    c = palette(theme)
    return f"""
    QWidget#PopupRoot {{
        background: {c['panel']};
        border: 1px solid {c['border']};
        border-radius: 12px;
    }}
    QLineEdit#SearchBox {{
        background: transparent;
        border: none;
        color: {c['text']};
        font-size: 20px;
        padding: 14px 16px;
        selection-background-color: {c['accent']};
    }}
    QLabel#Status {{
        color: {c['muted']};
        font-size: 12px;
        padding: 4px 16px 8px 16px;
    }}
    QListWidget#ResultList {{
        background: transparent;
        border: none;
        outline: 0;
        padding: 4px 6px 6px 6px;
    }}
    QListWidget#ResultList::item {{
        border: none;
        margin: 1px 4px;
    }}
    QListWidget#ResultList::item:selected {{
        background: {c['selected']};
        border-radius: 8px;
    }}
    QLabel#Preview {{
        color: {c['text']};
        font-size: 14px;
    }}
    QLabel#PreviewCode {{
        color: {c['text']};
        font-family: {MONOSPACE};
        font-size: 13px;
    }}
    QLabel#Meta {{
        color: {c['muted']};
        font-size: 12px;
    }}
    QLabel#Badge {{
        color: {c['badge_text']};
        background: {c['badge_bg']};
        border-radius: 7px;
        padding: 1px 8px;
        font-size: 11px;
    }}
    QLabel#Score {{
        color: {c['muted']};
        font-family: {MONOSPACE};
        font-size: 11px;
    }}
    QPushButton#Chip {{
        color: {c['badge_text']};
        background: {c['chip_bg']};
        border: none;
        border-radius: 9px;
        padding: 2px 10px;
        font-size: 12px;
    }}
    QPushButton#Chip:hover {{
        background: {c['hover']};
    }}
    QPushButton#Chip[dismissible="true"] {{
        padding-right: 6px;
    }}
    QLabel#Hint {{
        color: {c['muted']};
        font-size: 11px;
        padding: 6px 10px 8px 10px;
    }}
    QLabel#Toast {{
        color: {c['text']};
        background: {c['badge_bg']};
        border-radius: 8px;
        padding: 6px 12px;
        font-size: 12px;
    }}
    """


__all__ = ["MONOSPACE", "PALETTES", "detect_system_theme", "palette", "resolve_theme", "stylesheet"]
