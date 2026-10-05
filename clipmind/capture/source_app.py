"""Best-effort detection of the application that owns the clipboard.

There is no portable, dependency-free way to know which app copied some text.
ClipMind therefore degrades gracefully:

* macOS: uses ``AppKit`` (pyobjc) to read the frontmost application when the
  optional dependency is installed.
* any platform: honours the ``CLIPMIND_SOURCE_APP`` environment variable, which
  is what the test-suite and the demo script use.
* otherwise: returns ``None`` and the ``source_app`` column stays empty.

This is the hook the Phase 4 ignore-list ("never capture from 1Password") will
build on.
"""

from __future__ import annotations

import os
import sys
from typing import Optional


def frontmost_app_name() -> Optional[str]:
    """Return the name of the current frontmost application, or ``None``.

    Returns:
        A short application name such as ``"Safari"``, or ``None`` when it
        cannot be determined on this platform.
    """
    override = os.environ.get("CLIPMIND_SOURCE_APP")
    if override:
        return override
    if sys.platform == "darwin":  # pragma: no cover - requires pyobjc + GUI
        return _macos_frontmost_app()
    return None


def _macos_frontmost_app() -> Optional[str]:  # pragma: no cover - macOS GUI only
    """Read the frontmost app via AppKit; return ``None`` when unavailable."""
    try:
        from AppKit import NSWorkspace  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return None
        name = app.localizedName()
        return str(name) if name else None
    except Exception:
        return None
