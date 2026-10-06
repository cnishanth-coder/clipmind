"""User interface (Phase 5): popup, tray icon, global hotkey and the app.

Modules:

* :mod:`view_model` - display-free controller: debounce, stale cancellation,
  selection, filters, copy-back and delete/undo. No Qt import.
* :mod:`search_backend` - adapter to the Phase 3 hybrid + time-aware search.
* :mod:`hotkey` - ``pynput`` global hotkey + macOS permission helpers. No Qt.
* :mod:`single_instance` - the ``clipmind app`` process lock.
* :mod:`clipboard` - copy-back that suppresses re-capture.
* :mod:`theme`, :mod:`popup`, :mod:`tray`, :mod:`app` - the PySide6 layer.

Importing this package never imports PySide6; only the Qt-specific modules do,
which keeps the core CLI usable without the optional UI dependencies.
"""

from __future__ import annotations

__all__: list[str] = []
