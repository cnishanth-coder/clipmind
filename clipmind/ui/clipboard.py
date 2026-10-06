"""Copying an entry back to the system clipboard.

Copying a stored entry back is the popup's main action, and it must not feed
straight back into the capture pipeline as a *new* entry (or bump an existing
duplicate). Two independent guards prevent that:

1. :class:`SuppressingClipboard` writes the text and immediately tells the
   ``ClipboardMonitor`` that this exact value is the current baseline, so its
   next poll sees "unchanged" and stores nothing.
2. The database already deduplicates by content hash, so even if a poll slipped
   through before the baseline was updated, the row is rejected as a duplicate
   and its timestamp is not touched.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class SuppressingClipboard:
    """Writes to the system clipboard and suppresses the matching re-capture.

    Args:
        write: Callable that puts text on the real system clipboard
            (``pyperclip.copy`` in production).
        remember: Callable that tells the capture monitor the given text is now
            the expected clipboard baseline (``ClipboardMonitor.remember_clipboard``).
            When omitted, only the write happens.
    """

    def __init__(
        self,
        write: Callable[[str], None],
        remember: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Store the injected callables."""
        self._write = write
        self._remember = remember
        self.last_copied: Optional[str] = None

    def copy(self, text: str) -> None:
        """Put ``text`` on the clipboard and mark it as already-seen."""
        self._write(text)
        # Update the baseline *after* the write so the monitor only ever sees
        # the value once it is actually on the clipboard.
        if self._remember is not None:
            try:
                self._remember(text)
            except Exception:  # pragma: no cover - suppression is best effort
                logger.exception("could not update the clipboard baseline")
        self.last_copied = text


def make_pyperclip_writer() -> Callable[[str], None]:
    """Return a writer backed by :mod:`pyperclip` (imported lazily)."""
    import pyperclip

    def write(text: str) -> None:
        pyperclip.copy(text)

    return write


__all__ = ["SuppressingClipboard", "make_pyperclip_writer"]
