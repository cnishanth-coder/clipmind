"""Turn raw clipboard text into a storable, typed record.

The processor owns the "what do we actually keep?" rules so the monitor and the
storage layer stay simple:

1. Strip NUL bytes and surrounding blank lines (indentation is preserved so
   copied code survives intact).
2. Drop entries that are empty or whitespace-only.
3. Apply the configured oversize policy: truncate to ``Config.max_chars`` or
   skip entirely.
4. Detect the content type and compute the dedup hash over the *stored* text,
   so truncation can never make two different texts collide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..config import Config
from ..storage.db import hash_content
from .detect import detect_content_type


@dataclass(frozen=True, slots=True)
class ProcessedClip:
    """A clipboard entry that passed validation, ready for storage.

    Attributes:
        content: Normalised (and possibly truncated) text.
        content_type: Detected type (``url``/``code``/``email``/``number``/``text``).
        content_hash: SHA-256 of :attr:`content`, used for dedup.
        source_app: Source application name when known.
        truncated: True when the original text was longer than ``max_chars``.
    """

    content: str
    content_type: str
    content_hash: str
    source_app: Optional[str] = None
    truncated: bool = False


class ClipProcessor:
    """Normalise, classify and hash clipboard text."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Create a processor using ``config`` (defaults when omitted)."""
        self.config = config or Config()

    @staticmethod
    def normalize(text: str) -> str:
        """Remove NUL bytes and surrounding blank lines, preserving indentation."""
        return text.replace("\x00", "").strip("\r\n").rstrip()

    def process(
        self,
        raw_text: Optional[str],
        source_app: Optional[str] = None,
    ) -> Optional[ProcessedClip]:
        """Validate and describe ``raw_text``.

        Args:
            raw_text: Text read from the clipboard (may be ``None`` or empty).
            source_app: Name of the application the text came from, if known.

        Returns:
            A :class:`ProcessedClip`, or ``None`` when the text is empty or the
            oversize policy is ``"skip"`` and the text is too long.
        """
        if raw_text is None:
            return None

        content = self.normalize(raw_text)
        if not content.strip():
            return None

        truncated = False
        if len(content) > self.config.max_chars:
            if self.config.oversize_policy == "skip":
                return None
            content = content[: self.config.max_chars]
            truncated = True

        return ProcessedClip(
            content=content,
            content_type=detect_content_type(content),
            content_hash=hash_content(content),
            source_app=source_app,
            truncated=truncated,
        )
