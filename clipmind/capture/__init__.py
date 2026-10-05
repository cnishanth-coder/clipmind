"""Clipboard capture: polling, processing and CLI entry points."""

from __future__ import annotations

from .detect import CONTENT_TYPES, detect_content_type
from .monitor import ClipboardMonitor, ClipboardReader, PyperclipReader
from .processor import ClipProcessor, ProcessedClip
from .source_app import frontmost_app_name

__all__ = [
    "CONTENT_TYPES",
    "ClipboardMonitor",
    "ClipboardReader",
    "ClipProcessor",
    "ProcessedClip",
    "PyperclipReader",
    "detect_content_type",
    "frontmost_app_name",
]
