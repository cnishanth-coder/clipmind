"""ClipMind - a local-first desktop clipboard manager with semantic search.

Phase 1 provides clipboard capture, SQLite storage with FTS5 indexing, exact
duplicate detection and a small CLI. Nothing in this package ever touches the
network.
"""

from __future__ import annotations

__version__ = "0.5.0"
__all__ = ["__version__"]
