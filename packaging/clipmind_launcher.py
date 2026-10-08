"""PyInstaller entry point for the ClipMind desktop app.

PyInstaller needs a plain script as its entry point; ``python -m clipmind`` is
not one. This module mirrors ``clipmind/__main__.py`` and adds the two things a
frozen build needs:

* ``multiprocessing.freeze_support()`` (macOS/Windows correctness), and
* a default sub-command: double-clicking ``ClipMind.app`` runs ``clipmind app``
  rather than an argparse "a subcommand is required" error.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path


def _configure_bundled_model() -> None:
    """Point Hugging Face at a model bundled inside the ``.app``, when present.

    A model-bundled build (``CLIPMIND_BUNDLE_MODEL=1``) ships the
    ``all-MiniLM-L6-v2`` snapshot under ``clipmind_hf/``. Pointing ``HF_HOME``
    there and forcing offline mode makes the first search work with no network
    access at all.
    """
    if not getattr(sys, "frozen", False):
        return
    base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    hub = base / "clipmind_hf" / "hub"
    if hub.is_dir():
        os.environ.setdefault("HF_HOME", str(base / "clipmind_hf"))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def main() -> int:
    """Run the ClipMind CLI and return its exit code."""
    _configure_bundled_model()
    from clipmind.capture.cli import main as cli_main

    # A double-clicked .app passes no arguments; default to the desktop app.
    argv = sys.argv[1:]
    if not argv:
        argv = ["app"]
    return cli_main(argv)


if __name__ == "__main__":  # pragma: no cover - exercised only when frozen
    multiprocessing.freeze_support()
    raise SystemExit(main())
