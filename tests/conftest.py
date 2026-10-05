"""Shared pytest configuration.

* Makes ``clipmind`` and the local ``fakes`` module importable from the repo root.
* Gates ``@pytest.mark.slow`` tests (the ones that load the real
   ``all-MiniLM-L6-v2`` model) behind ``--runslow``, so the default suite stays
   fast and offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for path in (ROOT, TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--runslow`` flag."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests that download/load the real embedding model",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``slow`` tests unless ``--runslow`` was passed."""
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --runslow (loads the real embedding model)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
