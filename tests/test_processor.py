"""Tests for :mod:`clipmind.capture.processor`."""

from __future__ import annotations

from clipmind.capture.processor import ClipProcessor
from clipmind.config import Config
from clipmind.storage.db import hash_content


def _config(**kwargs) -> Config:
    return Config(db_path=":memory:", **kwargs)


def test_empty_and_none_are_rejected() -> None:
    processor = ClipProcessor(_config())
    assert processor.process(None) is None
    assert processor.process("") is None
    assert processor.process("   \n\t \n") is None


def test_normalize_strips_blank_edges_but_keeps_indentation() -> None:
    raw = "\n\n   def f():\n       return 1\n\n"
    processed = ClipProcessor(_config()).process(raw)
    assert processed is not None
    assert processed.content == "   def f():\n       return 1"
    assert processed.content_type == "code"


def test_nul_bytes_are_removed() -> None:
    processed = ClipProcessor(_config()).process("a\x00b")
    assert processed is not None
    assert processed.content == "ab"


def test_truncate_policy_hashes_stored_text() -> None:
    config = _config(max_chars=10)
    processed = ClipProcessor(config).process("x" * 50)
    assert processed is not None
    assert processed.truncated is True
    assert len(processed.content) == 10
    assert processed.content_hash == hash_content("x" * 10)


def test_skip_policy_drops_long_entries() -> None:
    config = _config(max_chars=10, oversize_policy="skip")
    assert ClipProcessor(config).process("x" * 50) is None


def test_short_entry_is_not_marked_truncated() -> None:
    processed = ClipProcessor(_config(max_chars=10)).process("hello")
    assert processed is not None
    assert processed.truncated is False
    assert processed.content_hash == hash_content("hello")
