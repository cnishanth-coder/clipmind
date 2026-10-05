"""Tests for :mod:`clipmind.capture.cli`."""

from __future__ import annotations

import json

import pytest

from clipmind.capture.cli import main


@pytest.fixture()
def db_path(tmp_path):
    return str(tmp_path / "cli.db")


def test_add_then_list_then_stats(db_path, capsys) -> None:
    assert main(["--db", db_path, "add", "SELECT id", "FROM", "orders"]) == 0
    assert "Stored" in capsys.readouterr().out

    assert main(["--db", db_path, "list"]) == 0
    listing = capsys.readouterr().out
    assert "SELECT id FROM orders" in listing
    assert "code" in listing

    assert main(["--db", db_path, "stats"]) == 0
    stats = capsys.readouterr().out
    assert "Total entries: 1" in stats


def test_add_detects_duplicates(db_path, capsys) -> None:
    main(["--db", db_path, "add", "same value"])
    capsys.readouterr()
    assert main(["--db", db_path, "add", "same value"]) == 0
    assert "Duplicate" in capsys.readouterr().out


def test_add_reads_stdin(db_path, capsys, monkeypatch) -> None:
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("line from stdin"))
    assert main(["--db", db_path, "add", "--stdin"]) == 0
    capsys.readouterr()
    main(["--db", db_path, "list", "--json"])
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["content"] == "line from stdin"


def test_add_rejects_empty_input(db_path, capsys) -> None:
    assert main(["--db", db_path, "add"]) == 2
    assert "Nothing to add" in capsys.readouterr().err


def test_list_type_filter_and_json(db_path, capsys) -> None:
    main(["--db", db_path, "add", "https://react.dev/learn"])
    main(["--db", db_path, "add", "a prose sentence about nothing"])
    capsys.readouterr()

    main(["--db", db_path, "list", "--type", "links", "--json"])
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert len(lines) == 1
    assert lines[0]["content_type"] == "url"


def test_list_empty_database_message(db_path, capsys) -> None:
    main(["--db", db_path, "list"])
    assert "No captured entries yet" in capsys.readouterr().out


def test_watch_once_without_clipboard_does_not_crash(db_path, capsys) -> None:
    # A broken/absent clipboard is tolerated; the command still exits 0.
    assert main(["--db", db_path, "watch", "--once"]) == 0
    assert "Polled once" in capsys.readouterr().out
