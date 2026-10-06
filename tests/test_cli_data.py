"""CLI tests for the Phase 4 data-management and config commands."""

from __future__ import annotations

import sys
import tomllib
from datetime import datetime, timedelta, timezone

import pytest

from clipmind.capture.cli import main, parse_duration
from clipmind.config import Config
from clipmind.storage.db import Database

UTC = timezone.utc


@pytest.fixture()
def db_path(tmp_path) -> str:
    return str(tmp_path / "cli-data.db")


class _NotATty:
    """Stand-in for a non-interactive stdin."""

    def isatty(self) -> bool:
        return False

    def read(self) -> str:
        return ""


# ------------------------------------------------------------------ pause/resume


def test_pause_resume_and_status(db_path, capsys) -> None:
    assert main(["--db", db_path, "pause"]) == 0
    assert "paused" in capsys.readouterr().out

    assert main(["--db", db_path, "status"]) == 0
    assert "State: paused" in capsys.readouterr().out

    assert main(["--db", db_path, "resume"]) == 0
    assert "resumed" in capsys.readouterr().out

    assert main(["--db", db_path, "status"]) == 0
    out = capsys.readouterr().out
    assert "State: running" in out
    assert "Retention: disabled" in out


# ------------------------------------------------------------------ delete/clear


def test_delete_missing_id_returns_one(db_path, capsys) -> None:
    assert main(["--db", db_path, "delete", "999"]) == 1
    assert "No entry with id 999" in capsys.readouterr().err


def test_delete_removes_entry_from_fts(db_path, capsys) -> None:
    assert main(["--db", db_path, "add", "order note for testing"]) == 0
    capsys.readouterr()

    assert main(["--db", db_path, "search", "order", "--keyword-only"]) == 0
    assert "order note for testing" in capsys.readouterr().out

    assert main(["--db", db_path, "delete", "1"]) == 0
    capsys.readouterr()

    assert main(["--db", db_path, "search", "order", "--keyword-only"]) == 0
    out = capsys.readouterr().out
    assert "order note for testing" not in out
    assert "No matches" in out


def test_clear_refuses_without_yes_when_not_a_tty(db_path, capsys, monkeypatch) -> None:
    main(["--db", db_path, "add", "keep me"])
    capsys.readouterr()
    monkeypatch.setattr(sys, "stdin", _NotATty())

    assert main(["--db", db_path, "clear"]) == 2
    assert "--yes" in capsys.readouterr().err
    with Database(config=Config(db_path=db_path)) as db:
        assert db.count() == 1


class _ATty:
    """Stand-in for an interactive stdin."""

    def isatty(self) -> bool:
        return True

    def read(self) -> str:
        return ""


def test_clear_prompts_and_proceeds_on_yes(db_path, capsys, monkeypatch) -> None:
    main(["--db", db_path, "add", "delete me"])
    capsys.readouterr()
    monkeypatch.setattr(sys, "stdin", _ATty())
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")

    assert main(["--db", db_path, "clear"]) == 0
    assert "Cleared 1" in capsys.readouterr().out
    with Database(config=Config(db_path=db_path)) as db:
        assert db.count() == 0


def test_clear_prompts_and_aborts_on_no(db_path, capsys, monkeypatch) -> None:
    main(["--db", db_path, "add", "keep me"])
    capsys.readouterr()
    monkeypatch.setattr(sys, "stdin", _ATty())
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")

    assert main(["--db", db_path, "clear"]) == 0
    assert "Aborted" in capsys.readouterr().out
    with Database(config=Config(db_path=db_path)) as db:
        assert db.count() == 1


def test_clear_with_yes_deletes_everything(db_path, capsys) -> None:
    main(["--db", db_path, "add", "one"])
    main(["--db", db_path, "add", "two"])
    capsys.readouterr()

    assert main(["--db", db_path, "clear", "--yes"]) == 0
    assert "Cleared 2" in capsys.readouterr().out
    with Database(config=Config(db_path=db_path)) as db:
        assert db.count() == 0


# ----------------------------------------------------------------------- purge


def test_purge_older_than_removes_old_rows(db_path, capsys) -> None:
    with Database(config=Config(db_path=db_path)) as db:
        db.insert_clip("ancient", content_type="text", created_at=datetime(2020, 1, 1, tzinfo=UTC))
        db.insert_clip("fresh", content_type="text", created_at=datetime.now(UTC))

    assert main(["--db", db_path, "purge", "--older-than", "30d"]) == 0
    assert "Purged 1" in capsys.readouterr().out
    with Database(config=Config(db_path=db_path)) as db:
        assert [r.content for r in db.list_recent()] == ["fresh"]


def test_purge_without_threshold_is_an_error(db_path, capsys) -> None:
    assert main(["--db", db_path, "purge"]) == 2
    assert "--older-than" in capsys.readouterr().err


def test_retention_purges_on_watch_startup(db_path, capsys, monkeypatch, tmp_path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text("retention_days = 1\n", encoding="utf-8")
    monkeypatch.setenv("CLIPMIND_CONFIG", str(config_file))

    with Database(config=Config(db_path=db_path)) as db:
        db.insert_clip("old note", content_type="text", created_at=datetime(2020, 1, 1, tzinfo=UTC))

    assert main(["--db", db_path, "watch", "--once", "--no-embed"]) == 0
    capsys.readouterr()
    with Database(config=Config(db_path=db_path)) as db:
        assert db.count() == 0


# ---------------------------------------------------------------------- config


def test_config_path(db_path, capsys, monkeypatch, tmp_path) -> None:
    target = tmp_path / "custom.toml"
    monkeypatch.setenv("CLIPMIND_CONFIG", str(target))
    assert main(["--db", db_path, "config", "path"]) == 0
    out = capsys.readouterr().out
    assert str(target) in out
    assert "not found" in out


def test_config_show(db_path, capsys) -> None:
    assert main(["--db", db_path, "config", "show"]) == 0
    out = capsys.readouterr().out
    assert "search_top_k = 5" in out
    assert "secret_actions:" in out
    assert "credit_card = skip" in out
    assert "privacy_enabled = true" in out


def test_config_show_template_is_valid_toml(db_path, capsys) -> None:
    assert main(["--db", db_path, "config", "show", "--template"]) == 0
    parsed = tomllib.loads(capsys.readouterr().out)
    assert parsed["search_top_k"] == 5
    assert parsed["secret_actions"]["private_key"] == "skip"


# ------------------------------------------------------------------- duration


@pytest.mark.parametrize(
    "text, expected",
    [
        ("30d", timedelta(days=30)),
        ("5", timedelta(days=5)),
        ("12h", timedelta(hours=12)),
        ("2w", timedelta(weeks=2)),
        ("3 days", timedelta(days=3)),
    ],
)
def test_parse_duration(text: str, expected: timedelta) -> None:
    assert parse_duration(text) == expected


@pytest.mark.parametrize("text", ["", "soon", "d", "-1d", "1 month"])
def test_parse_duration_rejects_bad_input(text: str) -> None:
    with pytest.raises(ValueError):
        parse_duration(text)
