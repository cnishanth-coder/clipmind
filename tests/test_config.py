"""Tests for ``config.toml`` loading, validation and environment overrides."""

from __future__ import annotations

from pathlib import Path

import pytest

from clipmind.config import (
    Config,
    ConfigError,
    config_file_path,
    default_config_toml,
    load_config_file,
)


def _write(tmp_path, content: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_missing_file_means_defaults(tmp_path) -> None:
    assert load_config_file(tmp_path / "nope.toml") == {}


def test_default_template_parses_and_loads(tmp_path) -> None:
    path = _write(tmp_path, default_config_toml())
    data = load_config_file(path)  # must not raise
    assert data["poll_interval"] == 0.5
    assert data["search_top_k"] == 5
    assert "retention_days" not in data  # commented out in the template
    assert data["secret_actions"]["credit_card"] == "skip"
    # Phase 5 keys are top-level: they must not land inside [secret_actions].
    assert data["theme"] == "system"
    assert data["hotkey"]
    assert data["ui_debounce_ms"] == 150
    assert data["ui_explain"] is False

    cfg = Config.load(path, use_env=False)
    assert cfg.search_top_k == 5
    assert cfg.privacy_enabled is True
    assert cfg.theme == "system"


def test_valid_overrides(tmp_path) -> None:
    path = _write(
        tmp_path,
        "\n".join(
            [
                "search_top_k = 9",
                "retention_days = 7",
                'db_path = "/tmp/clipmind-test.db"',
                'ignore_source_apps = ["Mail", "Notes"]',
                "[secret_actions]",
                'credit_card = "mask"',
                "",
            ]
        ),
    )
    cfg = Config.load(path, use_env=False)
    assert cfg.search_top_k == 9
    assert cfg.retention_days == 7
    assert cfg.db_path == Path("/tmp/clipmind-test.db")
    assert cfg.ignore_source_apps == ("Mail", "Notes")
    # A partial table only overrides the listed category.
    assert cfg.secret_actions == {"credit_card": "mask"}
    assert cfg.secret_actions.get("password", "skip") == "skip"


def test_retention_zero_disables(tmp_path) -> None:
    path = _write(tmp_path, "retention_days = 0\n")
    assert load_config_file(path)["retention_days"] is None


def test_unknown_key_is_reported(tmp_path) -> None:
    path = _write(tmp_path, "nonsense = 1\n")
    with pytest.raises(ConfigError) as excinfo:
        load_config_file(path)
    message = str(excinfo.value)
    assert "nonsense" in message
    assert "unknown key" in message


def test_invalid_toml_is_reported(tmp_path) -> None:
    path = _write(tmp_path, "this is not = = valid\n")
    with pytest.raises(ConfigError) as excinfo:
        load_config_file(path)
    assert "not valid TOML" in str(excinfo.value)


@pytest.mark.parametrize(
    "content, needle",
    [
        ("poll_interval = -1\n", "poll_interval"),
        ("poll_interval = true\n", "poll_interval"),
        ("max_chars = 0\n", "max_chars"),
        ("oversize_policy = \"nope\"\n", "oversize_policy"),
        ("min_score = 2.0\n", "min_score"),
        ("week_starts_on = 9\n", "week_starts_on"),
        ("day_part_morning = [1, 2, 3]\n", "day_part_morning"),
        ("day_part_night = [21, 99]\n", "day_part_night"),
        ("retention_days = -5\n", "retention_days"),
        ("retention_days = \"soon\"\n", "retention_days"),
        ("ignore_source_apps = [1, 2]\n", "ignore_source_apps"),
        ("privacy_enabled = \"yes\"\n", "privacy_enabled"),
        ("[secret_actions]\naws_access_key = \"delete\"\n", "aws_access_key"),
        ("secret_actions = 3\n", "secret_actions"),
        ('theme = "neon"\n', "theme"),
        ("theme = 1\n", "theme"),
        ('hotkey = ""\n', "hotkey"),
        ("hotkey = 5\n", "hotkey"),
        ("ui_list_limit = 0\n", "ui_list_limit"),
        ("ui_debounce_ms = -1\n", "ui_debounce_ms"),
        ('ui_explain = "yes"\n', "ui_explain"),
        ("hide_dock_icon = 1\n", "hide_dock_icon"),
    ],
)
def test_invalid_values_are_reported(tmp_path, content: str, needle: str) -> None:
    path = _write(tmp_path, content)
    with pytest.raises(ConfigError) as excinfo:
        load_config_file(path)
    assert needle in str(excinfo.value)


def test_env_overrides_layer_on_top_of_file(tmp_path, monkeypatch) -> None:
    path = _write(tmp_path, "search_top_k = 9\nmax_chars = 123\n")
    monkeypatch.setenv("CLIPMIND_TOP_K", "11")
    cfg = Config.load(path, use_env=True)
    assert cfg.search_top_k == 11  # env wins over the file
    assert cfg.max_chars == 123  # untouched file value survives


def test_env_only(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CLIPMIND_RETENTION_DAYS", "14")
    monkeypatch.setenv("CLIPMIND_PRIVACY", "0")
    monkeypatch.setenv("CLIPMIND_IGNORE_APPS", "Mail, Notes")
    cfg = Config.load(tmp_path / "missing.toml", use_env=True)
    assert cfg.retention_days == 14
    assert cfg.privacy_enabled is False
    assert cfg.ignore_source_apps == ("Mail", "Notes")


def test_phase5_env_overrides(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CLIPMIND_HOTKEY", "ctrl+alt+k")
    monkeypatch.setenv("CLIPMIND_THEME", "dark")
    monkeypatch.setenv("CLIPMIND_UI_LIST_LIMIT", "30")
    monkeypatch.setenv("CLIPMIND_UI_DEBOUNCE_MS", "250")
    monkeypatch.setenv("CLIPMIND_UI_EXPLAIN", "1")
    cfg = Config.load(tmp_path / "missing.toml", use_env=True)
    assert cfg.hotkey == "ctrl+alt+k"
    assert cfg.theme == "dark"
    assert cfg.ui_list_limit == 30
    assert cfg.ui_debounce_ms == 250
    assert cfg.ui_explain is True


def test_config_rejects_a_bad_theme_directly() -> None:
    with pytest.raises(ValueError):
        Config(db_path=":memory:", theme="neon")


def test_invalid_bool_env_is_reported(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CLIPMIND_PRIVACY", "maybe")
    with pytest.raises(ConfigError):
        Config.load(tmp_path / "missing.toml", use_env=True)


def test_config_file_path_honours_env(monkeypatch) -> None:
    monkeypatch.setenv("CLIPMIND_CONFIG", "/tmp/custom/clipmind.toml")
    assert config_file_path() == Path("/tmp/custom/clipmind.toml")


def test_config_file_path_default_is_under_home(monkeypatch) -> None:
    monkeypatch.delenv("CLIPMIND_CONFIG", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert config_file_path() == Path.home() / ".config" / "clipmind" / "config.toml"


def test_default_config_matches_dataclass_defaults() -> None:
    cfg = Config(db_path=":memory:")
    assert cfg.privacy_enabled is True
    assert cfg.retention_days is None
    assert cfg.secret_actions["credit_card"] == "skip"
    assert cfg.secret_actions["aws_access_key"] == "mask"
    assert "1Password" in cfg.ignore_source_apps
    assert cfg.theme == "system"
    assert cfg.ui_debounce_ms == 150
    assert cfg.ui_list_limit == 20
    assert cfg.hide_dock_icon is True
    assert cfg.hotkey

