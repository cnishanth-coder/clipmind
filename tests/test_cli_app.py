"""``clipmind app --check`` smoke test.

Runs the whole app wiring headlessly (offscreen Qt) without starting threads or
the event loop, and asserts the summary is printed. Skipped when PySide6 is not
installed, matching the rest of the UI tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from clipmind.capture.cli import main  # noqa: E402


@pytest.fixture()
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the app at a throwaway database (never the user's real one)."""
    db_path = tmp_path / "ui-db.sqlite"
    monkeypatch.setenv("CLIPMIND_DB", str(db_path))
    return db_path


def test_app_check_runs_headless(isolated_db: Path, capsys) -> None:
    rc = main(["app", "--check", "--no-hotkey", "--no-tray", "--no-embed"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "ClipMind app check" in out
    assert str(isolated_db) in out
    assert isolated_db.exists()


def test_app_check_reports_an_invalid_hotkey(
    isolated_db: Path, tmp_path: Path, capsys, monkeypatch
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('hotkey = "hyper+v"\n', encoding="utf-8")
    monkeypatch.setenv("CLIPMIND_CONFIG", str(config_path))

    rc = main(["app", "--check", "--no-tray", "--no-embed"])
    out = capsys.readouterr().out
    # An invalid hotkey is reported but must not stop the app from starting.
    assert rc == 0
    assert "invalid" in out
