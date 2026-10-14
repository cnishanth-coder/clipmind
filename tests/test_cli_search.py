"""Tests for the ``search``/``reindex`` CLI commands.

The CLI builds its own embedder from configuration, so these tests patch
``get_embedder`` in the two modules that call it. That keeps the commands under
test end to end (argument parsing, database access, formatting) while never
loading torch or touching the network.
"""

from __future__ import annotations

import json
import re

import pytest

from clipmind.capture.cli import main
from clipmind.embeddings.encoder import EmbeddingModelUnavailable
from fakes import FakeEmbedder

DOCS = [
    "SELECT id FROM orders WHERE status = open;",
    "https://react.dev/reference/react/hooks",
    "meeting at 5pm with Rahul",
]

#: ``reindex`` prints one progress line per batch as ``"  <done>/<total>"``.
#: Match the whole line rather than the bare ``"/3"`` substring, which also
#: occurs in the temp database path on some runners (e.g. ``/private/var/...``).
_PROGRESS_LINE = re.compile(r"^\s*(\d+)/(\d+)\s*$")


def _progress_lines(text: str) -> list[str]:
    """Return the ``done/total`` progress lines found in ``text``."""
    return [line for line in text.splitlines() if _PROGRESS_LINE.match(line)]


@pytest.fixture()
def fake_model(monkeypatch):
    """Patch both embedder factories to return one shared deterministic fake."""
    embedder = FakeEmbedder()
    monkeypatch.setattr("clipmind.search.semantic.get_embedder", lambda config=None: embedder)
    monkeypatch.setattr("clipmind.embeddings.service.get_embedder", lambda config=None: embedder)
    return embedder


@pytest.fixture()
def db_path(tmp_path, fake_model) -> str:
    """A database pre-populated with three entries (no embeddings yet)."""
    path = str(tmp_path / "cli.db")
    for doc in DOCS:
        assert main(["--db", path, "add", doc]) == 0
    return path


def test_reindex_embeds_entries_and_reports_counts(db_path, capsys) -> None:
    assert main(["--db", db_path, "reindex"]) == 0
    out = capsys.readouterr().out
    assert "Scanned 3 entr(y/ies)" in out
    assert "Embedded 3, failed 0" in out


def test_reindex_is_idempotent_from_the_cli(db_path, capsys) -> None:
    main(["--db", db_path, "reindex"])
    capsys.readouterr()
    assert main(["--db", db_path, "reindex"]) == 0
    out = capsys.readouterr().out
    assert "0 without embeddings" in out
    assert "Embedded 0, failed 0" in out


def test_reindex_prints_progress_without_quiet(db_path, capsys) -> None:
    assert main(["--db", db_path, "reindex"]) == 0
    lines = _progress_lines(capsys.readouterr().out)
    assert lines, "reindex should print at least one done/total progress line"
    assert lines[-1].strip() == "3/3"


def test_reindex_quiet_suppresses_progress(db_path, capsys) -> None:
    assert main(["--db", db_path, "reindex", "--quiet"]) == 0
    out = capsys.readouterr().out
    assert _progress_lines(out) == []
    # The summary still prints, so an empty run cannot pass as "suppressed".
    assert "Embedded 3, failed 0" in out


def test_reindex_force_reembeds_rows_that_already_have_vectors(db_path, capsys) -> None:
    main(["--db", db_path, "reindex"])
    capsys.readouterr()

    assert main(["--db", db_path, "reindex", "--force"]) == 0
    out = capsys.readouterr().out
    assert "Force reindex: scanned 3 entr(y/ies)" in out
    assert "Embedded 3, failed 0" in out


def test_stats_reports_embedded_count(db_path, capsys) -> None:
    capsys.readouterr()
    main(["--db", db_path, "reindex"])
    capsys.readouterr()
    assert main(["--db", db_path, "stats"]) == 0
    out = capsys.readouterr().out
    assert "Embedded: 3" in out
    assert "awaiting embedding: 0" in out


def test_search_prints_ranked_table(db_path, capsys) -> None:
    main(["--db", db_path, "reindex"])
    capsys.readouterr()

    assert main(["--db", db_path, "search", "react hooks reference"]) == 0
    out = capsys.readouterr().out
    assert "SCORE" in out
    assert "result(s) from 3 indexed" in out
    assert "https://react.dev/reference/react/hooks" in out
    first_data_line = [line for line in out.splitlines() if line.strip().startswith("1")][0]
    # The badge column shows the human label for the stored type ('url' -> 'link').
    assert "link" in first_data_line
    assert "react.dev" in first_data_line


def test_search_without_embeddings_explains_reindex(db_path, capsys) -> None:
    assert main(["--db", db_path, "search", "anything"]) == 0
    out = capsys.readouterr().out
    assert "No embeddings found" in out
    assert "clipmind reindex" in out


def test_search_reports_no_good_matches_above_threshold(db_path, capsys) -> None:
    main(["--db", db_path, "reindex"])
    capsys.readouterr()

    assert main(["--db", db_path, "search", "quantum gardening", "--min-score", "0.95"]) == 0
    out = capsys.readouterr().out
    assert "No good matches above 0.95" in out
    assert "--min-score" in out


def test_search_type_filter(db_path, capsys) -> None:
    main(["--db", db_path, "reindex"])
    capsys.readouterr()

    assert (
        main(["--db", db_path, "search", "order meeting react", "--type", "code", "--min-score", "0.0"])
        == 0
    )
    out = capsys.readouterr().out
    assert "SELECT id FROM orders" in out
    assert "react.dev" not in out


def test_search_top_option_limits_rows(db_path, capsys) -> None:
    main(["--db", db_path, "reindex"])
    capsys.readouterr()

    main(["--db", db_path, "search", "order meeting react", "--top", "1", "--min-score", "0.0", "--json"])
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert len(lines) == 1


def test_search_json_fields(db_path, capsys) -> None:
    main(["--db", db_path, "reindex"])
    capsys.readouterr()

    assert main(["--db", db_path, "search", "react hooks reference", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out.splitlines()[0])
    assert set(payload) == {
        "rank",
        "score",
        "id",
        "content",
        "content_type",
        "created_at",
        "source_app",
    }
    assert payload["rank"] == 1
    assert payload["content"] == "https://react.dev/reference/react/hooks"


def test_search_without_embedding_model_reports_error_cleanly(
    db_path, capsys, monkeypatch
) -> None:
    main(["--db", db_path, "reindex"])  # index with the fake model
    capsys.readouterr()

    def unavailable(config=None):
        raise EmbeddingModelUnavailable("sentence-transformers is not installed. Install the ...")

    monkeypatch.setattr("clipmind.search.semantic.get_embedder", unavailable)

    assert main(["--db", db_path, "search", "react hooks"]) == 1
    captured = capsys.readouterr()
    assert "sentence-transformers is not installed" in captured.err
    assert "Traceback" not in captured.err


def test_reindex_without_embedding_model_reports_error_cleanly(
    db_path, capsys, monkeypatch
) -> None:
    def unavailable(config=None):
        raise EmbeddingModelUnavailable("could not load embedding model 'all-MiniLM-L6-v2'")

    monkeypatch.setattr("clipmind.embeddings.service.get_embedder", unavailable)

    assert main(["--db", db_path, "reindex"]) == 1
    captured = capsys.readouterr()
    assert "error: could not load embedding model" in captured.err
    assert "Traceback" not in captured.err


def test_watch_no_embed_stores_without_embeddings(tmp_path, capsys) -> None:
    path = str(tmp_path / "noembed.db")
    assert main(["--db", path, "watch", "--once", "--no-embed"]) == 0
    assert "Polled once" in capsys.readouterr().out

    main(["--db", path, "add", "entry that stays unembedded"])
    capsys.readouterr()
    main(["--db", path, "stats"])
    assert "awaiting embedding: 1" in capsys.readouterr().out


# --------------------------------------------------- Phase 3 hybrid options


def test_keyword_only_finds_exact_string_without_embeddings(db_path, capsys) -> None:
    assert main(["--db", db_path, "add", "ERR_4021 database connection timeout"]) == 0
    capsys.readouterr()

    # No reindex: the keyword retriever needs no embeddings at all.
    assert main(["--db", db_path, "search", "ERR_4021", "--keyword-only"]) == 0
    out = capsys.readouterr().out
    assert "ERR_4021" in out
    assert "result(s) from 0 indexed" in out


def test_explain_prints_score_breakdown(db_path, capsys) -> None:
    main(["--db", db_path, "reindex"])
    capsys.readouterr()

    assert (
        main(["--db", db_path, "search", "react hooks reference", "--explain", "--min-score", "0.0"])
        == 0
    )
    out = capsys.readouterr().out
    assert "FINAL" in out and "SEM" in out and "KW" in out and "COV" in out
    assert "weights sem=" in out


def test_explain_json_adds_the_breakdown(db_path, capsys) -> None:
    main(["--db", db_path, "reindex"])
    capsys.readouterr()

    assert (
        main(["--db", db_path, "search", "react hooks reference", "--explain", "--json", "--min-score", "0.0"])
        == 0
    )
    payload = json.loads(capsys.readouterr().out.splitlines()[0])
    assert {"semantic_score", "keyword_score", "keyword_coverage", "bm25"} <= set(payload)


def test_semantic_only_skips_keyword(db_path, capsys) -> None:
    main(["--db", db_path, "reindex"])
    capsys.readouterr()

    assert (
        main(["--db", db_path, "search", "react hooks reference", "--semantic-only", "--explain", "--min-score", "0.0"])
        == 0
    )
    out = capsys.readouterr().out
    assert "react.dev" in out
    assert "sem=True kw=False" in out  # semantic-only is reflected in the summary


def test_semantic_only_and_keyword_only_are_mutually_exclusive(db_path) -> None:
    with pytest.raises(SystemExit):
        main(["--db", db_path, "search", "x", "--semantic-only", "--keyword-only"])


def test_time_phrase_shows_a_time_filter_header(db_path, capsys) -> None:
    main(["--db", db_path, "reindex"])
    capsys.readouterr()

    assert (
        main(["--db", db_path, "search", "react hooks reference today", "--min-score", "0.0"])
        == 0
    )
    out = capsys.readouterr().out
    assert "Time filter:" in out
    assert "today" in out


def test_ambiguous_time_phrase_is_reported_not_filtered(db_path, capsys) -> None:
    main(["--db", db_path, "reindex"])
    capsys.readouterr()

    assert main(["--db", db_path, "search", "monday.com link", "--min-score", "0.0"]) == 0
    out = capsys.readouterr().out
    assert "Time: not filtered (ambiguous" in out
