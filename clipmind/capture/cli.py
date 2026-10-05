"""Command-line interface for ClipMind.

Commands::

    clipmind watch                # run the background clipboard capture loop
    clipmind list                # show recent captures
    clipmind add ...             # push text through the capture pipeline
    clipmind search "<query>"    # hybrid (semantic + keyword) search
    clipmind reindex             # embed entries that have no embedding yet
    clipmind stats               # database summary

``search`` understands time phrases inline (``"react hooks yesterday"``) and
accepts ``--top``, ``--type``, ``--min-score``, ``--explain``, ``--semantic-only``
and ``--keyword-only``.

Run as ``python -m clipmind ...`` or via the installed ``clipmind`` script.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from .. import __version__
from ..config import Config
from ..embeddings.encoder import EmbeddingModelUnavailable
from ..embeddings.service import EmbeddingService, reindex_pending
from ..search.hybrid import HybridOutcome, HybridResult, hybrid_search
from ..storage.db import ClipRecord, Database, utcnow
from .detect import CONTENT_TYPES, TYPE_LABELS
from .monitor import ClipboardMonitor

_TYPE_ALIASES = {
    "all": None,
    "link": "url",
    "links": "url",
    "url": "url",
    "urls": "url",
    "code": "code",
    "text": "text",
    "email": "email",
    "emails": "email",
    "number": "number",
    "numbers": "number",
}


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="clipmind",
        description="Local-first clipboard manager with semantic search.",
    )
    parser.add_argument("--version", action="version", version=f"clipmind {__version__}")
    parser.add_argument(
        "--db",
        metavar="PATH",
        help="Database file to use (default: per-user data directory).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    watch = sub.add_parser("watch", help="Capture new clipboard entries in the background.")
    watch.add_argument(
        "--interval",
        type=float,
        default=None,
        metavar="SECONDS",
        help=f"Poll interval (default: {Config().poll_interval}).",
    )
    watch.add_argument(
        "--once",
        action="store_true",
        help="Poll the clipboard a single time, then exit.",
    )
    watch.add_argument(
        "--capture-initial",
        action="store_true",
        help="Also capture whatever is already on the clipboard at startup.",
    )
    watch.add_argument(
        "--iterations",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N polls (useful for scripted demos).",
    )
    watch.add_argument(
        "--quiet",
        action="store_true",
        help="Only log new captures, not startup/shutdown lines.",
    )
    watch.add_argument(
        "--no-embed",
        action="store_true",
        help="Store entries without computing semantic embeddings.",
    )

    listing = sub.add_parser("list", help="List recent clipboard entries.")
    listing.add_argument(
        "-n", "--limit", type=int, default=None, metavar="N",
        help="Number of entries to show (default: 20).",
    )
    listing.add_argument(
        "-t", "--type", default="all", choices=sorted(_TYPE_ALIASES),
        help="Filter by content type (default: all).",
    )
    listing.add_argument(
        "--json", action="store_true", help="Emit JSON lines instead of a table.",
    )

    add = sub.add_parser("add", help="Store text through the capture pipeline.")
    add.add_argument("text", nargs="*", help="Text to store.")
    add.add_argument(
        "--stdin", action="store_true", help="Read the text from standard input.",
    )
    add.add_argument("--source-app", default=None, help="Override the source application.")

    search = sub.add_parser(
        "search", help="Hybrid (semantic + keyword) search over stored entries."
    )
    search.add_argument("query", help="What you remember about the entry.")
    search.add_argument(
        "--top", type=int, default=None, metavar="N",
        help=f"Maximum results (default: {Config().search_top_k}).",
    )
    search.add_argument(
        "-t", "--type", default="all", choices=sorted(_TYPE_ALIASES),
        help="Restrict results to one content type (default: all).",
    )
    search.add_argument(
        "--min-score", type=float, default=None, metavar="S",
        help=f"Minimum combined score to keep a hit (default: {Config().min_score}).",
    )
    search.add_argument(
        "--explain", action="store_true",
        help="Show the semantic/keyword/final score breakdown for each hit.",
    )
    only = search.add_mutually_exclusive_group()
    only.add_argument(
        "--semantic-only", action="store_true",
        help="Ignore keyword scores (pure vector search).",
    )
    only.add_argument(
        "--keyword-only", action="store_true",
        help="Ignore semantic scores (pure FTS5 BM25 search; no model needed).",
    )
    search.add_argument(
        "--json", action="store_true", help="Emit JSON lines instead of a table.",
    )

    reindex = sub.add_parser(
        "reindex", help="Compute embeddings for entries that do not have one yet."
    )
    reindex.add_argument(
        "--batch", type=int, default=None, metavar="N",
        help=f"Texts per forward pass (default: {Config().embedding_batch_size}).",
    )
    reindex.add_argument(
        "--quiet", action="store_true", help="Suppress per-batch progress output.",
    )

    sub.add_parser("stats", help="Show database summary.")
    return parser


# ------------------------------------------------------------------ formatting


def format_entry(record: ClipRecord, now=None) -> str:
    """Return a single aligned table row for ``record``."""
    badge = TYPE_LABELS.get(record.content_type, record.content_type)[:6]
    app = (record.source_app or "-")[:14]
    local = record.local_time().strftime("%m-%d %H:%M")
    return (
        f"{record.id:>5}  {badge:<6}  {record.relative_time(now):>10}  {local}  "
        f"{app:<14}  {record.preview(64)}"
    )


def format_table(records: Sequence[ClipRecord]) -> str:
    """Render ``records`` as an aligned table with a header row."""
    if not records:
        return "No captured entries yet. Copy something with `clipmind watch` running."
    header = (
        f"{'ID':>5}  {'TYPE':<6}  {'AGE':>10}  {'LOCAL TIME':<11}  "
        f"{'SOURCE APP':<14}  PREVIEW"
    )
    now = utcnow()
    lines = [header, "-" * len(header)]
    lines.extend(format_entry(record, now) for record in records)
    return "\n".join(lines)


def record_to_dict(record: ClipRecord) -> dict[str, object]:
    """Return a JSON-serialisable dict for ``record`` (embedding excluded)."""
    return {
        "id": record.id,
        "content": record.content,
        "content_type": record.content_type,
        "source_app": record.source_app,
        "created_at": record.created_at.isoformat(),
        "content_hash": record.content_hash,
        "char_count": record.char_count,
    }


# --------------------------------------------------------------------- commands


def _resolve_type(value: str) -> Optional[str]:
    """Map a CLI type name (``links``, ``all``...) to a stored content type."""
    return _TYPE_ALIASES[value]


def cmd_watch(args: argparse.Namespace, config: Config) -> int:
    """Run the clipboard capture loop with background embedding.

    New captures are queued for the embedding worker, so the poll loop never
    blocks on the model; on exit the worker is drained before the database is
    closed.
    """
    if args.interval is not None:
        config = config.with_overrides(poll_interval=args.interval)
    if args.quiet:
        logging.getLogger("clipmind").setLevel(logging.WARNING)

    with Database(config=config) as db:
        service = None if args.no_embed else EmbeddingService(db, config=config)
        if service is not None:
            service.start()
        monitor = ClipboardMonitor(
            db,
            config,
            embedding_service=service,
            capture_initial=args.capture_initial,
        )
        try:
            if args.once:
                monitor.poll_once()
                print(f"Polled once. {monitor.stats}")
                return 0
            monitor.run(max_iterations=args.iterations)
            print(f"Capture finished. {monitor.stats}")
        finally:
            if service is not None:
                service.stop()
                stats = service.stats
                print(
                    f"Embeddings: {stats['embedded']} stored, {stats['skipped']} skipped, "
                    f"{stats['failed']} failed"
                )
    return 0


def cmd_list(args: argparse.Namespace, config: Config) -> int:
    """Print recent entries as a table or JSON lines."""
    content_type = _resolve_type(args.type)
    limit = args.limit if args.limit is not None else config.list_limit
    with Database(config=config) as db:
        records = db.list_recent(limit=limit, content_type=content_type)
    if args.json:
        for record in records:
            print(json.dumps(record_to_dict(record), ensure_ascii=False))
        return 0
    print(format_table(records))
    return 0


def cmd_add(args: argparse.Namespace, config: Config) -> int:
    """Store text from argv or stdin through the normal capture pipeline."""
    if args.stdin:
        text = sys.stdin.read()
    else:
        text = " ".join(args.text)
    if not text:
        print("Nothing to add (pass text arguments or --stdin).", file=sys.stderr)
        return 2

    with Database(config=config) as db:
        monitor = ClipboardMonitor(db, config, detect_source_app=False)
        result = monitor.capture(text, source_app=args.source_app)

    if result is None:
        print("Skipped: empty or too long (oversize policy is 'skip').")
        return 0
    if result.inserted and result.record is not None:
        print(
            f"Stored #{result.record.id} [{result.record.content_type}] "
            f"({result.record.char_count} chars)"
        )
        return 0
    if result.reason == "duplicate":
        print(f"Duplicate of #{result.duplicate_of} - skipped.")
        return 0
    print(f"Not stored ({result.reason}).")
    return 0


def cmd_search(args: argparse.Namespace, config: Config) -> int:
    """Run a hybrid search and print ranked results."""
    content_type = _resolve_type(args.type)
    try:
        with Database(config=config) as db:
            outcome = hybrid_search(
                db,
                args.query,
                top_k=args.top if args.top is not None else config.search_top_k,
                config=config,
                content_type=content_type,
                min_score=args.min_score,
                semantic_only=args.semantic_only,
                keyword_only=args.keyword_only,
            )
            db_path = db.path
            indexed_total = db.indexed_fingerprint()[0]
    except EmbeddingModelUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        for rank, result in enumerate(outcome.results, start=1):
            print(
                json.dumps(
                    search_result_to_dict(result, rank, explain=args.explain),
                    ensure_ascii=False,
                )
            )
        return 0

    time_header = _time_header(outcome)
    if time_header:
        print(time_header)

    if not outcome.results:
        print(
            _no_results_message(
                outcome, db_path, indexed_total, args.keyword_only
            )
        )
        return 0

    if args.explain:
        _print_explain_table(outcome)
    else:
        _print_result_table(outcome)

    if outcome.notes and args.explain:
        for note in outcome.notes:
            print(f"note: {note}")
    return 0


def _time_header(outcome: HybridOutcome) -> str:
    """Return a header line describing the parsed time window, if any."""
    query = outcome.query
    if query.ambiguous:
        return f"Time: not filtered (ambiguous: {query.note})"
    if query.has_range:
        return f"Time filter: {query.describe()}"
    return ""


def _print_result_table(outcome: HybridOutcome) -> None:
    """Print the default ranked table (score = final hybrid score)."""
    header = f"{'#':>3}  {'SCORE':>6}  {'AGE':>9}  {'TYPE':<6}  PREVIEW"
    print(header)
    print("-" * len(header))
    now = utcnow()
    for rank, result in enumerate(outcome.results, start=1):
        badge = TYPE_LABELS.get(result.content_type, result.content_type)[:6]
        print(
            f"{rank:>3}  {result.final_score:>6.3f}  {result.relative_time(now):>9}  "
            f"{badge:<6}  {result.preview(64)}"
        )
    print(f"\n{len(outcome.results)} result(s) from {outcome.index_size} indexed entr(y/ies).")


def _print_explain_table(outcome: HybridOutcome) -> None:
    """Print the ranked table with the semantic/keyword/final breakdown."""
    header = (
        f"{'#':>3}  {'FINAL':>6}  {'SEM':>6}  {'KW':>6}  {'COV':>5}  "
        f"{'AGE':>9}  {'TYPE':<6}  PREVIEW"
    )
    print(header)
    print("-" * len(header))
    now = utcnow()
    for rank, result in enumerate(outcome.results, start=1):
        badge = TYPE_LABELS.get(result.content_type, result.content_type)[:6]
        marker = "*" if result.exact_keyword and result.final_score < outcome.threshold else " "
        print(
            f"{rank:>3}  {result.final_score:>6.3f}  {result.semantic_score:>6.3f}  "
            f"{result.keyword_score:>6.3f}  {result.keyword_coverage:>5.2f}  "
            f"{result.relative_time(now):>9}  {badge:<6}{marker} {result.preview(64)}"
        )
    print(
        f"\n{len(outcome.results)} result(s); weights sem={outcome.used_semantic} "
        f"kw={outcome.used_keyword}; threshold {outcome.threshold:.2f}"
    )


def _no_results_message(
    outcome: HybridOutcome,
    db_path: Path,
    indexed_total: int,
    keyword_only: bool,
) -> str:
    """Explain an empty result set (no index at all vs. nothing good enough)."""
    if indexed_total == 0 and not keyword_only:
        return (
            f"No embeddings found in {db_path}.\n"
            "Run `clipmind reindex` to index existing entries; new captures are "
            "indexed automatically while `clipmind watch` runs."
        )
    if outcome.best_final is not None:
        return (
            f"No good matches above {outcome.threshold:.2f} "
            f"(best was {outcome.best_final:.3f}). "
            "Try a lower --min-score or different words."
        )
    return "No matches."


def search_result_to_dict(
    result: HybridResult, rank: int, *, explain: bool = False
) -> dict[str, object]:
    """Return a JSON-serialisable dict for one search hit.

    The base fields are stable for scripting; the score breakdown is only added
    when ``explain`` is true, matching the ``--explain`` CLI flag.
    """
    payload: dict[str, object] = {
        "rank": rank,
        "score": result.final_score,
        "id": result.id,
        "content": result.content,
        "content_type": result.content_type,
        "created_at": result.created_at.isoformat(),
        "source_app": result.source_app,
    }
    if explain:
        payload.update(
            {
                "semantic_score": result.semantic_score,
                "keyword_score": result.keyword_score,
                "keyword_coverage": result.keyword_coverage,
                "bm25": result.bm25,
                "semantic_rank": result.semantic_rank,
                "keyword_rank": result.keyword_rank,
                "exact_keyword": result.exact_keyword,
            }
        )
    return payload


def cmd_reindex(args: argparse.Namespace, config: Config) -> int:
    """Embed every eligible entry that has no embedding yet."""
    def progress(done: int, total: int) -> None:
        if not args.quiet:
            print(f"  {done}/{total}", flush=True)

    try:
        with Database(config=config) as db:
            before = db.count()
            report = reindex_pending(
                db, config=config, progress=progress, batch_size=args.batch
            )
            db_path = db.path
    except EmbeddingModelUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Database: {db_path}")
    print(
        f"Scanned {before} entr(y/ies): {report.missing} without embeddings "
        f"({report.eligible} eligible, {report.skipped} skipped as too short/typed)."
    )
    print(
        f"Embedded {report.embedded}, failed {report.failed} in {report.seconds:.1f}s."
    )
    if report.failed:
        return 1
    return 0


def cmd_stats(args: argparse.Namespace, config: Config) -> int:
    """Print counts per content type and the database location."""
    with Database(config=config) as db:
        total = db.count()
        counts = db.type_counts()
        indexed = db.indexed_fingerprint()[0]
        missing = db.count_missing_embeddings()
        db_path = db.path
    print(f"Database: {db_path}")
    print(f"Total entries: {total}")
    print(f"Embedded: {indexed}  (awaiting embedding: {missing})")
    for content_type in CONTENT_TYPES:
        count = counts.get(content_type, 0)
        if count:
            print(f"  {TYPE_LABELS.get(content_type, content_type):<8} {count}")
    return 0


_COMMANDS = {
    "watch": cmd_watch,
    "list": cmd_list,
    "add": cmd_add,
    "search": cmd_search,
    "reindex": cmd_reindex,
    "stats": cmd_stats,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for ``python -m clipmind`` and the ``clipmind`` script."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = Config.from_env()
    if args.db:
        config = config.with_overrides(db_path=Path(args.db))

    return _COMMANDS[args.command](args, config)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
