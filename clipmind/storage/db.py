"""SQLite storage layer for ClipMind.

Public surface:

* :class:`ClipRecord` - an immutable view of one stored clipboard entry.
* :class:`Database`  - connection management, schema creation, inserts with
  duplicate detection, and read helpers.

The database file is the only place ClipMind persists data. It lives under the
per-user data directory by default (see :mod:`clipmind.config`).
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

from ..config import Config

#: Bump when the on-disk schema changes in an incompatible way.
SCHEMA_VERSION = 1

#: ``meta`` key holding the persisted pause flag (``"1"`` when paused).
PAUSED_META_KEY = "paused"

#: Columns of the ``items`` table, in display order.
_ITEM_COLUMNS = (
    "id, content, content_type, source_app, created_at, content_hash, embedding"
)


def utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC :class:`datetime`."""
    return datetime.now(timezone.utc)


def utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (millisecond precision)."""
    return utcnow().isoformat(timespec="milliseconds")


def hash_content(content: str) -> str:
    """Return the SHA-256 hex digest used for exact duplicate detection."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ClipRecord:
    """A single stored clipboard entry.

    Attributes:
        id: Auto-incrementing primary key.
        content: Stored text (possibly truncated to ``Config.max_chars``).
        content_type: One of ``url``, ``code``, ``email``, ``number`` or ``text``.
        source_app: Best-effort name of the application the text came from.
        created_at: Timezone-aware UTC timestamp of first capture.
        content_hash: SHA-256 of :attr:`content`; unique across the table.
        embedding: Serialised embedding vector, or ``None`` until Phase 2 fills it.
    """

    id: int
    content: str
    content_type: str
    source_app: Optional[str]
    created_at: datetime
    content_hash: str
    embedding: Optional[bytes] = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ClipRecord":
        """Build a record from a :mod:`sqlite3` row."""
        return cls(
            id=row["id"],
            content=row["content"],
            content_type=row["content_type"],
            source_app=row["source_app"],
            created_at=datetime.fromisoformat(row["created_at"]),
            content_hash=row["content_hash"],
            embedding=row["embedding"],
        )

    @property
    def char_count(self) -> int:
        """Number of characters stored for this entry."""
        return len(self.content)

    def preview(self, max_len: int = 80) -> str:
        """Return a single-line preview of the content, at most ``max_len`` chars."""
        collapsed = " ".join(self.content.split())
        if len(collapsed) <= max_len:
            return collapsed
        return collapsed[: max_len - 1] + "\u2026"

    def relative_time(self, now: Optional[datetime] = None) -> str:
        """Return a short human-readable age such as ``"5m ago"`` or ``"2d ago"``."""
        now = now or utcnow()
        delta = now - self.created_at
        seconds = int(delta.total_seconds())
        if seconds < 0:
            return "just now"
        if seconds < 60:
            return "just now" if seconds < 5 else f"{seconds}s ago"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes}m ago"
        hours = minutes // 60
        if hours < 24:
            return f"{hours}h ago"
        days = hours // 24
        if days < 7:
            return f"{days}d ago"
        return self.created_at.astimezone().strftime("%Y-%m-%d")

    def local_time(self) -> datetime:
        """Return :attr:`created_at` converted to the local timezone."""
        return self.created_at.astimezone()


@dataclass(frozen=True, slots=True)
class InsertResult:
    """Outcome of an insert attempt.

    Attributes:
        inserted: True when a new row was written.
        reason: One of ``"inserted"``, ``"duplicate"``, ``"empty"``,
            ``"too_long"``, ``"invalid"``, ``"ignored"`` (source-app ignore
            list) or ``"secret"`` (a skip-configured secret was detected).
        record: The stored record when ``inserted`` is True, else ``None``.
        duplicate_of: Id of the existing row when ``reason == "duplicate"``.
    """

    inserted: bool
    reason: str
    record: Optional[ClipRecord] = None
    duplicate_of: Optional[int] = None


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS items (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    content       TEXT    NOT NULL,
    content_type  TEXT    NOT NULL DEFAULT 'text',
    source_app    TEXT,
    created_at    TEXT    NOT NULL,
    content_hash  TEXT    NOT NULL,
    embedding     BLOB
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_items_content_hash ON items(content_hash);
CREATE INDEX IF NOT EXISTS idx_items_created_at ON items(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_items_content_type ON items(content_type);

-- External-content FTS5 index: rows live in `items`, the index mirrors content.
CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
    content,
    content='items',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS items_ai AFTER INSERT ON items BEGIN
    INSERT INTO items_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS items_ad AFTER DELETE ON items BEGIN
    INSERT INTO items_fts(items_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS items_au AFTER UPDATE ON items BEGIN
    INSERT INTO items_fts(items_fts, rowid, content) VALUES ('delete', old.id, old.content);
    INSERT INTO items_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class Database(AbstractContextManager["Database"]):
    """Thread-safe wrapper around the ClipMind SQLite database.

    A single connection is shared and guarded by a re-entrant lock so the
    Phase 2 embedding worker thread can write without racing the clipboard
    poller. ``sqlite3`` connections are not thread-safe by default, hence the
    explicit lock and ``check_same_thread=False``.
    """

    def __init__(
        self,
        db_path: Optional[Path | str] = None,
        *,
        config: Optional[Config] = None,
    ) -> None:
        """Open (creating if necessary) the ClipMind database.

        Args:
            db_path: Explicit database path. ``":memory:"`` opens an in-memory
                database, which is handy for tests.
            config: Optional configuration object; used for ``db_path`` and
                the ``max_chars``/``oversize_policy`` insert rules.
        """
        self.config = config or Config()
        if db_path is None:
            db_path = self.config.db_path
        if isinstance(db_path, str) and db_path == ":memory:":
            self.path = Path(":memory:")
            self._in_memory = True
        else:
            self.path = Path(db_path)
            self._in_memory = False
            self.path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.path), check_same_thread=False, timeout=30.0
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    # ------------------------------------------------------------------ setup

    def _init_schema(self) -> None:
        """Create tables, indexes, the FTS5 mirror and triggers if missing."""
        with self._lock:
            self._conn.executescript(_SCHEMA_SQL)
            self._conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self._conn.commit()

    # ------------------------------------------------------------------ writes

    def insert_clip(
        self,
        content: str,
        content_type: str = "text",
        source_app: Optional[str] = None,
        created_at: Optional[datetime | str] = None,
        content_hash: Optional[str] = None,
        embedding: Optional[bytes] = None,
        *,
        dedup: Optional[bool] = None,
        allow_empty: bool = False,
    ) -> InsertResult:
        """Insert a clipboard entry, skipping exact duplicates.

        Args:
            content: The text to store.
            content_type: Content classification (``url``/``code``/``email``/
                ``number``/``text``).
            source_app: Name of the source application, if known.
            created_at: Capture timestamp; defaults to now (UTC). Naive
                datetimes are interpreted as UTC.
            content_hash: Pre-computed SHA-256 digest; computed when omitted.
            embedding: Optional serialised embedding blob (Phase 2).
            dedup: Override ``Config.dedup`` for this insert.
            allow_empty: Store empty strings (off by default).

        Returns:
            An :class:`InsertResult` describing what happened.
        """
        if content is None:
            return InsertResult(False, "invalid")
        if content == "" and not allow_empty:
            return InsertResult(False, "empty")

        if len(content) > self.config.max_chars:
            if self.config.oversize_policy == "skip":
                return InsertResult(False, "too_long")
            content = content[: self.config.max_chars]

        digest = content_hash or hash_content(content)

        timestamp = self._coerce_timestamp(created_at)
        is_dedup = self.config.dedup if dedup is None else dedup

        with self._lock:
            if is_dedup:
                existing = self._conn.execute(
                    "SELECT id FROM items WHERE content_hash = ?", (digest,)
                ).fetchone()
                if existing is not None:
                    return InsertResult(False, "duplicate", duplicate_of=existing["id"])

            try:
                cursor = self._conn.execute(
                    """
                    INSERT INTO items
                        (content, content_type, source_app, created_at, content_hash, embedding)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (content, content_type, source_app, timestamp, digest, embedding),
                )
            except sqlite3.IntegrityError:
                # Lost a race against another writer on the unique hash index.
                row = self._conn.execute(
                    "SELECT id FROM items WHERE content_hash = ?", (digest,)
                ).fetchone()
                self._conn.rollback()
                return InsertResult(False, "duplicate", duplicate_of=row["id"] if row else None)

            self._conn.commit()
            record = self.get_by_id(cursor.lastrowid)
            return InsertResult(True, "inserted", record=record)

    @staticmethod
    def _coerce_timestamp(value: Optional[datetime | str]) -> str:
        """Normalise ``value`` to a UTC ISO-8601 string."""
        if value is None:
            return utcnow_iso()
        if isinstance(value, str):
            return value
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat(timespec="milliseconds")

    def delete_all(self) -> int:
        """Delete every stored entry and return the number of rows removed."""
        with self._lock:
            count = self._conn.execute("SELECT COUNT(*) AS n FROM items").fetchone()["n"]
            self._conn.execute("DELETE FROM items")
            self._conn.execute("INSERT INTO items_fts(items_fts) VALUES ('rebuild')")
            self._conn.commit()
            return int(count)

    def delete_by_id(self, item_id: int) -> bool:
        """Delete one entry by id; return True when a row was removed.

        The ``items_ad`` trigger removes the row from the FTS5 index, so a
        deleted entry stops matching keyword/hybrid search immediately.
        """
        with self._lock:
            cursor = self._conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
            self._conn.commit()
            return cursor.rowcount > 0

    def delete_older_than(self, cutoff: datetime | str) -> int:
        """Delete every entry created strictly before ``cutoff``.

        Args:
            cutoff: Boundary instant; naive datetimes are treated as UTC.

        Returns:
            The number of rows deleted. The FTS index is kept in sync by the
            ``items_ad`` trigger on each deleted row.
        """
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM items WHERE created_at < ?",
                (self._coerce_timestamp(cutoff),),
            )
            self._conn.commit()
            return int(cursor.rowcount)

    def vacuum(self) -> None:
        """Rebuild the database file, reclaiming space after bulk deletes.

        ``VACUUM`` rewrites the whole file and can be slow on a large database,
        so callers run it only after a bulk delete (purge/clear), never per row.
        """
        with self._lock:
            self._conn.commit()
            self._conn.execute("VACUUM")
            self._conn.commit()

    # --------------------------------------------------------------- meta flags

    def get_meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Return the value stored under ``key`` in the ``meta`` table."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        """Insert or update a ``meta`` value."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            self._conn.commit()

    def is_paused(self) -> bool:
        """Return True when the persisted pause flag is set."""
        return self.get_meta(PAUSED_META_KEY, "0") == "1"

    def set_paused(self, paused: bool) -> None:
        """Persist the pause flag so ``watch`` can honour it live."""
        self.set_meta(PAUSED_META_KEY, "1" if paused else "0")

    def set_embedding(self, item_id: int, embedding: bytes) -> bool:
        """Attach a serialised embedding to an existing row (Phase 2 hook)."""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE items SET embedding = ? WHERE id = ?", (embedding, item_id)
            )
            self._conn.commit()
            return cursor.rowcount > 0

    # ------------------------------------------------------------------- reads

    def get_by_id(self, item_id: int) -> Optional[ClipRecord]:
        """Return the record with ``item_id``, or ``None``."""
        with self._lock:
            row = self._conn.execute(
                f"SELECT {_ITEM_COLUMNS} FROM items WHERE id = ?", (item_id,)
            ).fetchone()
        return ClipRecord.from_row(row) if row else None

    def get_by_hash(self, content_hash: str) -> Optional[ClipRecord]:
        """Return the record with the given content hash, or ``None``."""
        with self._lock:
            row = self._conn.execute(
                f"SELECT {_ITEM_COLUMNS} FROM items WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
        return ClipRecord.from_row(row) if row else None

    def list_recent(
        self,
        limit: int = 20,
        content_type: Optional[str] = None,
        *,
        since: Optional[datetime | str] = None,
        until: Optional[datetime | str] = None,
        order: str = "desc",
    ) -> list[ClipRecord]:
        """Return the most recent entries, newest first by default.

        Args:
            limit: Maximum number of rows to return (``<= 0`` means no limit).
            content_type: Optional content-type filter, e.g. ``"code"``.
            since: Only include entries created at or after this instant.
            until: Only include entries created strictly before this instant.
            order: ``"desc"`` (default) or ``"asc"``.

        Returns:
            A list of :class:`ClipRecord` objects.
        """
        clauses: list[str] = []
        params: list[object] = []
        if content_type:
            clauses.append("content_type = ?")
            params.append(content_type)
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(self._coerce_timestamp(since))
        if until is not None:
            clauses.append("created_at < ?")
            params.append(self._coerce_timestamp(until))

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        direction = "ASC" if order.lower() == "asc" else "DESC"
        sql = f"SELECT {_ITEM_COLUMNS} FROM items {where} ORDER BY created_at {direction}, id {direction}"
        if limit and limit > 0:
            sql += " LIMIT ?"
            params.append(limit)

        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [ClipRecord.from_row(row) for row in rows]

    def search_keywords(
        self,
        query: str,
        limit: int = 20,
        *,
        content_type: Optional[str] = None,
        since: Optional[datetime | str] = None,
        until: Optional[datetime | str] = None,
    ) -> list[tuple[ClipRecord, float]]:
        """Full-text search the FTS5 index, ranked by BM25.

        This is the keyword half of the Phase 3 hybrid search.

        Args:
            query: A *sanitised* FTS5 MATCH expression. Callers must quote the
                tokens (see :func:`clipmind.search.keyword.sanitize_fts_query`)
                so user punctuation can never be parsed as FTS5 syntax.
            limit: Maximum number of results.
            content_type: Optional content-type filter.
            since: Only include entries created at or after this instant.
            until: Only include entries created strictly before this instant.

        Returns:
            ``(record, bm25_score)`` pairs ordered by descending relevance.
            ``bm25()`` returns negative values where more negative is better,
            so this method orders ascending; callers normalise to ``0..1``.
        """
        clauses = ["items_fts MATCH ?"]
        params: list[object] = [query]
        if content_type:
            clauses.append("i.content_type = ?")
            params.append(content_type)
        if since is not None:
            clauses.append("i.created_at >= ?")
            params.append(self._coerce_timestamp(since))
        if until is not None:
            clauses.append("i.created_at < ?")
            params.append(self._coerce_timestamp(until))
        params.append(limit)

        sql = f"""
            SELECT {', '.join('i.' + c.strip() for c in _ITEM_COLUMNS.split(','))},
                   bm25(items_fts) AS score
            FROM items_fts
            JOIN items AS i ON i.id = items_fts.rowid
            WHERE {' AND '.join(clauses)}
            ORDER BY score
            LIMIT ?
        """
        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [(ClipRecord.from_row(row), float(row["score"])) for row in rows]

    def count(self, content_type: Optional[str] = None) -> int:
        """Return the number of stored entries, optionally filtered by type."""
        sql = "SELECT COUNT(*) AS n FROM items"
        params: Sequence[object] = ()
        if content_type:
            sql += " WHERE content_type = ?"
            params = (content_type,)
        with self._lock:
            return int(self._conn.execute(sql, params).fetchone()["n"])

    def type_counts(self) -> dict[str, int]:
        """Return a mapping of content type to number of stored entries."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT content_type, COUNT(*) AS n FROM items GROUP BY content_type ORDER BY n DESC"
            ).fetchall()
        return {row["content_type"]: int(row["n"]) for row in rows}

    def iter_missing_embeddings(self) -> Iterable[ClipRecord]:
        """Return records that still need an embedding, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_ITEM_COLUMNS} FROM items WHERE embedding IS NULL ORDER BY id"
            ).fetchall()
        return [ClipRecord.from_row(row) for row in rows]

    def count_missing_embeddings(self) -> int:
        """Return how many rows still have a ``NULL`` embedding."""
        with self._lock:
            return int(
                self._conn.execute(
                    "SELECT COUNT(*) AS n FROM items WHERE embedding IS NULL"
                ).fetchone()["n"]
            )

    def list_indexed(
        self,
        content_type: Optional[str] = None,
        *,
        since: Optional[datetime | str] = None,
        until: Optional[datetime | str] = None,
    ) -> list[ClipRecord]:
        """Return every record that has an embedding, newest first.

        This is the brute-force vector index source: the caller decodes each
        ``record.embedding`` BLOB into a matrix.

        Args:
            content_type: Optional content-type filter.
            since: Only include entries created at or after this instant.
            until: Only include entries created strictly before this instant.
        """
        sql = f"SELECT {_ITEM_COLUMNS} FROM items WHERE embedding IS NOT NULL"
        params: list[object] = []
        if content_type:
            sql += " AND content_type = ?"
            params.append(content_type)
        if since is not None:
            sql += " AND created_at >= ?"
            params.append(self._coerce_timestamp(since))
        if until is not None:
            sql += " AND created_at < ?"
            params.append(self._coerce_timestamp(until))
        sql += " ORDER BY created_at DESC, id DESC"
        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [ClipRecord.from_row(row) for row in rows]

    def indexed_fingerprint(
        self,
        content_type: Optional[str] = None,
        *,
        since: Optional[datetime | str] = None,
        until: Optional[datetime | str] = None,
    ) -> tuple[int, int]:
        """Return ``(row_count, max_id)`` of the embedded rows a filter selects.

        Used as a cheap cache key: any insert changes ``max_id`` and any delete
        changes ``row_count``, so a stable value means the index is unchanged.
        The time bounds are part of the fingerprint because they select a
        different subset of rows.
        """
        clauses = ["embedding IS NOT NULL"]
        params: list[object] = []
        if content_type:
            clauses.append("content_type = ?")
            params.append(content_type)
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(self._coerce_timestamp(since))
        if until is not None:
            clauses.append("created_at < ?")
            params.append(self._coerce_timestamp(until))
        sql = (
            "SELECT COUNT(*) AS n, COALESCE(MAX(id), 0) AS m FROM items "
            f"WHERE {' AND '.join(clauses)}"
        )
        with self._lock:
            row = self._conn.execute(sql, tuple(params)).fetchone()
        return int(row["n"]), int(row["m"])

    # ------------------------------------------------------------------ lifecyle

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()

    def __exit__(self, *exc_info: object) -> None:
        """Close the connection when leaving a ``with`` block."""
        self.close()
