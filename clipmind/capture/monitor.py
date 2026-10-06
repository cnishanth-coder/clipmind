"""Background clipboard polling.

:class:`ClipboardMonitor` reads the system clipboard every
``Config.poll_interval`` seconds, filters unchanged values with an in-memory
check, then hands the text to :class:`~clipmind.capture.processor.ClipProcessor`
and stores the result via :class:`~clipmind.storage.db.Database`.

The reader is injected, so tests and the demo script can drive the monitor
without touching a real clipboard.

Phase 4 layers on, in order: the persisted pause flag (checked every poll so a
running watcher honours ``pause``/``resume`` live), the source-app ignore list,
and the secret policy. When a secret is masked, the *masked* text is what gets
hashed, embedded and indexed; when it is configured to skip, nothing is stored.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Protocol

from ..config import Config
from ..embeddings.service import EmbeddingService
from ..privacy.policy import PrivacyFilter
from ..storage.db import ClipRecord, Database, InsertResult, hash_content, utcnow
from .detect import detect_content_type
from .processor import ClipProcessor, ProcessedClip
from .source_app import frontmost_app_name

logger = logging.getLogger(__name__)

#: ``meta`` key recording the last retention sweep, so it runs at most daily.
_LAST_PURGE_META_KEY = "last_purge_at"


class ClipboardReader(Protocol):
    """Minimal clipboard abstraction: just read the current text."""

    def paste(self) -> Optional[str]:
        """Return the clipboard text, or ``None`` when it is empty/not text."""
        ...


class PyperclipReader:
    """Default reader backed by the :mod:`pyperclip` package."""

    def __init__(self, paste_func=None) -> None:
        """Create a reader, optionally injecting a ``paste_func`` for tests."""
        if paste_func is None:
            import pyperclip  # imported lazily so headless tests never need it

            paste_func = pyperclip.paste
        self._paste = paste_func

    def paste(self) -> Optional[str]:
        """Return the current clipboard text, or ``None`` on failure."""
        try:
            value = self._paste()
        except Exception as exc:  # pragma: no cover - depends on OS clipboard
            logger.debug("clipboard read failed: %s", exc)
            return None
        return value if isinstance(value, str) else None


class ClipboardMonitor:
    """Polls the clipboard and stores new entries, skipping exact duplicates."""

    def __init__(
        self,
        db: Database,
        config: Optional[Config] = None,
        *,
        reader: Optional[ClipboardReader] = None,
        processor: Optional[ClipProcessor] = None,
        embedding_service: Optional[EmbeddingService] = None,
        privacy: Optional[PrivacyFilter] = None,
        capture_initial: bool = False,
        detect_source_app: bool = True,
    ) -> None:
        """Wire up the monitor.

        Args:
            db: Open database used for storage.
            config: Configuration (poll interval, max length, dedup, privacy).
            reader: Clipboard reader; defaults to :class:`PyperclipReader`.
            processor: Text processor; defaults to a fresh :class:`ClipProcessor`.
            embedding_service: Optional background embedder. When supplied, each
                newly stored entry is queued for embedding; the clipboard poll
                never waits for the model. The caller owns its lifecycle.
            privacy: Secret/ignore-list filter; defaults to one built from
                ``config``. Every stored entry passes through it, and the
                *masked* text is what gets hashed, embedded and indexed.
            capture_initial: When False (default) the value already on the
                clipboard at startup is ignored, so launching ClipMind does not
                re-store stale content.
            detect_source_app: When False, ``source_app`` is always left empty
                (useful for tests and repeatable benchmarks).
        """
        self.config = config or Config()
        self.db = db
        self.reader = reader or PyperclipReader()
        self.processor = processor or ClipProcessor(self.config)
        self.embedding_service = embedding_service
        self.privacy = privacy or PrivacyFilter(self.config)
        self.detect_source_app = detect_source_app

        self.capture_initial = capture_initial
        self._last_seen: Optional[str] = None
        self._seeded = False
        self._stats = {
            "polls": 0,
            "new": 0,
            "duplicates": 0,
            "skipped": 0,
            "ignored": 0,
            "secrets": 0,
            "paused": 0,
        }

    # ------------------------------------------------------------------ stats

    @property
    def stats(self) -> dict[str, int]:
        """Return a copy of the running counters (polls/new/duplicates/skipped)."""
        return dict(self._stats)

    # --------------------------------------------------------------- capturing

    def capture(
        self,
        text: Optional[str],
        source_app: Optional[str] = None,
    ) -> Optional[InsertResult]:
        """Process and store ``text`` directly, bypassing the clipboard.

        Args:
            text: Text to store.
            source_app: Overrides source-app detection when provided.

        Returns:
            The :class:`InsertResult`, or ``None`` when the text was empty or
            rejected by the oversize policy.
        """
        if self.privacy.ignored(source_app):
            # Never capture from a password manager / keychain, even if the
            # frontmost app was only matched loosely. Logs stay value-free.
            self._stats["ignored"] += 1
            logger.debug("ignoring capture from a configured source app")
            return InsertResult(False, "ignored")

        processed: Optional[ProcessedClip] = self.processor.process(text, source_app)
        if processed is None:
            self._stats["skipped"] += 1
            return None

        content = processed.content
        content_type = processed.content_type
        content_hash = processed.content_hash

        outcome = self.privacy.evaluate(content)
        if outcome.skipped:
            self._stats["secrets"] += 1
            logger.info("skipped an entry containing a configured secret (%s)", outcome.describe())
            return InsertResult(False, "secret")
        if outcome.masked:
            # The masked text is what gets hashed, embedded and indexed, so the
            # raw secret never reaches the DB, the FTS index or the model.
            self._stats["secrets"] += 1
            content = outcome.content
            content_type = detect_content_type(content)
            content_hash = hash_content(content)

        result = self.db.insert_clip(
            content=content,
            content_type=content_type,
            source_app=processed.source_app,
            content_hash=content_hash,
        )
        if result.inserted:
            self._stats["new"] += 1
            logger.info(
                "captured %s entry #%s (%d chars)%s%s",
                content_type,
                result.record.id if result.record else "?",
                len(content),
                " [truncated]" if processed.truncated else "",
                " [redacted]" if outcome.masked else "",
            )
            self._queue_embedding(result.record)
        else:
            self._stats["duplicates"] += 1
        return result

    def _queue_embedding(self, record: Optional[ClipRecord]) -> None:
        """Hand a freshly stored entry to the background embedder, if any."""
        if self.embedding_service is None or record is None:
            return
        try:
            self.embedding_service.submit(record.id, record.content, record.content_type)
        except Exception:  # pragma: no cover - never let this break capture
            logger.exception("queueing embedding for item %s failed", record.id)

    def poll_once(self) -> Optional[InsertResult]:
        """Read the clipboard once and store the value when it is new.

        Returns:
            The :class:`InsertResult` for a newly captured value, or ``None``
            when nothing changed.
        """
        self._stats["polls"] += 1
        if self.db.is_paused():
            # Honour the persisted pause flag live: skip the read entirely so
            # nothing is captured until `clipmind resume` clears it.
            self._stats["paused"] += 1
            return None
        try:
            value = self.reader.paste()
        except Exception as exc:
            # A flaky clipboard must never kill the capture daemon.
            logger.debug("clipboard read failed: %s", exc)
            return None
        if value is None:
            return None

        if not self._seeded:
            # First read establishes the baseline value on the clipboard.
            self._seeded = True
            self._last_seen = value
            if not self.capture_initial:
                logger.debug("ignoring clipboard content present at startup")
                return None
            return self.capture(value, source_app=self._source_app())

        if value == self._last_seen:
            return None
        self._last_seen = value

        return self.capture(value, source_app=self._source_app())

    def _source_app(self) -> Optional[str]:
        """Return the frontmost application name, or ``None`` when disabled."""
        return frontmost_app_name() if self.detect_source_app else None

    # -------------------------------------------------------------- retention

    def purge_expired(self, *, force: bool = False) -> int:
        """Delete entries older than ``Config.retention_days``, when configured.

        Runs once at watcher startup (``force=True``) and then at most once
        every 24 hours, so a long-running ``watch`` enforces retention daily.

        Args:
            force: Ignore the once-a-day guard and purge now.

        Returns:
            The number of rows deleted. ``VACUUM`` runs after the delete.
        """
        days = self.config.retention_days
        if not days:
            return 0
        now = utcnow()
        if not force:
            last = self.db.get_meta(_LAST_PURGE_META_KEY)
            if last:
                try:
                    last_dt: Optional[datetime] = datetime.fromisoformat(last)
                except ValueError:
                    last_dt = None
                if last_dt is not None and (now - last_dt) < timedelta(hours=24):
                    return 0
        removed = self.db.delete_older_than(now - timedelta(days=days))
        self.db.vacuum()
        self.db.set_meta(_LAST_PURGE_META_KEY, now.isoformat())
        if removed:
            logger.info(
                "retention: purged %d entr(y/ies) older than %d day(s)", removed, days
            )
        return removed

    # ----------------------------------------------------------------- loop

    def run(
        self,
        stop_event: Optional[threading.Event] = None,
        max_iterations: Optional[int] = None,
    ) -> None:
        """Poll the clipboard until stopped.

        Args:
            stop_event: When set, the loop exits. A private event is created
                when omitted.
            max_iterations: Stop after this many polls (used by tests and the
                demo script). ``None`` means run forever.
        """
        stop_event = stop_event or threading.Event()
        iterations = 0
        logger.info(
            "ClipMind capture started (poll every %.2fs, DB: %s)",
            self.config.poll_interval,
            self.db.path,
        )
        try:
            self.purge_expired(force=True)
        except Exception:  # retention must never block capture
            logger.exception("retention sweep failed")
        try:
            while not stop_event.is_set():
                if max_iterations is not None and iterations >= max_iterations:
                    break
                try:
                    self.purge_expired()
                except Exception:  # retention must never block capture
                    logger.exception("retention sweep failed")
                try:
                    self.poll_once()
                except Exception:  # keep the daemon alive on transient errors
                    logger.exception("clipboard poll failed")
                iterations += 1
                if stop_event.wait(self.config.poll_interval):
                    break
        except KeyboardInterrupt:  # pragma: no cover - interactive only
            logger.info("interrupted, stopping capture")
        finally:
            logger.info("ClipMind capture stopped (stats: %s)", self.stats)
