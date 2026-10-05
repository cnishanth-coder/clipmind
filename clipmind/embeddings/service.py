"""Turning stored entries into embeddings.

Two entry points:

* :class:`EmbeddingService` - a queue + daemon thread used by the capture
  monitor. New entries are handed over immediately; the clipboard poll loop
  never waits for the model.
* :func:`reindex_pending` - a synchronous sweep used by ``clipmind reindex``
  that embeds every eligible row whose ``embedding`` column is still ``NULL``.

Both funnel into :func:`embed_batch`, so the "what gets embedded and where it is
written" rules exist in exactly one place.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

from ..config import Config
from ..storage.db import Database
from .codec import encode_vector
from .encoder import Embedder, EmbeddingModelUnavailable, get_embedder, is_embeddable

logger = logging.getLogger(__name__)

#: Called as ``progress(done, total)`` while a reindex runs.
ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True, slots=True)
class EmbedItem:
    """One pending embedding job."""

    item_id: int
    text: str
    content_type: str


def embed_batch(
    db: Database,
    embedder: Embedder,
    items: Sequence[EmbedItem],
) -> tuple[int, int]:
    """Embed ``items`` and write the vectors into the database.

    Args:
        db: Open database to write to.
        embedder: Embedder to use.
        items: Batch of eligible items.

    Returns:
        ``(embedded, failed)`` counts. A batch that fails for a per-entry
        reason is reported, never raised, so one bad entry cannot kill the
        capture daemon.

    Raises:
        EmbeddingModelUnavailable: If the model itself cannot be loaded. That
            is an environment problem rather than a data problem, so it is
            propagated for a caller (``clipmind reindex``) to report clearly;
            the background worker catches it and keeps running.
    """
    if not items:
        return 0, 0
    try:
        vectors = embedder.embed_texts([item.text for item in items])
    except EmbeddingModelUnavailable:
        raise
    except Exception:
        logger.exception("embedding batch of %d item(s) failed", len(items))
        return 0, len(items)

    embedded = 0
    failed = 0
    for index, item in enumerate(items):
        try:
            db.set_embedding(item.item_id, encode_vector(vectors[index]))
            embedded += 1
        except Exception:
            logger.exception("storing embedding for item %d failed", item.item_id)
            failed += 1
    return embedded, failed


class EmbeddingService:
    """Embeds new captures on a background thread.

    The service is deliberately dumb about storage ordering: it receives a
    ``(item_id, text, content_type)`` triple at submit time and never re-reads
    the row, so an entry deleted while it is queued simply fails to update.
    """

    def __init__(
        self,
        db: Database,
        *,
        embedder: Optional[Embedder] = None,
        config: Optional[Config] = None,
        queue_maxsize: int = 0,
    ) -> None:
        """Create the service (nothing is started until :meth:`start`).

        Args:
            db: Database the worker writes embeddings into.
            embedder: Embedder to use; resolved lazily from config when omitted.
            config: Configuration (batch size, eligibility rules).
            queue_maxsize: Max queued jobs (``0`` means unbounded).
        """
        self.db = db
        self.config = config or Config()
        self._embedder = embedder
        self._queue: "queue.Queue[Optional[EmbedItem]]" = queue.Queue(maxsize=queue_maxsize)
        self._condition = threading.Condition()
        self._pending = 0
        self._thread: Optional[threading.Thread] = None
        self._stats = {"queued": 0, "embedded": 0, "skipped": 0, "failed": 0}
        self._last_error: Optional[str] = None

    # ------------------------------------------------------------------ lifecycle

    @property
    def embedder(self) -> Embedder:
        """Return the embedder, creating the configured one on first access."""
        if self._embedder is None:
            self._embedder = get_embedder(self.config)
        return self._embedder

    def start(self) -> "EmbeddingService":
        """Start the background worker thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return self
        self._thread = threading.Thread(
            target=self._run, name="clipmind-embedder", daemon=True
        )
        self._thread.start()
        logger.debug("embedding worker started")
        return self

    def stop(self, timeout: float = 30.0, *, drain: bool = True) -> None:
        """Stop the worker, optionally finishing queued work first.

        Args:
            timeout: Seconds to wait for each phase.
            drain: When True (default) queued jobs are embedded before exit.
        """
        if drain:
            self.flush(timeout=timeout)
        if self._thread is None:
            return
        self._queue.put(None)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():  # pragma: no cover - only on a wedged model
            logger.warning("embedding worker did not stop within %.1fs", timeout)
        self._thread = None

    def __enter__(self) -> "EmbeddingService":
        """Start the service when entering a ``with`` block."""
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        """Stop (and drain) the service when leaving a ``with`` block."""
        self.stop()

    # ------------------------------------------------------------------ submitting

    def submit(
        self,
        item_id: int,
        text: str,
        content_type: str = "text",
    ) -> bool:
        """Queue one entry for embedding.

        Ineligible entries (numbers, strings shorter than the configured
        minimum) are discarded immediately and counted as skipped.

        Args:
            item_id: Row id to attach the embedding to.
            text: Stored content to embed.
            content_type: Detected content type.

        Returns:
            True when the job was queued, False when it was skipped.
        """
        if not is_embeddable(text, content_type, self.config):
            self._stats["skipped"] += 1
            return False
        with self._condition:
            self._pending += 1
        self._queue.put(EmbedItem(item_id, text, content_type))
        self._stats["queued"] += 1
        return True

    def flush(self, timeout: float = 60.0) -> bool:
        """Block until every queued job has been processed.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            True if the queue drained in time, False on timeout.
        """
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._pending > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning("%d embedding job(s) still pending", self._pending)
                    return False
                self._condition.wait(timeout=remaining)
        return True

    @property
    def pending(self) -> int:
        """Number of jobs submitted but not yet processed."""
        with self._condition:
            return self._pending

    @property
    def stats(self) -> dict[str, int]:
        """Return a copy of the running counters."""
        return {**self._stats, "pending": self.pending}

    @property
    def last_error(self) -> Optional[str]:
        """Error message from the most recent failed batch, if any."""
        return self._last_error

    # ------------------------------------------------------------------ worker

    def _run(self) -> None:
        """Worker loop: gather batches and embed them until asked to stop."""
        while True:
            try:
                first = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            batch: list[EmbedItem] = []
            stop_after = False
            if first is not None:
                batch.append(first)
                while len(batch) < self.config.embedding_batch_size:
                    try:
                        nxt = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is None:
                        stop_after = True
                        break
                    batch.append(nxt)
            else:
                stop_after = True

            self._process(batch)
            if stop_after:
                break
        logger.debug("embedding worker stopped")

    def _process(self, batch: Sequence[EmbedItem]) -> None:
        """Embed one batch and update the counters."""
        embedded = failed = 0
        try:
            if batch:
                embedded, failed = embed_batch(self.db, self.embedder, batch)
                self._stats["embedded"] += embedded
                self._stats["failed"] += failed
                if failed:
                    self._last_error = f"{failed} of {len(batch)} embeddings failed"
        except EmbeddingModelUnavailable as exc:
            # A missing model is an environment problem: report it once, in one
            # line, and keep the capture daemon alive instead of crashing it.
            logger.error("embedding model unavailable: %s", exc)
            self._stats["failed"] += len(batch)
            self._last_error = str(exc)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("unexpected embedding worker error")
            self._stats["failed"] += len(batch)
            self._last_error = str(exc)
        finally:
            with self._condition:
                self._pending -= len(batch)
                if self._pending <= 0:
                    self._pending = 0
                self._condition.notify_all()


@dataclass(frozen=True, slots=True)
class ReindexReport:
    """Outcome of :func:`reindex_pending`.

    Attributes:
        missing: Rows whose ``embedding`` column was ``NULL``.
        eligible: Subset of ``missing`` that passed the embedding policy.
        skipped: ``missing - eligible`` (numbers, very short strings).
        embedded: Rows successfully embedded and written.
        failed: Rows whose embedding could not be computed or stored.
        seconds: Wall-clock duration of the sweep.
    """

    missing: int
    eligible: int
    skipped: int
    embedded: int
    failed: int
    seconds: float


def reindex_pending(
    db: Database,
    *,
    embedder: Optional[Embedder] = None,
    config: Optional[Config] = None,
    progress: Optional[ProgressCallback] = None,
    batch_size: Optional[int] = None,
) -> ReindexReport:
    """Embed every eligible row that has no embedding yet.

    Rows already carrying an embedding are never touched, which makes this
    safe (and cheap) to re-run at any time.

    Args:
        db: Database to sweep.
        embedder: Embedder to use; resolved from config when omitted.
        config: Configuration (batch size, eligibility rules).
        progress: Optional ``progress(done, total)`` callback, called per batch
            and once at the end.
        batch_size: Overrides ``config.embedding_batch_size``.

    Returns:
        A :class:`ReindexReport`.
    """
    started = time.monotonic()
    config = config or Config()
    embedder = embedder or get_embedder(config)
    size = batch_size or config.embedding_batch_size

    missing_records = list(db.iter_missing_embeddings())
    eligible: list[EmbedItem] = [
        EmbedItem(record.id, record.content, record.content_type)
        for record in missing_records
        if is_embeddable(record.content, record.content_type, config)
    ]

    total = len(eligible)
    embedded = failed = 0
    for start in range(0, total, size):
        chunk = eligible[start : start + size]
        done, errors = embed_batch(db, embedder, chunk)
        embedded += done
        failed += errors
        if progress is not None:
            progress(min(start + len(chunk), total), total)
    if progress is not None and total == 0:
        progress(0, 0)

    return ReindexReport(
        missing=len(missing_records),
        eligible=total,
        skipped=len(missing_records) - total,
        embedded=embedded,
        failed=failed,
        seconds=time.monotonic() - started,
    )


def pending_items(
    db: Database,
    config: Optional[Config] = None,
) -> Iterable[EmbedItem]:
    """Yield eligible items that still need an embedding (iterator, not a list)."""
    config = config or Config()
    for record in db.iter_missing_embeddings():
        if is_embeddable(record.content, record.content_type, config):
            yield EmbedItem(record.id, record.content, record.content_type)
