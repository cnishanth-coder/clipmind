"""Display-free view-model for the ClipMind popup.

Everything the popup needs to *decide* lives here, not in the widgets. That
keeps the interesting behaviour - debounced search-on-type, cancellation of
stale results, keyboard selection, type-filter cycling, time-chip dismissal,
copy-back and delete/undo - unit-testable without a display (and without
PySide6 installed at all).

Design
------
``SearchController`` owns an immutable :class:`SearchView` snapshot. Every
mutation rebuilds the snapshot and hands it to a :class:`ControllerListener`
(the Qt layer turns that into a signal, so the listener may be called from a
worker thread). Searches run through a :class:`Dispatcher`: the default is a
dedicated thread, tests inject a manual one so ordering is deterministic.

Stale-result cancellation works with a monotonically increasing *generation*.
Every query change bumps it; a result whose generation is older than the
current one is dropped before it can touch the view.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, Optional, Protocol, Sequence

from ..storage.db import preview_text, relative_age, utcnow

logger = logging.getLogger(__name__)

#: Type filters the popup cycles through with Tab, as ``(value, label)``.
#: ``None`` means "no content-type filter".
FILTERS: tuple[tuple[Optional[str], str], ...] = (
    (None, "All"),
    ("code", "Code"),
    ("url", "Links"),
    ("text", "Text"),
)

#: Single-character/word badge shown per content type in a row.
BADGE_LABELS = {
    "url": "url",
    "code": "code",
    "email": "email",
    "number": "number",
    "text": "text",
}


class UiStatus(str, Enum):
    """High-level state of the popup's data source."""

    LOADING = "loading"  # embedding model still warming up
    READY = "ready"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ResultItem:
    """One row of the popup, decoupled from the storage/hybrid layer.

    The backend adapts a :class:`~clipmind.search.hybrid.HybridResult` into this
    plain value object so the controller never imports the search package.
    """

    id: int
    content: str
    content_type: str
    created_at: datetime
    source_app: Optional[str] = None
    final_score: float = 0.0
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    #: Opaque storage details kept only so undo can restore a row losslessly.
    #: The view-model never inspects them.
    content_hash: Optional[str] = None
    embedding: Optional[bytes] = None

    def preview(self, max_len: int = 110) -> str:
        """Return a single-line, ellipsised preview of the content."""
        return preview_text(self.content, max_len)

    def relative_time(self, now: Optional[datetime] = None) -> str:
        """Return a short age string such as ``"2h ago"``."""
        return relative_age(self.created_at, now)

    @property
    def badge(self) -> str:
        """Return the type badge to show for this row."""
        return BADGE_LABELS.get(self.content_type, self.content_type)

    @property
    def is_code(self) -> bool:
        """True when the row should be rendered in a monospace font."""
        return self.content_type == "code"


@dataclass(frozen=True, slots=True)
class SearchRequest:
    """A single search job handed to the dispatcher."""

    query: str
    content_type: Optional[str]
    generation: int
    top_k: int
    ignore_time: bool = False


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    """What a backend returns for one :class:`SearchRequest`.

    ``error`` is set instead of raising: a locked database or an unavailable
    model must become a non-crashing error state in the popup, never an
    exception in a worker thread.
    """

    items: tuple[ResultItem, ...] = ()
    time_label: Optional[str] = None
    time_note: Optional[str] = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class SearchView:
    """Immutable snapshot rendered by the popup."""

    rows: tuple[ResultItem, ...] = ()
    query: str = ""
    filter: Optional[str] = None
    filter_label: str = "All"
    time_label: Optional[str] = None
    time_note: Optional[str] = None
    explain: bool = False
    status: UiStatus = UiStatus.READY
    status_message: Optional[str] = None
    selected_index: Optional[int] = None
    searching: bool = False
    has_undo: bool = False

    @property
    def selected_item(self) -> Optional[ResultItem]:
        """Return the currently selected row, or ``None``."""
        if self.selected_index is None:
            return None
        if 0 <= self.selected_index < len(self.rows):
            return self.rows[self.selected_index]
        return None


class SearchBackend(Protocol):
    """Business operations the controller needs; implemented by the app layer."""

    def run_search(self, request: SearchRequest) -> SearchOutcome:
        """Run one search and return its outcome (never raise for UI errors)."""
        ...

    def copy_to_clipboard(self, text: str) -> None:
        """Write ``text`` to the system clipboard (suppressing re-capture)."""
        ...

    def delete_item(self, item_id: int) -> bool:
        """Delete one stored entry; return True when a row was removed."""
        ...

    def restore_item(self, item: ResultItem) -> bool:
        """Re-insert a previously deleted row (used by undo)."""
        ...


class ControllerListener(Protocol):
    """Receives every new :class:`SearchView` snapshot."""

    def on_view(self, view: SearchView) -> None:
        """Handle a fresh view snapshot (may be called from any thread)."""
        ...


# --------------------------------------------------------------- scheduling


class Scheduler(Protocol):
    """Minimal abstraction over "call this later", so debounce is testable."""

    def call_later(self, delay_s: float, func: Callable[[], None]) -> object:
        """Schedule ``func`` to run after ``delay_s`` and return a handle."""
        ...

    def cancel(self, handle: object) -> None:
        """Cancel a handle previously returned by :meth:`call_later`."""
        ...


class ThreadingScheduler:
    """Default :class:`Scheduler` built on :class:`threading.Timer`."""

    def call_later(self, delay_s: float, func: Callable[[], None]) -> object:
        """Start a daemon timer."""
        timer = threading.Timer(delay_s, func)
        timer.daemon = True
        timer.start()
        return timer

    def cancel(self, handle: object) -> None:
        """Cancel the timer when it is still pending."""
        if isinstance(handle, threading.Timer):
            handle.cancel()


class Debouncer:
    """Runs ``callback`` once the call to :meth:`schedule` stops repeating.

    Each call to :meth:`schedule` restarts the timer, so a burst of keystrokes
    triggers exactly one search after the configured idle interval.
    """

    def __init__(
        self,
        delay_s: float,
        callback: Callable[[], None],
        *,
        scheduler: Optional[Scheduler] = None,
    ) -> None:
        """Store the delay/callback; nothing runs until :meth:`schedule`."""
        self._delay_s = delay_s
        self._callback = callback
        self._scheduler = scheduler or ThreadingScheduler()
        self._handle: Optional[object] = None

    def schedule(self) -> None:
        """(Re)start the countdown."""
        self.cancel()
        if self._delay_s <= 0:
            self._fire()
            return
        self._handle = self._scheduler.call_later(self._delay_s, self._fire)

    def cancel(self) -> None:
        """Cancel a pending call, if any."""
        if self._handle is not None:
            self._scheduler.cancel(self._handle)
            self._handle = None

    def _fire(self) -> None:
        """Clear the pending handle and invoke the callback."""
        self._handle = None
        try:
            self._callback()
        except Exception:  # a broken callback must not kill a timer thread
            logger.exception("debounced callback failed")


# --------------------------------------------------------------- dispatching


class Dispatcher(Protocol):
    """Runs searches off the UI thread."""

    def submit(
        self,
        request: SearchRequest,
        on_result: Callable[[SearchRequest, SearchOutcome], None],
    ) -> None:
        """Queue ``request``; call ``on_result(request, outcome)`` when done."""
        ...

    def shutdown(self) -> None:
        """Stop the worker, if any."""
        ...


class ThreadedDispatcher:
    """Single background worker that runs searches one at a time.

    The worker never raises: a backend failure is turned into a
    :class:`SearchOutcome` with ``error`` set, so one bad query cannot take the
    process down.
    """

    def __init__(
        self,
        run_search: Callable[[SearchRequest], SearchOutcome],
        *,
        name: str = "clipmind-search",
    ) -> None:
        """Start the worker thread."""
        self._run_search = run_search
        self._queue: "queue.Queue[Optional[tuple[SearchRequest, Callable[[SearchRequest, SearchOutcome], None]]]]" = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)
        self._thread.start()

    def submit(
        self,
        request: SearchRequest,
        on_result: Callable[[SearchRequest, SearchOutcome], None],
    ) -> None:
        """Enqueue a job (thread-safe)."""
        self._queue.put((request, on_result))

    def shutdown(self, timeout: float = 2.0) -> None:
        """Ask the worker to stop and join it."""
        self._queue.put(None)
        self._thread.join(timeout=timeout)

    def _loop(self) -> None:
        """Consume jobs until shutdown, converting errors into outcomes."""
        while True:
            item = self._queue.get()
            if item is None:
                break
            request, on_result = item
            try:
                outcome = self._run_search(request)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("search backend raised")
                outcome = SearchOutcome(error=str(exc))
            try:
                on_result(request, outcome)
            except Exception:  # pragma: no cover - defensive
                logger.exception("search result callback raised")


# --------------------------------------------------------------- controller


class SearchController:
    """Holds all popup state and drives searches.

    Args:
        backend: Business operations (search, copy, delete, restore).
        listener: Receives a new :class:`SearchView` on every change.
        dispatcher: Runs searches off-thread. Defaults to a threaded worker;
            tests inject a manual dispatcher.
        debounce_ms: Typing idle time before searching.
        top_k: Maximum rows to request per search.
        explain: Initial state of the score-breakdown toggle.
        now: Clock used for relative ages (injectable for tests).
    """

    def __init__(
        self,
        backend: SearchBackend,
        listener: ControllerListener,
        *,
        dispatcher: Optional[Dispatcher] = None,
        scheduler: Optional[Scheduler] = None,
        debounce_ms: int = 150,
        top_k: int = 20,
        explain: bool = False,
        now: Callable[[], datetime] = utcnow,
    ) -> None:
        """Create an idle controller showing nothing until :meth:`refresh`."""
        self._backend = backend
        self._listener = listener
        self._dispatcher = dispatcher or ThreadedDispatcher(backend.run_search)
        self._debouncer = Debouncer(
            max(0, debounce_ms) / 1000.0, self._on_debounce, scheduler=scheduler
        )
        self._top_k = max(1, top_k)
        self._now = now

        self._lock = threading.RLock()
        self._generation = 0
        self._query = ""
        self._filter_index = 0
        self._explain = explain
        self._ignore_time = False
        self._rows: tuple[ResultItem, ...] = ()
        self._selected: Optional[int] = None
        self._time_label: Optional[str] = None
        self._time_note: Optional[str] = None
        self._status = UiStatus.READY
        self._status_message: Optional[str] = None
        self._searching = False
        self._last_deleted: Optional[tuple[ResultItem, int]] = None

    # ------------------------------------------------------------- properties

    @property
    def filter(self) -> Optional[str]:
        """Current content-type filter value (``None`` means all)."""
        return FILTERS[self._filter_index][0]

    @property
    def filter_label(self) -> str:
        """Human label of the current filter (e.g. ``"Links"``)."""
        return FILTERS[self._filter_index][1]

    @property
    def explain(self) -> bool:
        """Whether the score breakdown is shown."""
        return self._explain

    def snapshot(self) -> SearchView:
        """Return the current immutable view snapshot."""
        with self._lock:
            return self._snapshot_locked()

    # ---------------------------------------------------------------- commands

    def set_status(self, status: UiStatus, message: Optional[str] = None) -> None:
        """Update the model/database status and re-render.

        Called by the app when the embedding model finishes warming up (or
        fails), so an early-opened popup can show "Loading model...".
        """
        with self._lock:
            self._status = status
            self._status_message = message
        self._emit()

    def set_query(self, text: str) -> None:
        """Record a new query and schedule a debounced search."""
        with self._lock:
            self._query = text
            self._generation += 1
            # A fresh query re-enables time parsing for a new phrase.
            self._ignore_time = False
            self._searching = True
        self._debouncer.schedule()
        self._emit()

    def refresh(self) -> None:
        """Run a search immediately (no debounce). Used when the popup opens."""
        with self._lock:
            self._generation += 1
            self._searching = True
        self._debouncer.cancel()
        self._emit()
        self._start_search()

    def cycle_filter(self) -> None:
        """Advance to the next type filter and re-search immediately."""
        with self._lock:
            self._filter_index = (self._filter_index + 1) % len(FILTERS)
        self._restart()

    def set_filter(self, value: Optional[str]) -> None:
        """Select a filter by value (e.g. ``"code"`` or ``None``)."""
        for index, (candidate, _label) in enumerate(FILTERS):
            if candidate == value:
                with self._lock:
                    self._filter_index = index
                self._restart()
                return

    def toggle_explain(self) -> None:
        """Flip the score-breakdown toggle and re-render (no re-search needed)."""
        with self._lock:
            self._explain = not self._explain
        self._emit()

    def set_explain(self, value: bool) -> None:
        """Set the score-breakdown toggle explicitly."""
        with self._lock:
            self._explain = bool(value)
        self._emit()

    def dismiss_time_filter(self) -> None:
        """Drop the detected time filter and re-search treating the phrase as text."""
        with self._lock:
            self._ignore_time = True
            self._time_label = None
        self._restart()

    # ---------------------------------------------------------------- selection

    def move_selection(self, delta: int) -> None:
        """Move the highlighted row by ``delta``, wrapping around the list."""
        with self._lock:
            if not self._rows:
                self._selected = None
            elif self._selected is None:
                self._selected = 0 if delta >= 0 else len(self._rows) - 1
            else:
                self._selected = (self._selected + delta) % len(self._rows)
        self._emit()

    def select_index(self, index: Optional[int]) -> None:
        """Highlight ``index`` (clamped to a valid row, or ``None``)."""
        with self._lock:
            if index is None or not self._rows:
                self._selected = None
            else:
                self._selected = max(0, min(int(index), len(self._rows) - 1))
        self._emit()

    def selected_item(self) -> Optional[ResultItem]:
        """Return the highlighted row, or ``None``."""
        return self.snapshot().selected_item

    # -------------------------------------------------------------- activation

    def activate_selected(self) -> Optional[ResultItem]:
        """Copy the highlighted row back to the clipboard.

        Returns:
            The copied item, or ``None`` when nothing was selected. The caller
            (the popup) closes itself on a non-``None`` result.
        """
        item = self.selected_item()
        if item is None:
            return None
        self._backend.copy_to_clipboard(item.content)
        logger.debug("copied item #%s back to the clipboard", item.id)
        return item

    # ------------------------------------------------------------- delete/undo

    def delete_selected(self) -> Optional[ResultItem]:
        """Delete the highlighted row and remember it for :meth:`undo_delete`.

        Returns:
            The deleted item, or ``None`` when nothing was selected or the
            backend refused the delete.
        """
        with self._lock:
            item = self._snapshot_locked().selected_item
            index = self._selected or 0
        if item is None:
            return None
        if not self._backend.delete_item(item.id):
            return None
        with self._lock:
            rows = tuple(row for row in self._rows if row.id != item.id)
            self._rows = rows
            self._last_deleted = (item, index)
            if not rows:
                self._selected = None
            else:
                self._selected = min(index, len(rows) - 1)
        self._emit()
        return item

    def undo_delete(self) -> Optional[ResultItem]:
        """Re-insert the most recently deleted row (best effort)."""
        with self._lock:
            pending = self._last_deleted
        if pending is None:
            return None
        item, index = pending
        if not self._backend.restore_item(item):
            return None
        with self._lock:
            self._last_deleted = None
            # Put it back where it was, if that position still makes sense.
            rows = list(self._rows)
            insert_at = max(0, min(index, len(rows)))
            rows.insert(insert_at, item)
            self._rows = tuple(rows)
            self._selected = insert_at
        self._emit()
        return item

    # ----------------------------------------------------------------- cleanup

    def close(self) -> None:
        """Cancel pending work and stop the dispatcher."""
        self._debouncer.cancel()
        self._dispatcher.shutdown()

    # ------------------------------------------------------------------ internals

    def _restart(self) -> None:
        """Re-run the current query immediately (used by filter/chip changes)."""
        self._debouncer.cancel()
        with self._lock:
            self._generation += 1
            self._searching = True
        self._emit()
        self._start_search()

    def _on_debounce(self) -> None:
        """Debounce interval elapsed: run a search."""
        self._start_search()

    def _start_search(self) -> None:
        """Submit the current query to the dispatcher."""
        with self._lock:
            request = SearchRequest(
                query=self._query,
                content_type=self.filter,
                generation=self._generation,
                top_k=self._top_k,
                ignore_time=self._ignore_time,
            )
        self._dispatcher.submit(request, self._on_result)

    def _on_result(self, request: SearchRequest, outcome: SearchOutcome) -> None:
        """Apply a search result, dropping it when a newer query superseded it."""
        with self._lock:
            if request.generation != self._generation:
                logger.debug(
                    "dropping stale search result (gen %s != %s)",
                    request.generation,
                    self._generation,
                )
                return
            self._searching = False
            self._time_note = None
            if outcome.error:
                self._status = UiStatus.ERROR
                self._status_message = outcome.error
                self._rows = ()
                self._selected = None
            else:
                if self._status is UiStatus.ERROR:
                    self._status = UiStatus.READY
                    self._status_message = None
                self._rows = outcome.items
                self._time_label = None if self._ignore_time else outcome.time_label
                self._time_note = outcome.time_note
                self._selected = 0 if outcome.items else None
        self._emit()

    def _snapshot_locked(self) -> SearchView:
        """Build a snapshot; the caller must hold the lock."""
        return SearchView(
            rows=self._rows,
            query=self._query,
            filter=self.filter,
            filter_label=self.filter_label,
            time_label=self._time_label,
            time_note=self._time_note,
            explain=self._explain,
            status=self._status,
            status_message=self._status_message,
            selected_index=self._selected,
            searching=self._searching,
            has_undo=self._last_deleted is not None,
        )

    def _emit(self) -> None:
        """Send the current snapshot to the listener."""
        view = self.snapshot()
        try:
            self._listener.on_view(view)
        except Exception:  # pragma: no cover - a broken listener must not crash
            logger.exception("view listener raised")


def result_items_from_records(records: Sequence[object]) -> tuple[ResultItem, ...]:
    """Adapt storage records into :class:`ResultItem` objects.

    Args:
        records: Objects exposing ``id``, ``content``, ``content_type``,
            ``created_at`` and ``source_app`` (typically ``ClipRecord``).

    Returns:
        A tuple of :class:`ResultItem` with zeroed scores.
    """
    return tuple(
        ResultItem(
            id=int(getattr(record, "id")),
            content=str(getattr(record, "content")),
            content_type=str(getattr(record, "content_type")),
            created_at=getattr(record, "created_at"),
            source_app=getattr(record, "source_app"),
            content_hash=getattr(record, "content_hash", None),
            embedding=getattr(record, "embedding", None),
        )
        for record in records
    )


__all__ = [
    "BADGE_LABELS",
    "Debouncer",
    "Dispatcher",
    "FILTERS",
    "ControllerListener",
    "ResultItem",
    "Scheduler",
    "SearchBackend",
    "SearchController",
    "SearchOutcome",
    "SearchRequest",
    "SearchView",
    "ThreadedDispatcher",
    "ThreadingScheduler",
    "UiStatus",
    "result_items_from_records",
]
