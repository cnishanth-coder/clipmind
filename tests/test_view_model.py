"""Tests for the display-free popup view-model.

None of these import PySide6: the interesting behaviour (debounce, stale-result
cancellation, selection, filter cycling, copy-back, delete/undo, error states)
lives in ``clipmind.ui.view_model`` and is driven through injected fakes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

from clipmind.ui.view_model import (
    FILTERS,
    Debouncer,
    ResultItem,
    SearchController,
    SearchOutcome,
    SearchRequest,
    UiStatus,
)

TZ = timezone(timedelta(hours=-4))
NOW = datetime(2026, 10, 6, 12, 0, tzinfo=TZ)


def make_item(item_id: int, content: str = "hello", content_type: str = "text") -> ResultItem:
    """Build a deterministic result row."""
    return ResultItem(
        id=item_id,
        content=content,
        content_type=content_type,
        created_at=NOW - timedelta(minutes=item_id),
        source_app="Safari",
        content_hash=f"hash-{item_id}",
    )


class ManualScheduler:
    """Scheduler that only fires when the test asks it to."""

    def __init__(self) -> None:
        self.pending: list[object] = []

    def call_later(self, delay_s, func):
        handle = _Handle(delay_s, func)
        self.pending.append(handle)
        return handle

    def cancel(self, handle) -> None:
        handle.cancelled = True

    def fire(self) -> None:
        """Run every non-cancelled pending callback, in order."""
        pending, self.pending = self.pending, []
        for handle in pending:
            if not handle.cancelled:
                handle.func()


class _Handle:
    """Cancellable timer handle for :class:`ManualScheduler`."""

    def __init__(self, delay_s, func) -> None:
        self.delay_s = delay_s
        self.func = func
        self.cancelled = False


class ManualDispatcher:
    """Dispatcher that records submissions and completes them on demand."""

    def __init__(self) -> None:
        self.requests: list[tuple[SearchRequest, object]] = []
        self.shutdown_called = False

    def submit(self, request, on_result) -> None:
        self.requests.append((request, on_result))

    def complete(self, index: int, outcome: SearchOutcome) -> None:
        """Deliver ``outcome`` for the request at ``index``."""
        request, on_result = self.requests[index]
        on_result(request, outcome)

    def latest_request(self) -> SearchRequest:
        """Return the most recently submitted request."""
        return self.requests[-1][0]

    def shutdown(self) -> None:
        self.shutdown_called = True


class FakeBackend:
    """Minimal :class:`SearchBackend` recording every call."""

    def __init__(self) -> None:
        self.calls: list[SearchRequest] = []
        self.copied: list[str] = []
        self.deleted: list[int] = []
        self.restored: list[ResultItem] = []
        self.outcome = SearchOutcome()
        self.delete_ok = True

    def run_search(self, request: SearchRequest) -> SearchOutcome:
        self.calls.append(request)
        return self.outcome

    def copy_to_clipboard(self, text: str) -> None:
        self.copied.append(text)

    def delete_item(self, item_id: int) -> bool:
        if not self.delete_ok:
            return False
        self.deleted.append(item_id)
        return True

    def restore_item(self, item: ResultItem) -> bool:
        self.restored.append(item)
        return True


class RecordingListener:
    """Listener that keeps every snapshot it is handed."""

    def __init__(self) -> None:
        self.views: list = []

    def on_view(self, view) -> None:
        self.views.append(view)

    @property
    def latest(self):
        return self.views[-1]


@pytest.fixture()
def env():
    """Return ``(controller, backend, dispatcher, scheduler, listener)``."""
    backend = FakeBackend()
    dispatcher = ManualDispatcher()
    scheduler = ManualScheduler()
    listener = RecordingListener()
    controller = SearchController(
        backend,
        listener,
        dispatcher=dispatcher,
        scheduler=scheduler,
        debounce_ms=150,
        top_k=20,
    )
    return controller, backend, dispatcher, scheduler, listener


# ------------------------------------------------------------------ debounce


def test_typing_coalesces_into_one_search(env) -> None:
    controller, _backend, dispatcher, scheduler, _listener = env
    for text in ("r", "re", "rea", "react"):
        controller.set_query(text)
    assert dispatcher.requests == []  # nothing fired yet
    scheduler.fire()
    assert len(dispatcher.requests) == 1
    assert dispatcher.latest_request().query == "react"


def test_refresh_skips_the_debounce(env) -> None:
    controller, _backend, dispatcher, scheduler, _listener = env
    controller.refresh()
    assert len(dispatcher.requests) == 1
    assert dispatcher.latest_request().query == ""
    assert len(scheduler.pending) == 0


# ------------------------------------------------------- stale cancellation


def test_stale_result_is_dropped(env) -> None:
    controller, _backend, dispatcher, scheduler, listener = env
    controller.set_query("first")
    scheduler.fire()  # request 0 (generation 1)
    controller.set_query("second")
    scheduler.fire()  # request 1 (generation 2)

    # The late result for the first query must not touch the view.
    dispatcher.complete(0, SearchOutcome(items=(make_item(1, "old"),)))
    assert listener.latest.rows == ()
    assert listener.latest.searching is True

    dispatcher.complete(1, SearchOutcome(items=(make_item(2, "new"),)))
    assert [row.content for row in listener.latest.rows] == ["new"]
    assert listener.latest.searching is False


def test_stale_result_does_not_clear_searching_for_newer_query(env) -> None:
    controller, _backend, dispatcher, scheduler, listener = env
    controller.set_query("a")
    scheduler.fire()
    controller.set_query("ab")
    scheduler.fire()
    dispatcher.complete(0, SearchOutcome(items=(make_item(1),)))
    # still waiting on the second search
    assert listener.latest.searching is True


# ------------------------------------------------------------------ results


def test_results_reset_selection_to_top(env) -> None:
    controller, _backend, dispatcher, _scheduler, listener = env
    controller.refresh()
    dispatcher.complete(0, SearchOutcome(items=(make_item(1), make_item(2))))
    assert listener.latest.selected_index == 0


def test_empty_result_clears_selection(env) -> None:
    controller, _backend, dispatcher, _scheduler, listener = env
    controller.refresh()
    dispatcher.complete(0, SearchOutcome(items=()))
    assert listener.latest.selected_index is None


def test_error_outcome_becomes_error_state(env) -> None:
    controller, _backend, dispatcher, _scheduler, listener = env
    controller.refresh()
    dispatcher.complete(0, SearchOutcome(error="database is locked"))
    view = listener.latest
    assert view.status is UiStatus.ERROR
    assert view.status_message == "database is locked"
    assert view.rows == ()


def test_a_later_success_clears_a_previous_error(env) -> None:
    controller, _backend, dispatcher, _scheduler, listener = env
    controller.refresh()
    dispatcher.complete(0, SearchOutcome(error="boom"))
    controller.refresh()
    dispatcher.complete(1, SearchOutcome(items=(make_item(1),)))
    assert listener.latest.status is UiStatus.READY
    assert listener.latest.status_message is None


# ------------------------------------------------------------------ selection


def test_selection_wraps_around(env) -> None:
    controller, _backend, dispatcher, _scheduler, _listener = env
    controller.refresh()
    dispatcher.complete(0, SearchOutcome(items=(make_item(1), make_item(2), make_item(3))))
    assert controller.selected_item().id == 1
    controller.move_selection(-1)
    assert controller.selected_item().id == 3
    controller.move_selection(1)
    assert controller.selected_item().id == 1


def test_selection_without_rows_is_none(env) -> None:
    controller, _backend, _dispatcher, _scheduler, _listener = env
    controller.move_selection(1)
    assert controller.selected_item() is None


def test_select_index_clamps(env) -> None:
    controller, _backend, dispatcher, _scheduler, _listener = env
    controller.refresh()
    dispatcher.complete(0, SearchOutcome(items=(make_item(1), make_item(2))))
    controller.select_index(99)
    assert controller.selected_item().id == 2
    controller.select_index(None)
    assert controller.selected_item() is None


# ------------------------------------------------------------------ filters


def test_filter_cycles_and_researches(env) -> None:
    controller, _backend, dispatcher, _scheduler, _listener = env
    labels = []
    for _ in range(len(FILTERS)):
        controller.cycle_filter()
        labels.append(controller.filter_label)
    assert labels == ["Code", "Links", "Text", "All"]
    # Each cycle re-runs the query with the matching content-type filter.
    assert [request.content_type for request, _ in dispatcher.requests] == [
        "code",
        "url",
        "text",
        None,
    ]
    assert controller.filter is None


def test_initial_filter_label_is_all(env) -> None:
    controller, *_ = env
    assert controller.filter is None
    assert controller.filter_label == "All"


# ------------------------------------------------------------------ time chip


def test_time_label_is_kept(env) -> None:
    controller, _backend, dispatcher, _scheduler, listener = env
    controller.set_query("react hooks yesterday")
    _scheduler.fire()
    dispatcher.complete(0, SearchOutcome(items=(make_item(1),), time_label="Yesterday"))
    assert listener.latest.time_label == "Yesterday"


def test_dismissing_the_time_chip_reruns_without_time(env) -> None:
    controller, _backend, dispatcher, scheduler, listener = env
    controller.set_query("react hooks yesterday")
    scheduler.fire()
    dispatcher.complete(0, SearchOutcome(items=(make_item(1),), time_label="Yesterday"))
    assert listener.latest.time_label == "Yesterday"

    controller.dismiss_time_filter()
    request = dispatcher.latest_request()
    assert request.ignore_time is True
    assert request.query == "react hooks yesterday"
    dispatcher.complete(len(dispatcher.requests) - 1, SearchOutcome(items=(make_item(1),)))
    assert listener.latest.time_label is None


def test_time_filter_is_reenabled_by_a_new_query(env) -> None:
    controller, _backend, dispatcher, scheduler, listener = env
    controller.set_query("x")
    scheduler.fire()
    controller.dismiss_time_filter()
    assert dispatcher.latest_request().ignore_time is True
    controller.set_query("y")
    scheduler.fire()
    assert dispatcher.latest_request().ignore_time is False


def test_ambiguous_time_note_is_surfaced(env) -> None:
    controller, _backend, dispatcher, _scheduler, listener = env
    controller.refresh()
    dispatcher.complete(
        0, SearchOutcome(items=(make_item(1),), time_note="a bare weekday was left in")
    )
    assert "bare weekday" in listener.latest.time_note


# ------------------------------------------------------------------ explain


def test_explain_toggle_does_not_trigger_a_search(env) -> None:
    controller, _backend, dispatcher, _scheduler, listener = env
    controller.refresh()
    dispatcher.complete(0, SearchOutcome(items=(make_item(1),)))
    before = len(dispatcher.requests)
    controller.toggle_explain()
    assert listener.latest.explain is True
    assert len(dispatcher.requests) == before


# --------------------------------------------------------------- copy-back


def test_copy_back_copies_the_selected_item(env) -> None:
    controller, backend, dispatcher, _scheduler, _listener = env
    controller.refresh()
    dispatcher.complete(0, SearchOutcome(items=(make_item(1, "first"), make_item(2, "second"))))
    controller.move_selection(1)
    item = controller.activate_selected()
    assert item is not None and item.content == "second"
    assert backend.copied == ["second"]


def test_copy_back_with_no_selection_is_a_no_op(env) -> None:
    controller, backend, _dispatcher, _scheduler, _listener = env
    assert controller.activate_selected() is None
    assert backend.copied == []


# ------------------------------------------------------------- delete / undo


def test_delete_removes_the_row_and_offers_undo(env) -> None:
    controller, backend, dispatcher, _scheduler, listener = env
    controller.refresh()
    dispatcher.complete(
        0, SearchOutcome(items=(make_item(1, "a"), make_item(2, "b"), make_item(3, "c")))
    )
    controller.move_selection(1)  # select "b"
    deleted = controller.delete_selected()
    assert deleted is not None and deleted.content == "b"
    assert backend.deleted == [2]
    assert [row.content for row in listener.latest.rows] == ["a", "c"]
    assert listener.latest.has_undo is True

    restored = controller.undo_delete()
    assert restored is not None and restored.content == "b"
    assert [row.content for row in listener.latest.rows] == ["a", "b", "c"]
    assert backend.restored == [restored]
    assert listener.latest.has_undo is False


def test_delete_failure_keeps_the_row(env) -> None:
    controller, backend, dispatcher, _scheduler, listener = env
    backend.delete_ok = False
    controller.refresh()
    dispatcher.complete(0, SearchOutcome(items=(make_item(1), make_item(2))))
    assert controller.delete_selected() is None
    assert len(listener.latest.rows) == 2
    assert listener.latest.has_undo is False


def test_undo_without_a_delete_is_a_no_op(env) -> None:
    controller, backend, _dispatcher, _scheduler, _listener = env
    assert controller.undo_delete() is None
    assert backend.restored == []


def test_deleting_the_last_row_clears_selection(env) -> None:
    controller, _backend, dispatcher, _scheduler, listener = env
    controller.refresh()
    dispatcher.complete(0, SearchOutcome(items=(make_item(1),)))
    controller.delete_selected()
    assert listener.latest.rows == ()
    assert listener.latest.selected_index is None


# --------------------------------------------------------------- model state


def test_model_status_is_reported(env) -> None:
    controller, _backend, _dispatcher, _scheduler, listener = env
    controller.set_status(UiStatus.LOADING, None)
    assert listener.latest.status is UiStatus.LOADING
    controller.set_status(UiStatus.READY, None)
    assert listener.latest.status is UiStatus.READY


def test_close_shuts_down_the_dispatcher(env) -> None:
    controller, _backend, dispatcher, _scheduler, _listener = env
    controller.close()
    assert dispatcher.shutdown_called is True


def test_a_listener_exception_does_not_propagate() -> None:
    class BadListener:
        def on_view(self, view) -> None:
            raise RuntimeError("listener exploded")

    controller = SearchController(
        FakeBackend(),
        BadListener(),
        dispatcher=ManualDispatcher(),
        scheduler=ManualScheduler(),
    )
    controller.set_query("x")  # must not raise


# ------------------------------------------------------------------ helpers


def test_debouncer_with_zero_delay_fires_immediately() -> None:
    fired = []
    debouncer = Debouncer(0.0, lambda: fired.append(True), scheduler=ManualScheduler())
    debouncer.schedule()
    assert fired == [True]


def test_search_request_defaults() -> None:
    request = SearchRequest(query="q", content_type=None, generation=1, top_k=5)
    assert request.ignore_time is False
