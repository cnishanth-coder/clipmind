"""Tests for :mod:`clipmind.search.timeparse` (no model, no network).

Time parsing is pinned with an injectable ``now`` and a fixed UTC-4 offset so
the expectations are identical on every machine regardless of the local
timezone or daylight-saving state.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from clipmind.config import Config
from clipmind.search.timeparse import TimeQuery, parse_time_query

#: Fixed, DST-free reference timezone for deterministic wall-clock maths.
TZ = timezone(timedelta(hours=-4))

#: Wednesday, 7 October 2026, 14:30 local. (2026-10-05 is a Monday.)
NOW = datetime(2026, 10, 7, 14, 30, tzinfo=TZ)


def _config(**kwargs) -> Config:
    return Config(db_path=":memory:", **kwargs)


def _parse(query: str, **kwargs) -> TimeQuery:
    kwargs.setdefault("now", NOW)
    kwargs.setdefault("config", _config())
    return parse_time_query(query, **kwargs)


# --------------------------------------------------------------------- basics


def test_no_time_phrase_leaves_query_untouched() -> None:
    parsed = _parse("SELECT id FROM orders WHERE status = open")
    assert parsed.cleaned_query == "SELECT id FROM orders WHERE status = open"
    assert not parsed.has_range
    assert not parsed.ambiguous
    assert parsed.start is None and parsed.end is None


def test_raw_query_is_preserved() -> None:
    parsed = _parse("react hooks article yesterday")
    assert parsed.raw == "react hooks article yesterday"


def test_returns_time_query_instance_even_when_defaults_are_used() -> None:
    # No ``now`` injected: must not raise and must produce tz-aware bounds.
    parsed = parse_time_query("notes yesterday")
    assert isinstance(parsed, TimeQuery)
    assert parsed.start is not None and parsed.start.tzinfo is not None


# ----------------------------------------------------------- required phrases


def test_today() -> None:
    parsed = _parse("today")
    assert parsed.cleaned_query == ""
    assert parsed.start == datetime(2026, 10, 7, tzinfo=TZ)
    assert parsed.end == datetime(2026, 10, 8, tzinfo=TZ)


def test_yesterday_strips_phrase_from_a_longer_query() -> None:
    parsed = _parse("react hooks article yesterday")
    assert parsed.cleaned_query == "react hooks article"
    assert parsed.start == datetime(2026, 10, 6, tzinfo=TZ)
    assert parsed.end == datetime(2026, 10, 7, tzinfo=TZ)


def test_phrase_in_the_middle_is_removed_cleanly() -> None:
    parsed = _parse("notes from yesterday about budget")
    assert parsed.cleaned_query == "notes from about budget"
    assert parsed.has_range


def test_empty_cleaned_query_for_a_bare_time_phrase() -> None:
    for query in ("yesterday", "today", "this week", "last night"):
        parsed = _parse(query)
        assert parsed.cleaned_query == "", query
        assert parsed.has_range, query


def test_this_morning_uses_configured_window() -> None:
    parsed = _parse("link about react hooks from this morning")
    assert parsed.cleaned_query == "link about react hooks from"
    assert parsed.start == datetime(2026, 10, 7, 5, tzinfo=TZ)
    assert parsed.end == datetime(2026, 10, 7, 12, tzinfo=TZ)


def test_this_afternoon_and_evening() -> None:
    afternoon = _parse("meeting this afternoon")
    assert afternoon.start == datetime(2026, 10, 7, 12, tzinfo=TZ)
    assert afternoon.end == datetime(2026, 10, 7, 17, tzinfo=TZ)

    evening = _parse("dinner this evening")
    assert evening.start == datetime(2026, 10, 7, 17, tzinfo=TZ)
    assert evening.end == datetime(2026, 10, 7, 21, tzinfo=TZ)


def test_last_night_is_yesterdays_night_window() -> None:
    parsed = _parse("last night")
    assert parsed.start == datetime(2026, 10, 6, 21, tzinfo=TZ)
    assert parsed.end == datetime(2026, 10, 7, 5, tzinfo=TZ)


def test_n_hours_ago_is_a_window_centred_on_the_offset() -> None:
    parsed = _parse("copied 5 hours ago")
    assert parsed.cleaned_query == "copied"
    assert parsed.start == datetime(2026, 10, 7, 9, 0, tzinfo=TZ)
    assert parsed.end == datetime(2026, 10, 7, 10, 0, tzinfo=TZ)


def test_n_days_ago_is_the_whole_day() -> None:
    parsed = _parse("2 days ago")
    assert parsed.start == datetime(2026, 10, 5, tzinfo=TZ)
    assert parsed.end == datetime(2026, 10, 6, tzinfo=TZ)


def test_n_weeks_ago() -> None:
    parsed = _parse("1 week ago")
    assert parsed.start == datetime(2026, 9, 30, tzinfo=TZ)
    assert parsed.end == datetime(2026, 10, 1, tzinfo=TZ)


def test_n_minutes_ago_uses_a_minute_window() -> None:
    parsed = _parse("30 minutes ago")
    assert parsed.start == datetime(2026, 10, 7, 13, 59, 30, tzinfo=TZ)
    assert parsed.end == datetime(2026, 10, 7, 14, 0, 30, tzinfo=TZ)


def test_last_monday_from_a_wednesday() -> None:
    parsed = _parse("deploy notes last Monday")
    assert parsed.cleaned_query == "deploy notes"
    assert parsed.start == datetime(2026, 10, 5, tzinfo=TZ)
    assert parsed.end == datetime(2026, 10, 6, tzinfo=TZ)


def test_last_monday_on_a_monday_means_the_previous_week() -> None:
    monday = datetime(2026, 10, 5, 10, 0, tzinfo=TZ)
    parsed = _parse("last monday", now=monday)
    assert parsed.start == datetime(2026, 9, 28, tzinfo=TZ)
    assert parsed.end == datetime(2026, 9, 29, tzinfo=TZ)


def test_this_week_starts_on_monday_by_default() -> None:
    parsed = _parse("this week")
    assert parsed.start == datetime(2026, 10, 5, tzinfo=TZ)
    assert parsed.end == datetime(2026, 10, 12, tzinfo=TZ)


def test_last_week_is_the_previous_full_week() -> None:
    parsed = _parse("last week")
    assert parsed.start == datetime(2026, 9, 28, tzinfo=TZ)
    assert parsed.end == datetime(2026, 10, 5, tzinfo=TZ)


def test_between_yesterday() -> None:
    parsed = _parse("between 2pm and 4pm yesterday")
    assert parsed.cleaned_query == ""
    assert parsed.start == datetime(2026, 10, 6, 14, tzinfo=TZ)
    assert parsed.end == datetime(2026, 10, 6, 16, tzinfo=TZ)


def test_between_defaults_to_today() -> None:
    parsed = _parse("pricing between 2pm and 4pm")
    assert parsed.cleaned_query == "pricing"
    assert parsed.start == datetime(2026, 10, 7, 14, tzinfo=TZ)
    assert parsed.end == datetime(2026, 10, 7, 16, tzinfo=TZ)


def test_between_handles_an_overnight_range() -> None:
    parsed = _parse("between 11pm and 1am")
    assert parsed.start == datetime(2026, 10, 7, 23, tzinfo=TZ)
    assert parsed.end == datetime(2026, 10, 8, 1, tzinfo=TZ)


def test_between_with_minutes() -> None:
    parsed = _parse("between 2:15pm and 4:45pm yesterday")
    assert parsed.start == datetime(2026, 10, 6, 14, 15, tzinfo=TZ)
    assert parsed.end == datetime(2026, 10, 6, 16, 45, tzinfo=TZ)


# ------------------------------------------------------------------- no misfire


def test_todays_meeting_notes_filters_to_today_and_cleans_possessive() -> None:
    parsed = _parse("today's meeting notes")
    assert parsed.cleaned_query == "meeting notes"
    assert parsed.start == datetime(2026, 10, 7, tzinfo=TZ)
    assert parsed.end == datetime(2026, 10, 8, tzinfo=TZ)
    assert not parsed.ambiguous


def test_monday_com_link_is_not_read_as_a_weekday() -> None:
    parsed = _parse("monday.com link")
    assert parsed.cleaned_query == "monday.com link"
    assert not parsed.has_range
    assert parsed.ambiguous
    assert "monday.com" in (parsed.note or "")


def test_bare_clock_time_is_ambiguous_not_a_filter() -> None:
    parsed = _parse("meeting at 5pm with Rahul")
    assert parsed.cleaned_query == "meeting at 5pm with Rahul"
    assert not parsed.has_range
    assert parsed.ambiguous
    assert "clock time" in (parsed.note or "")


# ------------------------------------------------------------- configuration


def test_day_part_windows_are_configurable() -> None:
    config = _config(day_part_morning=(6, 11), day_part_night=(22, 3))
    morning = _parse("this morning", config=config)
    assert morning.start == datetime(2026, 10, 7, 6, tzinfo=TZ)
    assert morning.end == datetime(2026, 10, 7, 11, tzinfo=TZ)

    night = _parse("last night", config=config)
    assert night.start == datetime(2026, 10, 6, 22, tzinfo=TZ)
    assert night.end == datetime(2026, 10, 7, 3, tzinfo=TZ)


def test_week_start_is_configurable() -> None:
    config = _config(week_starts_on=6)  # Sunday
    parsed = _parse("this week", config=config)
    assert parsed.start == datetime(2026, 10, 4, tzinfo=TZ)  # Sunday
    assert parsed.end == datetime(2026, 10, 11, tzinfo=TZ)


def test_describe_reports_the_matched_phrase_and_bounds() -> None:
    parsed = _parse("yesterday")
    described = parsed.describe()
    assert "yesterday" in described
    assert "2026-10-06 00:00" in described
    assert "2026-10-07 00:00" in described


def test_describe_is_empty_without_a_range() -> None:
    assert _parse("no time here").describe() == ""


# --------------------------------------------------------------------- orders


def test_between_is_matched_before_yesterday() -> None:
    # "between 2pm and 4pm yesterday" must not be consumed by the "yesterday"
    # rule (which would yield a whole-day window instead).
    parsed = _parse("between 2pm and 4pm yesterday")
    assert parsed.end - parsed.start == timedelta(hours=2)
    assert parsed.matched_phrase is not None
    assert parsed.matched_phrase.lower().startswith("between")


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        "notification",
        "the quick brown fox",
        "a@b.com",
    ],
)
def test_non_time_queries_have_no_range_and_are_not_ambiguous(query: str) -> None:
    parsed = _parse(query)
    assert not parsed.has_range
    assert not parsed.ambiguous
