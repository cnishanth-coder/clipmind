"""Time-aware query parsing.

Turns a natural query like ``"react hooks article yesterday"`` into a cleaned
query (``"react hooks article"``) plus a local-time ``[start, end)`` window.

Why a hand-written parser instead of ``dateparser``
---------------------------------------------------
``dateparser`` is ~10 MB of locale data and, more importantly, is *eager*: it
will happily read ``"monday.com"`` as a weekday and ``"5pm"`` in ordinary prose
as a clock time. The requirement is explicitly the opposite ("must not
misfire"), and the set of phrases to support is a small, fixed list. A ~150-line
regex parser is deterministic, has no dependencies, and makes every supported
phrase an explicit, testable rule.

Design rules
------------
* All arithmetic is in the **local timezone**; ``now`` is injectable so tests
  can pin it without ``freezegun``.
* A phrase is only consumed when matched by a specific rule. A bare weekday
  (``"monday"``) or a bare clock time (``"5pm"``) is **ambiguous**: it is left
  in the cleaned query and flagged, rather than silently filtering.
* If nothing matches, the query is returned untouched with no range.
* If the cleaned query ends up empty (e.g. the query was just ``"yesterday"``),
  the caller should return everything in the range, newest first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional

from ..config import Config

#: Weekday names in Python's ``date.weekday()`` order (Monday == 0).
WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

_WEEKDAY_ALT = "|".join(WEEKDAYS)

#: A day-part name -> Config field mapping.
_DAY_PART_FIELDS = {
    "morning": "day_part_morning",
    "afternoon": "day_part_afternoon",
    "evening": "day_part_evening",
    "night": "day_part_night",
}


@dataclass(frozen=True, slots=True)
class TimeQuery:
    """A parsed query split into free text and an optional time window.

    Attributes:
        raw: The original query string.
        cleaned_query: The query with the recognised time phrase removed.
        start: Inclusive lower bound (local timezone), or ``None``.
        end: Exclusive upper bound (local timezone), or ``None``.
        matched_phrase: The exact substring that was recognised, if any.
        ambiguous: True when a time-like phrase was seen but deliberately left
            in :attr:`cleaned_query` because it may not be a time reference.
        note: Human-readable explanation, shown by ``--explain``.
    """

    raw: str
    cleaned_query: str
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    matched_phrase: Optional[str] = None
    ambiguous: bool = False
    note: Optional[str] = None

    @property
    def has_range(self) -> bool:
        """True when at least one of :attr:`start`/:attr:`end` is set."""
        return self.start is not None or self.end is not None

    def describe(self) -> str:
        """Return a short human-readable description for CLI output."""
        if not self.has_range:
            return ""
        start = self.start.strftime("%Y-%m-%d %H:%M") if self.start else "…"
        end = self.end.strftime("%Y-%m-%d %H:%M") if self.end else "…"
        if self.matched_phrase:
            return f"{self.matched_phrase}  ({start} \u2192 {end})"
        return f"{start} \u2192 {end}"


# --------------------------------------------------------------------- helpers


def _start_of_day(moment: datetime) -> datetime:
    """Return midnight local time on the day of ``moment``."""
    return moment.replace(hour=0, minute=0, second=0, microsecond=0)


def _day_window(moment: datetime) -> tuple[datetime, datetime]:
    """Return the ``[start, end)`` window covering the whole day of ``moment``."""
    start = _start_of_day(moment)
    return start, start + timedelta(days=1)


def _day_part_window(
    moment: datetime, bounds: tuple[int, int]
) -> tuple[datetime, datetime]:
    """Return the window for a day-part on the day of ``moment``.

    ``bounds`` is a local ``(start_hour, end_hour)``. When ``end_hour`` is not
    greater than ``start_hour`` the window wraps past midnight (as ``night``
    does).
    """
    start_hour, end_hour = bounds
    start = _start_of_day(moment) + timedelta(hours=start_hour)
    if end_hour > start_hour:
        end = _start_of_day(moment) + timedelta(hours=end_hour)
    elif end_hour < start_hour:
        end = _start_of_day(moment) + timedelta(days=1, hours=end_hour)
    else:
        end = start + timedelta(days=1)
    return start, end


def _start_of_week(moment: datetime, week_starts_on: int) -> datetime:
    """Return midnight on the configurable first day of ``moment``'s week."""
    day = _start_of_day(moment)
    delta = (day.weekday() - week_starts_on) % 7
    return day - timedelta(days=delta)


def _last_weekday_window(
    moment: datetime, target: int
) -> tuple[datetime, datetime]:
    """Return the window for the most recent ``target`` weekday before today.

    ``"last Monday"`` on a Wednesday means two days ago; on a Monday it means
    the Monday of the previous week (strictly before today).
    """
    today = _start_of_day(moment)
    delta = (today.weekday() - target) % 7
    if delta == 0:
        delta = 7
    day = today - timedelta(days=delta)
    return day, day + timedelta(days=1)


def _clock_time(hour: int, minute: int, meridiem: Optional[str]) -> tuple[int, int]:
    """Convert a parsed ``(hour, minute, am/pm)`` into 24-hour time."""
    if meridiem == "pm" and hour != 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    return hour % 24, minute


def _clean_query(text: str) -> str:
    """Tidy a query after a time phrase has been removed.

    Drops possessive remnants (``"today's"`` leaves ``"'s"`` behind) and
    normalises whitespace/punctuation so the remaining text is usable as the
    semantic/keyword query.
    """
    text = re.sub(r"'s\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t,.;:-")


# --------------------------------------------------------------------- matchers


@dataclass(frozen=True, slots=True)
class _Match:
    """Internal result of one matcher rule."""

    phrase: str
    start: datetime
    end: datetime
    note: Optional[str] = None


_Matcher = Callable[[str, datetime, Config], Optional[_Match]]


# "between 2pm and 4pm [today|yesterday]"
_BETWEEN_RE = re.compile(
    r"\bbetween\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\s+and\s+"
    r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)(?:\s+(today|yesterday))?\b",
    re.IGNORECASE,
)

# "N hours/days/weeks/minutes ago"
_AGO_RE = re.compile(
    r"\b(\d+)\s+(hour|hours|day|days|week|weeks|minute|minutes)\s+ago\b",
    re.IGNORECASE,
)

_THIS_DAYPART_RE = re.compile(
    r"\bthis\s+(morning|afternoon|evening|night)\b", re.IGNORECASE
)
_LAST_NIGHT_RE = re.compile(r"\blast\s+night\b", re.IGNORECASE)
_LAST_WEEKDAY_RE = re.compile(rf"\blast\s+({_WEEKDAY_ALT})\b", re.IGNORECASE)
_WEEK_RE = re.compile(r"\b(this|last)\s+week\b", re.IGNORECASE)
_TODAY_YESTERDAY_RE = re.compile(r"\b(today|yesterday)\b", re.IGNORECASE)

# Ambiguity probes: a time shape that is *not* specific enough to filter on.
_BARE_WEEKDAY_RE = re.compile(rf"\b({_WEEKDAY_ALT})\b", re.IGNORECASE)
_BARE_CLOCK_RE = re.compile(
    r"(?:\bat\s+)?\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", re.IGNORECASE
)


def _match_between(text: str, now: datetime, config: Config) -> Optional[_Match]:
    """Match ``between 2pm and 4pm [yesterday]``."""
    match = _BETWEEN_RE.search(text)
    if match is None:
        return None
    start_hour, start_min = _clock_time(
        int(match.group(1)), int(match.group(2) or 0), match.group(3).lower()
    )
    end_hour, end_min = _clock_time(
        int(match.group(4)), int(match.group(5) or 0), match.group(6).lower()
    )
    day_word = (match.group(7) or "today").lower()
    base = now - timedelta(days=1) if day_word == "yesterday" else now
    day_start = _start_of_day(base)
    start = day_start + timedelta(hours=start_hour, minutes=start_min)
    end = day_start + timedelta(hours=end_hour, minutes=end_min)
    if end <= start:  # overnight range such as "between 11pm and 1am"
        end += timedelta(days=1)
    return _Match(match.group(0), start, end)


def _match_ago(text: str, now: datetime, config: Config) -> Optional[_Match]:
    """Match ``N hours/days/weeks/minutes ago``."""
    match = _AGO_RE.search(text)
    if match is None:
        return None
    count = int(match.group(1))
    unit = match.group(2).lower().rstrip("s")
    if unit == "hour":
        # A one-hour window centred on "N hours ago" (colloquial "around").
        centre = now - timedelta(hours=count)
        return _Match(match.group(0), centre - timedelta(minutes=30), centre + timedelta(minutes=30))
    if unit == "minute":
        centre = now - timedelta(minutes=count)
        return _Match(match.group(0), centre - timedelta(seconds=30), centre + timedelta(seconds=30))
    days = count * 7 if unit == "week" else count
    return _Match(match.group(0), *_day_window(now - timedelta(days=days)))


def _match_this_daypart(text: str, now: datetime, config: Config) -> Optional[_Match]:
    """Match ``this morning/afternoon/evening/night``."""
    match = _THIS_DAYPART_RE.search(text)
    if match is None:
        return None
    part = match.group(1).lower()
    bounds = getattr(config, _DAY_PART_FIELDS[part])
    start, end = _day_part_window(now, bounds)
    return _Match(match.group(0), start, end)


def _match_last_night(text: str, now: datetime, config: Config) -> Optional[_Match]:
    """Match ``last night`` (yesterday's night window)."""
    match = _LAST_NIGHT_RE.search(text)
    if match is None:
        return None
    start, end = _day_part_window(now - timedelta(days=1), config.day_part_night)
    return _Match(match.group(0), start, end)


def _match_last_weekday(text: str, now: datetime, config: Config) -> Optional[_Match]:
    """Match ``last Monday`` / ``last Friday``."""
    match = _LAST_WEEKDAY_RE.search(text)
    if match is None:
        return None
    start, end = _last_weekday_window(now, WEEKDAYS[match.group(1).lower()])
    return _Match(match.group(0), start, end)


def _match_week(text: str, now: datetime, config: Config) -> Optional[_Match]:
    """Match ``this week`` / ``last week``."""
    match = _WEEK_RE.search(text)
    if match is None:
        return None
    this_week = _start_of_week(now, config.week_starts_on)
    if match.group(1).lower() == "this":
        return _Match(match.group(0), this_week, this_week + timedelta(days=7))
    return _Match(match.group(0), this_week - timedelta(days=7), this_week)


def _match_today_yesterday(text: str, now: datetime, config: Config) -> Optional[_Match]:
    """Match ``today`` / ``yesterday``."""
    match = _TODAY_YESTERDAY_RE.search(text)
    if match is None:
        return None
    base = now - timedelta(days=1) if match.group(1).lower() == "yesterday" else now
    return _Match(match.group(0), *_day_window(base))


#: Matchers in priority order: the most specific phrase shapes run first so
#: "between 2pm and 4pm yesterday" is not first consumed by the "yesterday" rule.
_MATCHERS: tuple[tuple[str, _Matcher], ...] = (
    ("between", _match_between),
    ("ago", _match_ago),
    ("this_daypart", _match_this_daypart),
    ("last_night", _match_last_night),
    ("last_weekday", _match_last_weekday),
    ("week", _match_week),
    ("today_yesterday", _match_today_yesterday),
)


def _detect_ambiguity(text: str) -> Optional[str]:
    """Return a note when ``text`` still looks time-related but is ambiguous."""
    if _BARE_CLOCK_RE.search(text):
        return (
            "a bare clock time was left in the query because it may be part of "
            "the content; add a day (e.g. 'yesterday') or 'between' to filter"
        )
    if _BARE_WEEKDAY_RE.search(text):
        return (
            "a bare weekday was left in the query because it may be part of the "
            "content (e.g. 'monday.com'); say 'last <weekday>' to filter"
        )
    return None


def parse_time_query(
    query: str,
    *,
    now: Optional[datetime] = None,
    config: Optional[Config] = None,
) -> TimeQuery:
    """Split ``query`` into free text plus an optional local time window.

    Args:
        query: The raw user query.
        now: Reference instant (local timezone). Naive values are assumed to be
            local; defaults to ``datetime.now().astimezone()``. Injectable so
            tests are deterministic without ``freezegun``.
        config: Configuration supplying the day-part hour windows and week start.

    Returns:
        A :class:`TimeQuery`. When no phrase matches, ``start``/``end`` are
        ``None`` and ``cleaned_query`` equals the input.
    """
    config = config or Config()
    if now is None:
        now = datetime.now().astimezone()
    elif now.tzinfo is None:
        now = now.astimezone()

    raw = query or ""
    cleaned = raw

    for _name, matcher in _MATCHERS:
        found = matcher(raw, now, config)
        if found is not None:
            cleaned = _clean_query(cleaned.replace(found.phrase, " ", 1))
            return TimeQuery(
                raw=raw,
                cleaned_query=cleaned,
                start=found.start,
                end=found.end,
                matched_phrase=found.phrase,
                note=found.note,
            )

    # No range recognised - keep the query intact, but report ambiguous shapes.
    note = _detect_ambiguity(raw)
    return TimeQuery(
        raw=raw,
        cleaned_query=_clean_query(raw),
        ambiguous=note is not None,
        note=note,
    )
