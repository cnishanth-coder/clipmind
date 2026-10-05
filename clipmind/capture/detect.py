"""Heuristic content-type detection for clipboard entries.

The classifier intentionally stays simple and dependency-free: it answers the
question "what kind of thing did the user copy?" well enough to power type
badges and the Phase 5 All / Code / Links / Text filters.

Types produced (and stored in the ``content_type`` column):

``url``    A single http(s)/ftp/www token.
``email``  A single ``local@domain`` token.
``number`` A single numeric token (int, float, thousands separators, 1_000, etc.).
``code``   Text that looks like source code, SQL, shell or markup.
``text``   Anything else, including prose and short phrases.
"""

from __future__ import annotations

import re

#: All content types the detector can return, in filter order.
CONTENT_TYPES = ("url", "code", "email", "number", "text")

# Human-friendly labels used by the CLI (and later the UI).
TYPE_LABELS: dict[str, str] = {
    "url": "link",
    "code": "code",
    "email": "email",
    "number": "number",
    "text": "text",
}

_URL_RE = re.compile(
    r"^(?:https?|ftp)://\S+$|^www\.\S+\.\S+$",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
_NUMBER_RE = re.compile(
    r"^[+-]?(?:\d{1,3}(?:[,_ ]\d{3})+|\d+)(?:[.,]\d+)?(?:[eE][+-]?\d+)?%?$"
)

# Distinctive code keywords only: common English words (``this``, ``new``,
# ``from``, ``let``, ``as``, ``with``) are deliberately absent so ordinary
# prose is not mistaken for source code. Matching is case-sensitive; SQL is
# handled separately by :data:`_SQL_RE` because SQL keywords are usually
# uppercase.
_CODE_KEYWORDS = re.compile(
    r"""\b(?:
        def|class|import|return|lambda|yield|async|await|self
        |function|const|typeof|interface|enum|struct|impl|fn
        |public|private|protected|static|void|unsigned|sprintf
        |func|package|println|printf|scanf|stdout|stderr
        |elif|else|except|finally|raise|switch|break|continue
        |echo|grep|awk|sed|sudo|npm|pip|docker|kubectl|chmod
    )\b""",
    re.VERBOSE,
)

# Strong single-token signals. ``\#`` is escaped because a bare ``#`` would
# start a comment under ``re.VERBOSE``, silently creating an empty alternative
# that matches every string (and classified all prose as code).
_CODE_STRONG = re.compile(
    r"(?:\$\{|\{\{|=>|->|::|</|/>"
    r"|\#\s*include|</?[A-Za-z][\w\-]*>"
    r"|\)\s*[:{]|\}\s*else"
    r"|\b\w+\.\w+\(|\b\w+\([^\n)]*\)\s*;"
    r")"
)

# Case-sensitive on purpose: SQL keywords are conventionally uppercase, while
# the same words in lowercase prose ("select from the menu") should not count.
_SQL_RE = re.compile(
    r"\b(?:SELECT|FROM|WHERE|INSERT\s+INTO|UPDATE|DELETE\s+FROM|CREATE\s+TABLE"
    r"|ALTER\s+TABLE|DROP\s+TABLE|GROUP\s+BY|ORDER\s+BY|JOIN|VALUES)\b"
)

_LINE_END_RE = re.compile(r"[;{}\[\]]\s*$")


def looks_like_url(text: str) -> bool:
    """Return True when ``text`` is a single URL-like token."""
    stripped = text.strip()
    return "\n" not in stripped and bool(_URL_RE.match(stripped))


def looks_like_email(text: str) -> bool:
    """Return True when ``text`` is a single email address."""
    stripped = text.strip()
    return "\n" not in stripped and bool(_EMAIL_RE.match(stripped))


def looks_like_number(text: str) -> bool:
    """Return True when ``text`` is a single numeric token."""
    stripped = text.strip()
    return "\n" not in stripped and bool(_NUMBER_RE.match(stripped))


def looks_like_code(text: str) -> bool:
    """Return True when ``text`` heuristically looks like source code or markup.

    The check accumulates weak and strong signals and requires a score of at
    least 3, which keeps ordinary prose containing parentheses or semicolons
    from being mislabelled.
    """
    if not text.strip():
        return False

    score = 0
    if _CODE_KEYWORDS.search(text):
        score += 2
    if _CODE_STRONG.search(text):
        score += 2

    punctuation = text.count(";") + text.count("{") + text.count("}") + text.count("=")
    if punctuation >= 1 and re.search(r"[;{}]|\w\s*=\s*\S", text):
        score += 1

    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) >= 2:
        indented = sum(1 for line in lines if line.startswith(("    ", "\t")))
        if indented >= 1:
            score += 1
        ending = sum(1 for line in lines if _LINE_END_RE.search(line))
        if ending / len(lines) >= 0.5:
            score += 1

    # Two or more SQL keywords is a strong signal; a single one only counts as
    # a weak nudge so stray uppercase words do not label prose as code.
    sql_hits = len(_SQL_RE.findall(text))
    if sql_hits >= 2:
        score += 3
    elif sql_hits == 1:
        score += 2

    return score >= 3


def detect_content_type(text: str) -> str:
    """Classify ``text`` and return one of :data:`CONTENT_TYPES`.

    Args:
        text: The raw clipboard text.

    Returns:
        ``"url"``, ``"email"``, ``"number"``, ``"code"`` or ``"text"``.
    """
    if not text or not text.strip():
        return "text"
    if looks_like_url(text):
        return "url"
    if looks_like_email(text):
        return "email"
    if looks_like_number(text):
        return "number"
    if looks_like_code(text):
        return "code"
    return "text"
