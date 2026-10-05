"""Tests for :mod:`clipmind.capture.detect`."""

from __future__ import annotations

import pytest

from clipmind.capture.detect import detect_content_type


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("https://example.com/docs?q=1", "url"),
        ("http://localhost:8000/api/orders", "url"),
        ("www.react.dev", "url"),
        ("andrew@example.com", "email"),
        ("first.last+tag@sub.domain.co.uk", "email"),
        ("42", "number"),
        ("-3.14", "number"),
        ("1,234,567.89", "number"),
        ("Because I said so, and that's final.", "text"),
        ("meeting notes for monday", "text"),
    ],
)
def test_simple_types(text: str, expected: str) -> None:
    assert detect_content_type(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "SELECT id, total FROM orders WHERE created_at > '2024-01-01' ORDER BY total DESC;",
        "def add(a, b):\n    return a + b",
        "const useThing = () => {\n  return 1;\n};",
        "<div class=\"card\">\n  <h1>Hello</h1>\n</div>",
        "#include <stdio.h>\nint main(void) { return 0; }",
        "import requests\n\nresponse = requests.get('https://x.dev')\nprint(response.status_code)",
    ],
)
def test_code_detection(text: str) -> None:
    assert detect_content_type(text) == "code"


@pytest.mark.parametrize(
    "text",
    [
        "Hello world",
        "",
        "   ",
        "Buy milk; eggs; bread",
        "This (mostly) works!",
    ],
)
def test_prose_is_not_code(text: str) -> None:
    assert detect_content_type(text) == "text"
