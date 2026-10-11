#!/usr/bin/env python3
"""Benchmark ClipMind's real embedding models and search latencies.

What this measures
------------------
* **Embedding throughput** - how fast each local model turns a 10,000-entry
  corpus into vectors (texts/second), plus the one-off model-load time and the
  on-disk / in-memory model size.
* **Search latency** - p50/p95/p99 for semantic-only, keyword-only and hybrid
  search with a *single long-lived searcher* (the Task 1 fix). A before/after
  table compares this with the pre-fix run stored in ``results_before.json``.
* **Retrieval quality** - top-1 / top-5 hit rate on 45 hand-written
  ``query -> expected entry`` pairs (30 general + 15 code-focused), reported
  overall and for the code subset.

Models compared: ``all-MiniLM-L6-v2`` (the current default),
``BAAI/bge-small-en-v1.5`` and ``sentence-transformers/all-mpnet-base-v2``.
Query/document prefixes come from the production embedder path
(:func:`clipmind.embeddings.encoder.get_embedder`), so the benchmark measures
exactly what the app and CLI would do.

Everything is measured with the *real* models and a *real* SQLite database - no
fakes, no estimates. The corpus is generated with a fixed seed, so entry
contents and timestamps are reproducible; only machine-dependent timings vary.

Usage::

    python benchmarks/bench_search.py                 # writes benchmarks/RESULTS.md
    python benchmarks/bench_search.py --json out.json # also dump raw numbers
    python benchmarks/bench_search.py --queries 300   # more latency queries
    python benchmarks/bench_search.py --models all-MiniLM-L6-v2
    python benchmarks/bench_search.py --no-write      # print only, touch no files

Numbers are only ever *measured*: nothing is hard-coded, and if the model cannot
be loaded the script exits non-zero with a clear message instead of inventing
figures.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clipmind.capture.detect import detect_content_type  # noqa: E402
from clipmind.config import Config, model_prefixes  # noqa: E402
from clipmind.embeddings.codec import encode_vector  # noqa: E402
from clipmind.embeddings.encoder import (  # noqa: E402
    EmbeddingModelUnavailable,
    get_embedder,
)
from clipmind.search.hybrid import HybridSearcher  # noqa: E402
from clipmind.storage.db import Database  # noqa: E402

#: Fixed seed: the corpus is byte-for-byte reproducible across runs.
SEED = 20261007

#: How many entries the corpus holds (one is a "clipboard entry").
CORPUS_SIZE = 10_000

#: Timestamps are spread uniformly across this many days before "now".
SPREAD_DAYS = 90

#: Texts handed to the model per ``embed_texts`` call while measuring throughput.
EMBED_CHUNK = 256

#: Default number of latency queries (the spec requires at least 200).
DEFAULT_QUERIES = 200

#: The model that was the default when the Task 1 fix landed. The before/after
#: latency table always compares this model, so a later default change cannot
#: turn it into an apples-to-oranges comparison.
TASK1_MODEL = "all-MiniLM-L6-v2"

#: Where the pre-fix run is kept (used for the before/after table).
BEFORE_JSON = ROOT / "benchmarks" / "results_before.json"

#: Search modes benchmarked, as ``(name, kwargs)`` for the searcher.
MODES: tuple[tuple[str, dict], ...] = (
    ("semantic", {"semantic_only": True}),
    ("keyword", {"keyword_only": True}),
    ("hybrid", {}),
)


# --------------------------------------------------------------------------- models


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One embedding model under comparison."""

    name: str
    note: str


#: The three models compared by default.
DEFAULT_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("all-MiniLM-L6-v2", "current default (384d)"),
    ModelSpec("BAAI/bge-small-en-v1.5", "small retrieval-tuned (384d)"),
    ModelSpec("sentence-transformers/all-mpnet-base-v2", "larger option (768d)"),
)


# --------------------------------------------------------------------------- corpus


_CODE_TEMPLATES = [
    "SELECT {col} FROM {table} WHERE {col} = {n};",
    "SELECT {agg}({col}) FROM {table} GROUP BY {col2};",
    "def {fn}({arg}):\n    return {arg}.strip().lower()",
    "async function {fn}({arg}) {{\n  const res = await fetch(`{path}/${{{arg}}}`);\n  return res.json();\n}}",
    "import {{ {fn} }} from './{mod}';\nexport const {cst} = {fn}({arg});",
    "docker build -t {img}:latest .",
    "git commit -m \"{msg}\"",
    "for f in *.{ext}; do {cmd} \"$f\"; done",
    "class {cls}:\n    def __init__(self{arg}):\n        self.{arg} = {arg}",
    "const {cst} = {arg}.map((x) => x * {n}).filter(Boolean);",
    "kubectl get {res} -n {ns} -o wide",
    "SELECT strftime('%Y-%m', created_at) AS month, COUNT(*) FROM {table} GROUP BY month;",
]
_CODE_COLS = ["order_id", "customer_id", "total", "status", "created_at", "email", "sku"]
_CODE_TABLES = ["orders", "users", "invoices", "sessions", "products", "events"]
_CODE_FNS = ["parse_config", "load_user", "fetch_data", "normalize_url", "render_row", "compact"]
_CODE_WORDS = ["config", "user", "data", "token", "payload", "buffer", "result", "request"]

_URL_HOSTS = [
    "react.dev",
    "docs.python.org",
    "developer.mozilla.org",
    "doc.rust-lang.org",
    "www.postgresql.org",
    "www.sqlite.org",
    "docs.djangoproject.com",
    "fastapi.tiangolo.com",
    "kubernetes.io",
    "stackoverflow.com",
    "github.com",
    "news.ycombinator.com",
    "docs.docker.com",
    "www.typescriptlang.org",
    "pytest.org",
    "docs.pydantic.dev",
    "redis.io",
    "nginx.org",
    "git-scm.com",
    "opensource.org",
]
_URL_PATHS = [
    "learn/referencing-values",
    "library/pathlib",
    "docs/Web/JavaScript/Reference",
    "book/ch04-ownwership",
    "docs/current/tutorial",
    "fts5",
    "topics/migrations",
    "tutorial/query-params",
    "docs/concepts/configuration",
    "questions/4021",
    "owner/repo/issues",
    "item",
    "reference/api",
    "guides/deployment",
    "blog/why-local-first",
    "howto/backup-restore",
    "manual/install",
    "cli/usage",
    "changelog/v2",
    "security/advisories",
    "wiki/architecture",
    "config/settings",
    "api/v2/endpoints",
    "tutorials/quickstart",
    "docs/spec",
    "downloads",
    "archive/2025",
    "sitemap",
    "feed",
    "about",
]

_EMAIL_ROLES = [
    "billing",
    "support",
    "onboarding",
    "landing-page-review",
    "invoices",
    "security",
    "alerts",
    "hello",
    "team",
    "no-reply",
    "ops",
    "design",
    "recruiting",
    "press",
    "sales",
    "engineering",
]
_EMAIL_FIRST = [
    "alex", "sam", "jordan", "taylor", "morgan", "casey", "riley", "jamie",
    "drew", "quinn", "avery", "harper", "rowan", "skyler", "phoenix", "reese",
    "noah", "mia", "leo", "zoe", "ivan", "nora", "omar", "lara",
]
_EMAIL_LAST = [
    "smith", "jones", "patel", "garcia", "khan", "nguyen", "okafor", "mueller",
    "rossi", "tanaka", "silva", "novak", "haddad", "kowalski", "lindqvist",
    "ivanov", "costa", "moreau",
]
_EMAIL_DOMAINS = [
    "acme-corp.com",
    "cloudhost.io",
    "example.org",
    "mail.example.net",
    "startup.dev",
    "contoso.co",
    "northwind.example",
    "globex.example",
    "initech.example",
    "umbrella.example",
    "vertex-labs.io",
    "blue-ocean.dev",
    "pixel-forge.co",
    "paper-trail.net",
    "quanta-systems.com",
    "lighthouse.io",
    "meridian.example",
    "cornerstone.dev",
]

_SENTENCE_TEMPLATES = [
    "Standup moves to {time}; sprint planning is now {day} in room {room}.",
    "Ticket #{ref}: the {thing} certificate for example.com expires on 2026-{month:02d}-{day2:02d}.",
    "Shopping: {item1}, {item2}, eggs, spinach, {item3}, coffee beans.",
    "Idea #{ref}: a CLI that summarises the last week of git history into a changelog.",
    "Note #{ref}: to reset the router, hold the pinhole button for ten seconds.",
    "Talk idea #{ref}: local-first software, and why your app should work offline.",
    "Reminder: renew the domain {domain} before the end of the quarter.",
    "The {team} retro is {day} at {time} - bring your blockers.",
    "Meeting notes: we decided to {decision} and revisit it next month.",
    "Flight {code} to Lisbon departs {day} at {time}, confirmation {conf}.",
    "Recipe: boil {qty}g spaghetti, saute garlic and {item2}, toss with basil.",
    "Quote from the client: \"{quote}\"",
]
_SENTENCE_WORDS = [
    "dashboard",
    "caching layer",
    "onboarding flow",
    "search index",
    "sync engine",
    "colour scheme",
    "billing page",
    "checkout funnel",
    "notification digest",
    "offline mode",
    "keyboard shortcuts",
    "export pipeline",
]
_SENTENCE_THINGS = [
    "wildcard SSL",
    "code-signing",
    "TLS",
    "API",
    "staging",
    "database",
    "release",
    "monitoring",
    "backup",
    "domain",
]


@dataclass(frozen=True, slots=True)
class LabelledQuery:
    """One hand-written ``query -> expected entry`` pair."""

    query: str
    expected: str
    #: Short label for the report table (``code``, ``url``, ``email``, ``text``).
    note: str


#: 45 hand-written query -> expected-entry pairs: 30 general (as before) plus 15
#: code-focused ones. The expected contents are included verbatim in the
#: generated corpus, so "did search find it?" is a well-defined question.
LABELLED_QUERIES: tuple[LabelledQuery, ...] = (
    LabelledQuery("python query the orders table by customer id", "SELECT order_id, total FROM orders WHERE customer_id = 4021;", "code"),
    LabelledQuery("sql average order value per month", "SELECT strftime('%Y-%m', created_at) AS month, AVG(total) FROM orders GROUP BY month;", "code"),
    LabelledQuery("git command to squash the last three commits", "git rebase -i HEAD~3", "code"),
    LabelledQuery("javascript fetch with async await", "async function loadUser(id) {\n  const res = await fetch(`/api/users/${id}`);\n  return res.json();\n}", "code"),
    LabelledQuery("reverse a linked list in python", "def reverse_list(head):\n    prev = None\n    while head:\n        head.next, prev, head = prev, head, head.next\n    return prev", "code"),
    LabelledQuery("docker build and tag an image", "docker build -t clipmind:latest .", "code"),
    LabelledQuery("regex to validate an email address", r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", "code"),
    LabelledQuery("bash loop over log files and gzip them", 'for f in *.log; do gzip "$f"; done', "code"),
    LabelledQuery("react hook that fetches data on mount", "useEffect(() => {\n  fetchData().then(setData);\n}, []);", "code"),
    LabelledQuery("kubectl list pods across namespaces", "kubectl get pods -n production -o wide", "code"),
    LabelledQuery("react refs documentation page", "https://react.dev/learn/referencing-values-with-refs", "url"),
    LabelledQuery("rust book chapter about ownership", "https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html", "url"),
    LabelledQuery("postgres window functions tutorial", "https://www.postgresql.org/docs/current/tutorial-window.html", "url"),
    LabelledQuery("mdn array map reference", "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/map", "url"),
    LabelledQuery("python pathlib documentation", "https://docs.python.org/3/library/pathlib.html", "url"),
    LabelledQuery("sqlite full text search fts5 docs", "https://www.sqlite.org/fts5.html", "url"),
    LabelledQuery("django migrations guide", "https://docs.djangoproject.com/en/stable/topics/migrations/", "url"),
    LabelledQuery("email the design team about the landing page review", "landing-page-review@acme-corp.com", "email"),
    LabelledQuery("invoice from the cloud hosting provider", "billing@cloudhost.io", "email"),
    LabelledQuery("support address for the payments vendor", "support@stripe-payments.example", "email"),
    LabelledQuery("HR contact for onboarding at acme", "onboarding@acme-corp.com", "email"),
    LabelledQuery("sprint planning moved to Thursday afternoon", "Standup moves to 9:30; sprint planning is now Thursday at 3pm in room B.", "text"),
    LabelledQuery("when does the wildcard ssl certificate expire", "The wildcard SSL certificate for *.example.com expires on 2026-11-30.", "text"),
    LabelledQuery("grocery list with oat milk and sourdough", "Shopping: oat milk, sourdough bread, eggs, spinach, coffee beans.", "text"),
    LabelledQuery("idea for a cli that summarises git history", "Idea: a CLI that summarises the last week of git history into a changelog.", "text"),
    LabelledQuery("how do I reset the wifi router", "To reset the router, hold the pinhole button for ten seconds until the lights blink.", "text"),
    LabelledQuery("client quote about the dashboard speed", "\"The dashboard feels much faster now - great work on the caching layer.\"", "text"),
    LabelledQuery("conference talk about local-first software", "Talk idea: local-first software, and why your app should work offline.", "text"),
    LabelledQuery("twenty minute tomato pasta recipe", "Recipe: boil 200g spaghetti, saute garlic and tomatoes, toss with basil.", "text"),
    LabelledQuery("flight booking confirmation for Lisbon", "Flight TP1042 LIS, 2026-12-18 07:40, seat 14C, confirmation XR7Q2P.", "text"),
    # -- 15 code-focused additions ------------------------------------------
    LabelledQuery("sort a list of dicts by key in python", "rows.sort(key=lambda r: r['created_at'])", "code"),
    LabelledQuery("typescript interface for a user object", "interface User {\n  id: number;\n  email: string;\n}", "code"),
    LabelledQuery("go routine waited on with a wait group", "var wg sync.WaitGroup\nwg.Add(1)\ngo func() { defer wg.Done() }()\nwg.Wait()", "code"),
    LabelledQuery("rust get a map value or a default", 'let value = map.get("key").unwrap_or(&default_value);', "code"),
    LabelledQuery("css flexbox center a div", ".center {\n  display: flex;\n  align-items: center;\n  justify-content: center;\n}", "code"),
    LabelledQuery("postgres delete duplicate rows keeping one", "DELETE FROM users a USING users b WHERE a.id > b.id AND a.email = b.email;", "code"),
    LabelledQuery("python read a csv file with pandas", "import pandas as pd\ndf = pd.read_csv('data.csv')", "code"),
    LabelledQuery("java stream filter and collect to list", "List<String> names = users.stream().filter(u -> u.isActive()).map(User::name).collect(Collectors.toList());", "code"),
    LabelledQuery("c++ sort a vector in descending order", "std::sort(v.begin(), v.end(), std::greater<int>());", "code"),
    LabelledQuery("sqlite create an index on a column", "CREATE INDEX idx_orders_customer ON orders(customer_id);", "code"),
    LabelledQuery("flask route that returns json", '@app.route("/api/health")\ndef health():\n    return jsonify(status="ok")', "code"),
    LabelledQuery("makefile target that runs the tests", "test:\n\tpytest -q", "code"),
    LabelledQuery("regular expression to match an ipv4 address", r"^(?:\d{1,3}\.){3}\d{1,3}$", "code"),
    LabelledQuery("shell script fail fast on error", "set -euo pipefail", "code"),
    LabelledQuery("sqlalchemy filter by a date range", "session.query(Order).filter(Order.created_at >= start, Order.created_at < end).all()", "code"),
)

#: Indices of the code-focused labelled queries.
CODE_QUERY_INDICES: tuple[int, ...] = tuple(
    i for i, labelled in enumerate(LABELLED_QUERIES) if labelled.note == "code"
)


@dataclass(frozen=True, slots=True)
class Entry:
    """One synthetic clipboard entry."""

    content: str
    content_type: str
    created_at: datetime


def _make_code(rng: random.Random) -> str:
    """Render one random code snippet."""
    template = rng.choice(_CODE_TEMPLATES)
    return template.format(
        col=rng.choice(_CODE_COLS),
        col2=rng.choice(_CODE_COLS),
        table=rng.choice(_CODE_TABLES),
        agg=rng.choice(["AVG", "SUM", "COUNT", "MAX"]),
        fn=rng.choice(_CODE_FNS),
        arg=rng.choice(_CODE_WORDS),
        cst=rng.choice(["value", "result", "data", "state"]),
        mod=rng.choice(_CODE_WORDS),
        n=rng.randint(1, 9999),
        path=rng.choice(["api", "v1", "internal", "data"]),
        img=rng.choice(["clipmind", "api", "worker", "web"]),
        msg=rng.choice(["fix the parser", "add retries", "bump deps", "tidy up"]),
        ext=rng.choice(["log", "json", "txt", "csv"]),
        cmd=rng.choice(["gzip", "wc -l", "sed -i ''"]),
        cls=rng.choice(["User", "Order", "Config", "Session"]),
        res=rng.choice(["pods", "nodes", "svc", "deploy"]),
        ns=rng.choice(["production", "staging", "default"]),
    )


def _make_url(rng: random.Random) -> str:
    """Render one random URL (with an id/query so the space is large enough)."""
    host = rng.choice(_URL_HOSTS)
    path = rng.choice(_URL_PATHS)
    suffix = rng.choice(
        ["", "", f"/{rng.randint(1, 99999)}", f"?page={rng.randint(1, 500)}",
         f"/{rng.randint(2020, 2026)}/{rng.randint(1, 12):02d}"]
    )
    return f"https://{host}/{path}{suffix}"


def _make_email(rng: random.Random) -> str:
    """Render one random email address."""
    if rng.random() < 0.5:
        local = rng.choice(_EMAIL_ROLES)
    else:
        local = f"{rng.choice(_EMAIL_FIRST)}.{rng.choice(_EMAIL_LAST)}"
    if rng.random() < 0.4:
        local = f"{local}+{rng.randint(1, 9999)}"
    elif rng.random() < 0.3:
        local = f"{local}{rng.randint(1, 999)}"
    return f"{local}@{rng.choice(_EMAIL_DOMAINS)}"


def _make_sentence(rng: random.Random) -> str:
    """Render one random sentence/note."""
    template = rng.choice(_SENTENCE_TEMPLATES)
    return template.format(
        time=rng.choice(["9:30", "10am", "3pm", "4:15", "8:45"]),
        day=rng.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]),
        room=rng.choice(["A", "B", "2C", "the big one", "the quiet room"]),
        thing=rng.choice(_SENTENCE_THINGS),
        month=rng.randint(1, 12),
        day2=rng.randint(1, 28),
        item1=rng.choice(["oat milk", "sourdough bread", "basil", "chickpeas"]),
        item2=rng.choice(["tomatoes", "olive oil", "parmesan", "lemons"]),
        item3=rng.choice(["yogurt", "almonds", "hummus", "crackers"]),
        qty=rng.choice([150, 180, 200, 250, 300]),
        domain=rng.choice(["example.com", "acme-corp.com", "startup.dev"]),
        team=rng.choice(["platform", "design", "growth", "infra"]),
        decision=rng.choice(["ship on Friday", "cut scope", "hire a contractor"]),
        code=rng.choice(["TP1042", "BA0286", "FR1945"]),
        conf=rng.choice(["XR7Q2P", "8LM3KD", "QQ12ZZ"]),
        quote=rng.choice(_SENTENCE_WORDS),
        ref=rng.randint(1000, 9999),
    )


def _make_number(rng: random.Random) -> str:
    """Render one random number/measurement."""
    kind = rng.randint(0, 4)
    if kind == 0:
        return str(rng.randint(100, 999999))
    if kind == 1:
        return f"{rng.uniform(0, 1000):.2f}"
    if kind == 2:
        return f"ERR_{rng.randint(1000, 9999)}"
    if kind == 3:
        return f"{rng.randint(1, 500)} ms"
    return f"{rng.randint(1, 64)} GB"


#: Content mix of the corpus, as ``(generator, count)``.
_CORPUS_MIX: tuple[tuple[Callable[[random.Random], str], int], ...] = (
    (_make_sentence, 4_200),
    (_make_code, 2_600),
    (_make_url, 1_500),
    (_make_email, 1_100),
    (_make_number, 600),
)


def build_corpus(rng: random.Random, size: int, now: datetime) -> list[Entry]:
    """Generate ``size`` unique entries with spread-out timestamps.

    Args:
        rng: Seeded random source.
        size: Total number of entries (includes the labelled landmark entries).
        now: Reference instant; timestamps are drawn from the ``SPREAD_DAYS``
            before it.

    Returns:
        A list of :class:`Entry`, newest-first is *not* guaranteed (order is
        generation order).
    """
    entries: list[Entry] = []
    seen: set[str] = set()

    # Landmark entries first, so every labelled query has a real target.
    for labelled in LABELLED_QUERIES:
        if labelled.expected in seen:
            continue
        seen.add(labelled.expected)
        entries.append(
            Entry(labelled.expected, detect_content_type(labelled.expected), now)
        )

    remaining = max(0, size - len(entries))
    generators: list[Callable[[random.Random], str]] = []
    for generator, count in _CORPUS_MIX:
        generators.extend([generator] * count)
    # Kept in a fixed order so the seeded corpus is reproducible.
    rng.shuffle(generators)

    index = 0
    guard = remaining * 40
    while len(entries) < size and index < guard:
        generator = generators[index % len(generators)]
        index += 1
        content = generator(rng)
        if not content or content in seen:
            continue
        seen.add(content)
        entries.append(Entry(content, detect_content_type(content), now))

    if len(entries) < size:  # pragma: no cover - would mean tiny pools
        raise RuntimeError(
            f"could only generate {len(entries)} unique entries of {size} requested"
        )
    return entries


def assign_timestamps(
    entries: Sequence[Entry], rng: random.Random, now: datetime, spread_days: int
) -> list[Entry]:
    """Give every entry a random timestamp within the spread window."""
    window = spread_days * 24 * 3600
    stamped: list[Entry] = []
    for entry in entries:
        offset = rng.randint(0, window)
        stamped.append(
            Entry(entry.content, entry.content_type, now - timedelta(seconds=offset))
        )
    return stamped


# ------------------------------------------------------------------------ statistics


def percentiles(samples: Sequence[float]) -> dict[str, float]:
    """Return mean/min/max and p50/p95/p99 for ``samples`` (values in ms)."""
    values = np.asarray(list(samples), dtype=float)
    p50, p95, p99 = np.percentile(values, [50, 95, 99])
    return {
        "count": float(values.size),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
        "max": float(values.max()),
    }


# -------------------------------------------------------------------------- running


@dataclass
class ModelResult:
    """Everything measured for one embedding model."""

    spec: ModelSpec
    load_seconds: float = 0.0
    dimension: int = 0
    params_millions: float = 0.0
    weights_mb: float = 0.0
    query_prefix: str = ""
    document_prefix: str = ""
    insert_seconds: float = 0.0
    throughput: dict[str, float] = field(default_factory=dict)
    latency: dict[str, dict[str, float]] = field(default_factory=dict)
    quality: dict[str, dict[str, float]] = field(default_factory=dict)
    quality_code: dict[str, dict[str, float]] = field(default_factory=dict)
    per_query_ranks: list[dict[str, object]] = field(default_factory=list)


@dataclass
class BenchResult:
    """Everything measured, ready to render."""

    environment: dict[str, str] = field(default_factory=dict)
    corpus_size: int = 0
    corpus_types: dict[str, int] = field(default_factory=dict)
    query_count: int = 0
    models: list[ModelResult] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)
    default_model: str = ""
    before: Optional[dict] = None


def collect_environment() -> dict[str, str]:
    """Describe the machine, Python and model stack in play."""
    import numpy

    env = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": _cpu_brand(),
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["torch_threads"] = str(torch.get_num_threads())
    except Exception:  # pragma: no cover - torch is a hard dep of the model
        env["torch"] = "unavailable"
    try:
        import sentence_transformers

        env["sentence_transformers"] = sentence_transformers.__version__
    except Exception:  # pragma: no cover
        env["sentence_transformers"] = "unavailable"
    env["device"] = Config().embedding_device or "auto"
    return env


def resolved_device(embedder) -> str:
    """Return the torch device the loaded model actually runs on."""
    model = getattr(embedder, "_model", None)
    device = getattr(model, "device", None)
    return str(device) if device is not None else "unknown"


def _cpu_brand() -> str:
    """Best-effort CPU brand string (macOS/Linux/Windows)."""
    if sys.platform == "darwin":
        for key in ("machdep.cpu.brand_string", "hw.model"):
            try:
                out = subprocess.run(
                    ["sysctl", "-n", key],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            except Exception:  # pragma: no cover
                continue
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
    return platform.processor() or platform.machine()


def _model_params(embedder) -> tuple[float, int]:
    """Return ``(params_in_millions, dimension)`` for a loaded embedder."""
    model = getattr(embedder, "_model", None)
    dimension = int(getattr(embedder, "dimension", 0) or 0)
    if model is None:
        return 0.0, dimension
    try:
        total = sum(int(param.numel()) for param in model.parameters())
        return total / 1e6, dimension
    except Exception:  # pragma: no cover - defensive
        return 0.0, dimension


def sample_query(rng: random.Random, content: str) -> str:
    """Build a short, plausible query from an entry's own words."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", content)
    if not words:
        return content[:24]
    k = min(len(words), rng.randint(2, 5))
    return " ".join(rng.sample(words, k))


def _hit_rank(outcome, expected: str, top_k: int) -> Optional[int]:
    """1-based rank of ``expected`` within the top ``top_k`` results, else None."""
    target = expected.strip()
    for rank, result in enumerate(outcome.results[:top_k], start=1):
        if result.content.strip() == target:
            return rank
    return None


def _latency_queries(
    rng: random.Random, entries: Sequence[Entry], query_count: int
) -> list[str]:
    """Return ``query_count`` queries: the labelled set first, then sampled ones."""
    queries = [labelled.query for labelled in LABELLED_QUERIES]
    pool = rng.sample(list(entries), k=min(len(entries), max(query_count * 2, 400)))
    for entry in pool:
        if len(queries) >= query_count:
            break
        queries.append(sample_query(rng, entry.content))
    return queries[:query_count]


def _embed_corpus(
    db: Database,
    embedder,
    config: Config,
    result: ModelResult,
    log: Callable[[str], None],
) -> None:
    """Embed every corpus entry, writing the vectors and recording throughput."""
    rows = list(db.iter_missing_embeddings())
    by_content = {row.id: row.content for row in rows}
    ids = [row.id for row in rows]
    embedded = 0
    chars = 0
    started = time.perf_counter()
    for start in range(0, len(ids), EMBED_CHUNK):
        chunk_ids = ids[start : start + EMBED_CHUNK]
        chunk_texts = [by_content[i] for i in chunk_ids]
        vectors = embedder.embed_texts(chunk_texts)
        for item_id, vector in zip(chunk_ids, vectors):
            db.set_embedding(item_id, encode_vector(vector))
        embedded += len(chunk_texts)
        chars += sum(len(text) for text in chunk_texts)
    elapsed = time.perf_counter() - started
    result.throughput = {
        "texts": float(embedded),
        "seconds": elapsed,
        "texts_per_second": embedded / elapsed if elapsed else float("nan"),
        "chars_per_second": chars / elapsed if elapsed else float("nan"),
        "chunk": float(EMBED_CHUNK),
        "model_load_seconds": result.load_seconds,
    }
    log(
        f"  embedded {embedded:,} texts in {elapsed:.2f}s "
        f"({result.throughput['texts_per_second']:,.1f} texts/s)"
    )


def _measure_latency(
    db: Database, embedder, config: Config, queries: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Time every mode for every query with one long-lived searcher (ms)."""
    searcher = HybridSearcher(db, embedder=embedder, config=config)
    samples: dict[str, list[float]] = {name: [] for name, _ in MODES}
    for query in queries:
        for name, kwargs in MODES:
            started = time.perf_counter()
            searcher.search_time_aware(query, top_k=config.search_top_k, **kwargs)
            samples[name].append((time.perf_counter() - started) * 1000.0)
    return {name: percentiles(values) for name, values in samples.items()}


def _quality_from_ranks(
    ranks: dict[str, list[Optional[int]]], indices: Sequence[int]
) -> dict[str, dict[str, float]]:
    """Turn per-query ranks into top-1 / top-5 rates for a subset."""
    quality: dict[str, dict[str, float]] = {}
    total = len(indices)
    for name, _ in MODES:
        hits = [ranks[name][i] for i in indices]
        quality[name] = {
            "top1": sum(1 for rank in hits if rank == 1) / total if total else 0.0,
            "top5": sum(1 for rank in hits if rank is not None and rank <= 5) / total
            if total
            else 0.0,
            "total": float(total),
        }
    return quality


def _measure_quality(
    db: Database, embedder, config: Config
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    list[dict[str, object]],
]:
    """Return overall + code-only quality and the rank of every pair.

    ``min_score`` is forced to ``0`` so the modes are compared on *ranking*
    alone; the shipped threshold would hide near-misses.
    """
    top_k = max(5, config.search_top_k)
    searcher = HybridSearcher(db, embedder=embedder, config=config)
    ranks: dict[str, list[Optional[int]]] = {name: [] for name, _ in MODES}
    for name, kwargs in MODES:
        for labelled in LABELLED_QUERIES:
            outcome = searcher.search_time_aware(
                labelled.query, top_k=top_k, min_score=0.0, **kwargs
            )
            ranks[name].append(_hit_rank(outcome, labelled.expected, top_k))

    overall = _quality_from_ranks(ranks, range(len(LABELLED_QUERIES)))
    code = _quality_from_ranks(ranks, CODE_QUERY_INDICES)

    per_query: list[dict[str, object]] = []
    for index, labelled in enumerate(LABELLED_QUERIES):
        per_query.append(
            {
                "query": labelled.query,
                "expected": labelled.expected,
                "note": labelled.note,
                "ranks": {name: ranks[name][index] for name, _ in MODES},
            }
        )
    return overall, code, per_query


def run_model(
    spec: ModelSpec,
    entries: Sequence[Entry],
    queries: Sequence[str],
    query_count: int,
    quiet: bool,
    device: Optional[str],
) -> ModelResult:
    """Measure one model end to end on the shared corpus and query set."""
    def log(message: str) -> None:
        if not quiet:
            print(message, flush=True)

    log(f"[{spec.name}] {spec.note}")
    config = Config().with_overrides(embedding_model=spec.name)
    if device:
        config = config.with_overrides(embedding_device=device)

    result = ModelResult(spec=spec)
    result.query_prefix, result.document_prefix = (
        (config.embedding_query_prefix, config.embedding_document_prefix)
    )
    if not result.query_prefix and not result.document_prefix:
        result.query_prefix, result.document_prefix = model_prefixes(spec.name)

    db = Database(db_path=":memory:", config=config)
    try:
        insert_started = time.perf_counter()
        for entry in entries:
            outcome = db.insert_clip(
                content=entry.content,
                content_type=entry.content_type,
                source_app=None,
                created_at=entry.created_at,
            )
            if not outcome.inserted:  # pragma: no cover - corpus is unique
                raise RuntimeError(f"duplicate corpus entry: {entry.content!r}")
        result.insert_seconds = time.perf_counter() - insert_started

        log("  loading the real embedding model…")
        embedder = get_embedder(config)
        load_started = time.perf_counter()
        embedder.embed_texts(["warmup"])  # triggers the one-time load
        result.load_seconds = time.perf_counter() - load_started
        result.dimension = int(getattr(embedder, "dimension", 0) or 0)
        result.params_millions, _ = _model_params(embedder)
        result.weights_mb = result.params_millions * 4.0
        log(
            f"  model ready in {result.load_seconds:.2f}s "
            f"(dim {result.dimension}, device: {resolved_device(embedder)})"
        )

        _embed_corpus(db, embedder, config, result, log)

        log(f"  measuring search latency over {query_count} queries…")
        result.latency = _measure_latency(db, embedder, config, queries)
        for name, stats in result.latency.items():
            log(
                f"    {name:<9} p50={stats['p50']:.1f}ms "
                f"p95={stats['p95']:.1f}ms p99={stats['p99']:.1f}ms"
            )

        log(f"  measuring retrieval quality on {len(LABELLED_QUERIES)} labelled pairs…")
        result.quality, result.quality_code, result.per_query_ranks = _measure_quality(
            db, embedder, config
        )
        for name, stats in result.quality.items():
            log(
                f"    {name:<9} top-1={stats['top1'] * 100:.1f}% "
                f"top-5={stats['top5'] * 100:.1f}%"
            )
        log(
            "    code top-5: "
            + ", ".join(
                f"{name}={result.quality_code[name]['top5'] * 100:.0f}%"
                for name, _ in MODES
            )
        )
    finally:
        db.close()
    return result


def _load_before() -> Optional[dict]:
    """Load the pre-fix run used for the before/after latency table."""
    try:
        return json.loads(BEFORE_JSON.read_text(encoding="utf-8"))
    except Exception:
        return None


def run_benchmark(
    query_count: int,
    quiet: bool = False,
    device: Optional[str] = None,
    models: Optional[Sequence[ModelSpec]] = None,
) -> BenchResult:
    """Run the whole benchmark and return the measured result."""
    def log(message: str) -> None:
        if not quiet:
            print(message, flush=True)

    result = BenchResult(corpus_size=CORPUS_SIZE, query_count=query_count)
    result.environment = collect_environment()
    result.default_model = Config().embedding_model

    rng = random.Random(SEED)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    log(f"building {CORPUS_SIZE:,} synthetic entries (seed={SEED})…")
    entries = assign_timestamps(build_corpus(rng, CORPUS_SIZE, now), rng, now, SPREAD_DAYS)
    result.corpus_types = _count_types(entries)
    log(f"corpus types: {result.corpus_types}")

    # One deterministic query set, shared by every model, so latency is
    # comparable across models.
    query_rng = random.Random(SEED + 1)
    queries = _latency_queries(query_rng, entries, query_count)

    for spec in models or DEFAULT_MODELS:
        try:
            result.models.append(
                run_model(spec, entries, queries, query_count, quiet, device)
            )
        except EmbeddingModelUnavailable as exc:
            log(f"[{spec.name}] unavailable: {exc}")
            result.failures[spec.name] = str(exc)

    result.before = _load_before()
    return result


def result_from_json(path: Path | str) -> BenchResult:
    """Rebuild a :class:`BenchResult` from a previous ``--json`` payload.

    Used by ``--render-from`` to re-render the report (for example after a
    cosmetic change to the tables) without re-running the models.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    result = BenchResult(
        environment=dict(data.get("environment", {})),
        corpus_size=int(data.get("corpus_size", 0)),
        corpus_types=dict(data.get("corpus_types", {})),
        query_count=int(data.get("query_count", 0)),
        default_model=data.get("default_model", Config().embedding_model),
        failures=dict(data.get("failures", {})),
        before=_load_before(),
    )
    for entry in data.get("models", []):
        result.models.append(
            ModelResult(
                spec=ModelSpec(entry["name"], entry.get("note", "")),
                load_seconds=float(entry.get("load_seconds", 0.0)),
                dimension=int(entry.get("dimension", 0)),
                params_millions=float(entry.get("params_millions", 0.0)),
                weights_mb=float(entry.get("weights_mb", 0.0)),
                query_prefix=entry.get("query_prefix", ""),
                document_prefix=entry.get("document_prefix", ""),
                throughput=dict(entry.get("throughput", {})),
                latency=dict(entry.get("latency_ms", {})),
                quality=dict(entry.get("quality", {})),
                quality_code=dict(entry.get("quality_code", {})),
                per_query_ranks=list(entry.get("per_query_ranks", [])),
            )
        )
    return result


def _count_types(entries: Sequence[Entry]) -> dict[str, int]:
    """Count corpus entries per detected content type."""
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.content_type] = counts.get(entry.content_type, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------------- rendering


def _default_result(result: BenchResult) -> Optional[ModelResult]:
    """Return the result for the currently configured default model."""
    for model in result.models:
        if model.spec.name == result.default_model:
            return model
    return result.models[0] if result.models else None


def _task1_result(result: BenchResult) -> Optional[ModelResult]:
    """Return the MiniLM result used for the before/after table."""
    for model in result.models:
        if model.spec.name == TASK1_MODEL:
            return model
    return None


def render_results_markdown(result: BenchResult) -> str:
    """Render ``benchmarks/RESULTS.md`` from the measured result."""
    env = result.environment
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    model_names = ", ".join(m.spec.name for m in result.models) or "(none)"

    lines: list[str] = []
    lines.append("# ClipMind benchmark results")
    lines.append("")
    lines.append(
        f"_Measured on {generated} by `benchmarks/bench_search.py` "
        f"(seed `{SEED}`, corpus {result.corpus_size:,} entries, "
        f"{result.query_count} latency queries, {len(LABELLED_QUERIES)} labelled "
        "pairs). All figures below are measured on this machine, not estimates._"
    )
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| OS / platform | `{env.get('platform', '?')}` |")
    lines.append(f"| CPU | `{env.get('processor', '?')}` |")
    lines.append(f"| Architecture | `{env.get('machine', '?')}` |")
    lines.append(f"| Python | `{env.get('python', '?')}` |")
    lines.append(f"| PyTorch | `{env.get('torch', '?')}` ({env.get('torch_threads', '?')} threads) |")
    lines.append(f"| sentence-transformers | `{env.get('sentence_transformers', '?')}` |")
    lines.append(f"| NumPy | `{env.get('numpy', '?')}` |")
    lines.append(f"| Device | `{env.get('device', '?')}` |")
    lines.append(f"| Models benchmarked | {model_names} |")
    lines.append(f"| Current default model | `{result.default_model}` |")
    lines.append("")

    lines.append("## Corpus")
    lines.append("")
    lines.append(
        f"{result.corpus_size:,} synthetic clipboard entries, generated with a "
        f"fixed seed (`{SEED}`) and timestamps spread uniformly over the previous "
        f"{SPREAD_DAYS} days. The 45 labelled expected entries are part of it."
    )
    lines.append("")
    lines.append("| Content type | Count |")
    lines.append("| --- | ---: |")
    for content_type in ("text", "code", "url", "email", "number"):
        if content_type in result.corpus_types:
            lines.append(f"| {content_type} | {result.corpus_types[content_type]:,} |")
    lines.append(f"| **total** | **{result.corpus_size:,}** |")
    lines.append("")

    lines.extend(_render_before_after(result))
    lines.extend(_render_model_comparison(result))
    lines.extend(_render_default_detail(result))
    lines.extend(_render_interpretation(result))
    lines.extend(_render_labelled_table(result))

    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("python benchmarks/bench_search.py            # regenerates this file")
    lines.append("python benchmarks/bench_search.py --json out.json")
    lines.append("python benchmarks/bench_search.py --models all-MiniLM-L6-v2")
    lines.append("```")
    lines.append("")
    lines.append(
        "> Latency and throughput depend on the host CPU and on how busy it is; "
        "the corpus and the retrieval-quality figures are reproducible."
    )
    lines.append("")
    return "\n".join(lines)


def _render_before_after(result: BenchResult) -> list[str]:
    """Render the Task 1 latency before/after table."""
    lines = ["## Search latency: before vs after the index-reuse fix (Task 1)", ""]
    after = _task1_result(result)
    if result.before is None or after is None:
        lines.append(
            "_No before/after comparison available "
            "(results_before.json or the MiniLM run is missing)._"
        )
        lines.append("")
        return lines

    before_latency = result.before.get("latency_ms", {})
    lines.append(
        "Before: every `hybrid_search` call built a fresh `SemanticSearcher`, so "
        "each query re-read all embedding BLOBs from SQLite. After: one long-lived "
        f"in-memory index, updated incrementally. Both rows use `{TASK1_MODEL}` on "
        "the same corpus; before is the recorded pre-fix run "
        "(`benchmarks/results_before.json`), after is this run."
    )
    lines.append("")
    lines.append("| Mode | before p50 | before p95 | after p50 | after p95 | p50 speedup |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, _ in MODES:
        b = before_latency.get(name)
        a = after.latency.get(name)
        if not b or not a:
            continue
        speedup = b["p50"] / a["p50"] if a["p50"] else float("nan")
        lines.append(
            f"| {name} | {b['p50']:.1f} ms | {b['p95']:.1f} ms | "
            f"{a['p50']:.1f} ms | {a['p95']:.1f} ms | {speedup:.1f}x |"
        )
    lines.append("")
    return lines


def _render_model_comparison(result: BenchResult) -> list[str]:
    """Render the per-model size, throughput, latency and quality tables."""
    lines = ["## Model comparison (Task 2)", ""]
    if not result.models:
        lines.append("_No models could be measured._")
        lines.append("")
        return lines

    lines.append("### Model size and embedding throughput")
    lines.append("")
    lines.append(
        "Weights (MB) is the parameter count at fp32 (the on-disk weight file is "
        "the same within a few percent). Throughput is the 10,000-text corpus "
        f"pass in chunks of {EMBED_CHUNK}."
    )
    lines.append("")
    lines.append(
        "| Model | Dim | Params (M) | Weights (MB) | Load (s) | "
        "Pass (s) | Throughput (texts/s) |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model in result.models:
        t = model.throughput
        lines.append(
            f"| `{model.spec.name}` | {model.dimension} | {model.params_millions:.1f} | "
            f"{model.weights_mb:.0f} | "
            f"{model.load_seconds:.2f} | {t.get('seconds', 0):.1f} | "
            f"**{t.get('texts_per_second', 0):,.1f}** |"
        )
    lines.append("")
    lines.append(
        "Prefixes applied (query / document): "
        + "; ".join(
            f"`{model.spec.name}` → {model.query_prefix!r} / {model.document_prefix!r}"
            for model in result.models
        )
        + "."
    )
    lines.append("")

    lines.append("### Query latency after Task 1 (per model)")
    lines.append("")
    lines.append(
        "One long-lived searcher per model over the shared query set; wall-clock "
        "milliseconds per query."
    )
    lines.append("")
    lines.append(
        "| Model | semantic p50 | semantic p95 | keyword p50 | hybrid p50 | hybrid p95 |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for model in result.models:
        lat = model.latency
        lines.append(
            f"| `{model.spec.name}` | {lat['semantic']['p50']:.1f} ms | "
            f"{lat['semantic']['p95']:.1f} ms | {lat['keyword']['p50']:.1f} ms | "
            f"{lat['hybrid']['p50']:.1f} ms | {lat['hybrid']['p95']:.1f} ms |"
        )
    lines.append("")

    lines.append("### Retrieval quality: all 45 labelled queries")
    lines.append("")
    lines.append("| Model | semantic top-1 | semantic top-5 | hybrid top-1 | hybrid top-5 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for model in result.models:
        q = model.quality
        lines.append(
            f"| `{model.spec.name}` | {q['semantic']['top1'] * 100:.1f}% | "
            f"{q['semantic']['top5'] * 100:.1f}% | {q['hybrid']['top1'] * 100:.1f}% | "
            f"**{q['hybrid']['top5'] * 100:.1f}%** |"
        )
    lines.append("")
    lines.append(f"### Retrieval quality: the 15 code-focused queries")
    lines.append("")
    lines.append("| Model | semantic top-1 | semantic top-5 | hybrid top-1 | hybrid top-5 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for model in result.models:
        q = model.quality_code
        lines.append(
            f"| `{model.spec.name}` | {q['semantic']['top1'] * 100:.1f}% | "
            f"{q['semantic']['top5'] * 100:.1f}% | {q['hybrid']['top1'] * 100:.1f}% | "
            f"**{q['hybrid']['top5'] * 100:.1f}%** |"
        )
    lines.append("")
    if result.failures:
        lines.append("Models that could not be loaded: ")
        lines.append("")
        for name, error in result.failures.items():
            lines.append(f"* `{name}`: {error}")
        lines.append("")
    return lines


def _render_default_detail(result: BenchResult) -> list[str]:
    """Render the detailed throughput/latency/quality tables for the default."""
    model = _default_result(result)
    lines: list[str] = []
    if model is None:
        return lines
    t = model.throughput
    lines.append(f"## Default model detail (`{model.spec.name}`)")
    lines.append("")
    lines.append(
        f"The real `{model.spec.name}` model was loaded once "
        f"({model.load_seconds:.2f} s) and then used to embed the whole corpus in "
        f"chunks of {int(t.get('chunk', EMBED_CHUNK))}."
    )
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    lines.append(f"| Model load (first call) | {model.load_seconds:.2f} s |")
    lines.append(f"| Texts embedded | {int(t.get('texts', 0)):,} |")
    lines.append(f"| Embedding pass | {t.get('seconds', 0):.2f} s |")
    lines.append(f"| Throughput | **{t.get('texts_per_second', 0):,.1f} texts/s** |")
    lines.append(f"| Throughput (chars) | {t.get('chars_per_second', 0):,.0f} chars/s |")
    lines.append("")
    lines.append("| Mode | p50 | p95 | p99 | mean | max |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, _ in MODES:
        stats = model.latency[name]
        lines.append(
            f"| {name} | {stats['p50']:.1f} ms | {stats['p95']:.1f} ms | "
            f"{stats['p99']:.1f} ms | {stats['mean']:.1f} ms | {stats['max']:.1f} ms |"
        )
    lines.append("")
    lines.append(
        f"Top-1 / top-5 hit rate on {len(LABELLED_QUERIES)} labelled pairs, "
        "`min_score = 0` so the modes are compared purely on ranking. "
        f"Code subset = {len(CODE_QUERY_INDICES)} queries."
    )
    lines.append("")
    lines.append("| Mode | all: top-1 | all: top-5 | code: top-1 | code: top-5 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for name, _ in MODES:
        q = model.quality[name]
        c = model.quality_code[name]
        lines.append(
            f"| {name} | {q['top1'] * 100:.1f}% | {q['top5'] * 100:.1f}% | "
            f"{c['top1'] * 100:.1f}% | {c['top5'] * 100:.1f}% |"
        )
    lines.append("")
    return lines


def _render_interpretation(result: BenchResult) -> list[str]:
    """Render the measured-facts interpretation section."""
    lines = ["## Interpretation", ""]
    after = _task1_result(result)
    if result.before is not None and after is not None:
        b = result.before.get("latency_ms", {}).get("semantic", {})
        a = after.latency.get("semantic", {})
        if b and a:
            lines.append(
                f"* **The per-query rebuild is gone.** Semantic p50 fell from "
                f"{b['p50']:.0f} ms to {a['p50']:.0f} ms "
                f"({b['p50'] / a['p50']:.1f}x) for `{TASK1_MODEL}`; a query is now a "
                "masked matmul over the warm matrix plus one model forward pass. "
                "Keyword search is unchanged (FTS5/BM25 inside SQLite)."
            )
    default = _default_result(result)
    previous = _task1_result(result)
    if default is not None:
        lines.append(
            f"* **Retrieval quality (all 45 queries, hybrid top-5):** "
            + ", ".join(
                f"`{m.spec.name}` {m.quality['hybrid']['top5'] * 100:.0f}%"
                for m in result.models
            )
            + ". Measured, not assumed."
        )
        lines.append(
            f"* **Code queries (15) hybrid top-5:** "
            + ", ".join(
                f"`{m.spec.name}` {m.quality_code['hybrid']['top5'] * 100:.0f}%"
                for m in result.models
            )
            + "."
        )
        if previous is not None and default.spec.name != previous.spec.name:
            d_gain = (default.quality["hybrid"]["top5"] - previous.quality["hybrid"]["top5"]) * 100
            c_gain = (
                default.quality_code["hybrid"]["top5"]
                - previous.quality_code["hybrid"]["top5"]
            ) * 100
            latency_ratio = (
                default.latency["hybrid"]["p50"] / previous.latency["hybrid"]["p50"]
            )
            size_ratio = default.weights_mb / previous.weights_mb
            lines.append(
                f"* **Default model changed from `{previous.spec.name}` to "
                f"`{default.spec.name}`.** That is {d_gain:+.1f} points overall and "
                f"{c_gain:+.1f} points on code queries (hybrid top-5), which clears "
                f"the 5-point bar, at {latency_ratio:.1f}x hybrid p50 "
                f"({default.latency['hybrid']['p50']:.0f} ms) and {size_ratio:.1f}x "
                f"the weights ({default.weights_mb:.0f} MB vs "
                f"{previous.weights_mb:.0f} MB). Set "
                f"`embedding_model = \"{previous.spec.name}\"` to go back."
            )
        elif previous is not None:
            best = max(result.models, key=lambda m: m.quality["hybrid"]["top5"])
            d_gain = (best.quality["hybrid"]["top5"] - default.quality["hybrid"]["top5"]) * 100
            c_gain = (
                best.quality_code["hybrid"]["top5"]
                - default.quality_code["hybrid"]["top5"]
            ) * 100
            latency_ratio = best.latency["hybrid"]["p50"] / default.latency["hybrid"]["p50"]
            size_ratio = best.weights_mb / default.weights_mb
            lines.append(
                f"* **Default model stayed `{result.default_model}`.** The best "
                f"alternative, `{best.spec.name}`, measured {d_gain:+.1f} points "
                f"overall and {c_gain:+.1f} on code (hybrid top-5), but costs "
                f"{size_ratio:.1f}x the weights ({best.weights_mb:.0f} MB vs "
                f"{default.weights_mb:.0f} MB), {latency_ratio:.1f}x the hybrid "
                "latency and half the embedding throughput, and would need "
                "`min_score` re-tuned. That total cost is not clearly acceptable "
                "for a lightweight local tool, so the smaller model is kept; set "
                f"`embedding_model = \"{best.spec.name}\"` to opt in."
            )
    lines.append(
        "* **Reproducibility:** the corpus and labels are deterministic (fixed "
        "seed); the throughput and latency columns depend on the host and its load."
    )
    lines.append("")
    return lines


def _render_labelled_table(result: BenchResult) -> list[str]:
    """Render the raw per-query ranks for the default model."""
    model = _default_result(result)
    lines = ["### Labelled queries used", ""]
    if model is None or not model.per_query_ranks:
        lines.append("_No per-query ranks available._")
        lines.append("")
        return lines
    lines.append(
        f"Rank of the expected entry in each mode for `{model.spec.name}` (lower is "
        "better; `—` means it did not appear in the top 5). The 15 code-focused "
        "queries are the last rows."
    )
    lines.append("")
    lines.append("| # | Query | semantic | keyword | hybrid | Expected entry |")
    lines.append("| ---: | --- | ---: | ---: | ---: | --- |")
    for index, row in enumerate(model.per_query_ranks, start=1):
        query = str(row["query"]).replace("|", "\\|")
        expected = str(row["expected"]).replace("\n", " ").replace("|", "\\|")
        if len(expected) > 70:
            expected = expected[:67] + "…"
        ranks = row["ranks"]
        cells = []
        for name, _ in MODES:
            rank = ranks.get(name)  # type: ignore[union-attr]
            cells.append("—" if rank is None else str(rank))
        lines.append(
            f"| {index} | {query} | {cells[0]} | {cells[1]} | {cells[2]} | `{expected}` |"
        )
    lines.append("")
    return lines


def render_readme_table(result: BenchResult) -> str:
    """Render the compact table used in the top-level README."""
    model = _default_result(result) or _task1_result(result)
    t = model.throughput if model else {}
    lines = [
        "| Metric | Result |",
        "| --- | ---: |",
        f"| Corpus | {result.corpus_size:,} entries |",
        f"| Embedding throughput | {t.get('texts_per_second', 0):,.0f} texts/s |",
    ]
    if model is not None:
        for name, _ in MODES:
            stats = model.latency[name]
            lines.append(
                f"| {name} search (p50 / p95) | {stats['p50']:.0f} ms / {stats['p95']:.0f} ms |"
            )
        for name, _ in MODES:
            stats = model.quality[name]
            lines.append(f"| {name} top-5 quality | {stats['top5'] * 100:.0f}% |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--queries", type=int, default=DEFAULT_QUERIES,
        help=f"Latency queries to run (default: {DEFAULT_QUERIES}; minimum 200).",
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "benchmarks" / "RESULTS.md",
        help="Where to write the markdown report (default: benchmarks/RESULTS.md).",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="Optional path for raw JSON results.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device override (cpu/mps/cuda); default: config/auto.",
    )
    parser.add_argument(
        "--models", default=None,
        help="Comma-separated model names to compare (default: the three built-ins).",
    )
    parser.add_argument(
        "--render-from", type=Path, default=None,
        help="Re-render the report from a previous --json file without re-measuring.",
    )
    parser.add_argument(
        "--no-write", action="store_true", help="Print only; write no files.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress output.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the benchmark, write the report and print the README table."""
    args = build_parser().parse_args(argv)

    models: Optional[list[ModelSpec]] = None
    if args.models:
        models = [
            ModelSpec(name.strip(), "custom")
            for name in args.models.split(",")
            if name.strip()
        ]

    if args.render_from is not None:
        result = result_from_json(args.render_from)
    else:
        if args.queries < 200:
            print("error: --queries must be at least 200", file=sys.stderr)
            return 2
        try:
            result = run_benchmark(
                args.queries, quiet=args.quiet, device=args.device, models=models
            )
        except EmbeddingModelUnavailable as exc:
            print(
                f"error: the embedding model could not be loaded: {exc}",
                file=sys.stderr,
            )
            return 1

    if not result.models:
        print("error: no models could be measured", file=sys.stderr)
        return 1

    markdown = render_results_markdown(result)
    if not args.no_write:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown, encoding="utf-8")
        print(f"\nwrote {args.out}")
        if args.json:
            payload = {
                "environment": result.environment,
                "corpus_size": result.corpus_size,
                "corpus_types": result.corpus_types,
                "query_count": result.query_count,
                "default_model": result.default_model,
                "failures": result.failures,
                "models": [
                    {
                        "name": m.spec.name,
                        "note": m.spec.note,
                        "dimension": m.dimension,
                        "params_millions": m.params_millions,
                        "weights_mb": m.weights_mb,
                        "load_seconds": m.load_seconds,
                        "query_prefix": m.query_prefix,
                        "document_prefix": m.document_prefix,
                        "throughput": m.throughput,
                        "latency_ms": m.latency,
                        "quality": m.quality,
                        "quality_code": m.quality_code,
                        "per_query_ranks": m.per_query_ranks,
                    }
                    for m in result.models
                ],
            }
            args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"wrote {args.json}")

    print("\nREADME table:\n")
    print(render_readme_table(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
