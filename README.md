# ClipMind

A **local-first** clipboard manager for macOS/Windows/Linux. It runs quietly in
the background, records everything you copy, and lets you find it again with
natural language — *"that SQL query about orders I copied yesterday"*,
*"the link about React hooks from this morning"*.

Nothing ever leaves your machine. There are no network calls after the one-time
embedding-model download, and no telemetry of any kind.

---

## Status

| Phase | Scope | State |
| ----- | ----- | ----- |
| **1** | Clipboard capture, SQLite storage, dedup, CLI | ✅ done |
| **2** | Local embeddings (`all-MiniLM-L6-v2`), background worker, semantic search via CLI | ✅ done |
| **3** | Hybrid search (semantic + FTS5 BM25) + time-aware query parsing | ✅ done |
| **4** | Privacy filters (secret detection, ignore list, retention) + `config.toml` | ✅ done |
| **5** | PySide6 popup UI, global hotkey, tray icon | ✅ **done (this release)** |
| 6 | Benchmark script + PyInstaller packaging | planned (test suite exists) |

Current version: **0.5.0** (Phase 5).

---

## Problem statement

The system clipboard holds exactly **one** item. The moment you copy something
else, what you had is gone — even if you need it five minutes (or five days)
later. Existing clipboard managers give you a flat, chronological list that only
supports exact-substring search, which is useless when you remember *what a
thing was about* rather than *what it said*.

ClipMind fixes this with three ideas:

1. **Capture everything, cheaply** — poll the clipboard, store text locally in
   SQLite, and never store the same thing twice.
2. **Two kinds of search at once** — meaning-based (embeddings) *and*
   literal (FTS5 keyword/BM25), so `ERR_4021` or a URL still match exactly.
3. **Time as a first-class filter** — "yesterday", "last Monday" and "2 hours
   ago" are parsed out of the query and applied as a date range before the
   remaining words are searched.

Phase 1 delivers the foundation: reliable capture, typed storage, deduplication,
and full-text indexing, all driven from the command line.

---

## Architecture

```
                    ┌─────────────────────────────┐
                    │      system clipboard       │
                    └──────────────┬──────────────┘
                                   │ (poll every 0.5s)
                    ┌──────────────▼──────────────┐
                    │  capture/                   │
                    │  monitor.py  ── polling loop│
                    │  processor.py ─ normalize,  │
                    │                 truncate,   │
                    │                 hash        │
                    │  detect.py   ── type tags   │
                    │  source_app.py ── app probe │
                    └──────────────┬──────────────┘
                                   │ ProcessedClip
                    ┌──────────────▼──────────────┐
                    │  storage/db.py              │
                    │  SQLite (WAL)               │
                    │   ├─ items   (+ embedding)  │
                    │   ├─ items_fts (FTS5 BM25)  │
                    │   └─ triggers keep them in  │
                    │       sync automatically    │
                    └──────────────┬──────────────┘
                                   │
        ┌──────────────────────────┴─────────────────────────┐
        │                                                    │
┌───────▼────────────┐  ┌─────────────────┐  ┌───────────────▼─────────┐
│  search/           │  │  embeddings/    │  │  ui/                    │
│  SemanticSearcher  │  │  queue + worker │  │  view_model (no Qt)     │
│  KeywordSearcher   │◄─┤  MiniLM (384d)  │  │  popup, tray, hotkey    │
│  HybridSearcher    │  │  load once      │  │  app: one process       │
│  TimeQuery parser  │  └────────┬────────┘  └─────────────────────────┘
└────────────────────┘           │
                                 │ writes float32 BLOBs
                    ┌────────────▼────────────────┐
                    │  privacy/                   │
                    │  detectors + policy +       │
                    │  ignore list + retention    │
                    └─────────────────────────────┘
```

Capture data flow:

```
clipboard ──► monitor.poll_once() ──► processor.process() ──► Database.insert_clip()
                   │                        │                    │        │
             in-memory "unchanged?"   normalize → truncate →   dedup by     │
             short-circuit            detect type → SHA-256    content_hash │
                                                                           │
                                              submit(item_id, text, type)  │
                                                       │                   │
                                        ┌──────────────▼───────────┐       │
                                        │ EmbeddingService queue   │       │
                                        │  (never blocks the poll) │       │
                                        └──────────────┬───────────┘       │
                                                       ▼                   ▼
                                        embed_texts(batch) ──► set_embedding(id, blob)
```

Query data flow:

```
clipmind search "react hooks article yesterday"
        │
        ▼
parse_time_query()  ──►  cleaned_query = "react hooks article"
        │                 start/end = yesterday's local window
        ▼
HybridSearcher.search(cleaned_query, since=start, until=end)
        │
        ├─► SemanticSearcher  ── embed query, cosine top-N    (w_sem = 0.7)
        │
        ├─► KeywordSearcher   ── FTS5 MATCH, BM25 top-N        (w_kw  = 0.3)
        │        (query tokens quoted; nothing is parsed as FTS syntax)
        │
        ├─► merge by item id: final = 0.7*sem + 0.3*kw*coverage
        │
        └─► drop below min_score, unless a strong exact keyword hit
```

### Design decisions

* **SQLite + FTS5 in one file.** `items_fts` is an *external-content* FTS5
  table over `items.content`; triggers keep the index in sync on insert, update
  and delete, so there is no second source of truth to drift.
* **Dedup by content hash, not by text comparison.** `content_hash` carries a
  `UNIQUE` index, so duplicates are rejected by the database even if two
  writers race.
* **Hash the *stored* text.** Truncation happens before hashing, so two long
  texts sharing a 10,000-character prefix cannot collide.
* **One connection, one lock.** The Phase 2 embedding worker will write from a
  background thread; `Database` already guards a shared connection with an
  `RLock` and `check_same_thread=False`.
* **Pluggable clipboard reader.** `ClipboardReader` is a one-method protocol,
  so tests and the demo script never touch a real clipboard.
* **No FTS query syntax leaks into the UI.** `Database.search_keywords()` takes
  a raw FTS5 MATCH expression; Phase 3 owns quoting and ranking.
* **Embeddings are float32 BLOBs, not pickles.** 384 dims * 4 bytes = 1536
  bytes per row, readable from any language, no ``.npy`` header to parse.
* **The model is a lazy singleton** keyed by ``(model_name, device)``. Importing
  the package never imports torch; the first ``embed_texts`` call pays for the
  load, and it happens on the worker thread, not the clipboard thread.
* **Search is behind `VectorIndex`.** Swapping in FAISS or Chroma means writing
  a class with ``size`` and ``search``, nothing else changes.
* **User text never reaches the FTS5 parser.** `sanitize_fts_query` reduces a
  query to alphanumeric tokens and quotes each one, so `SELECT "id" FROM ...`,
  URLs, `C++` and unbalanced quotes can only ever produce *no results*, never an
  `fts5: syntax error`.
* **Time phrases are parsed by hand, not with `dateparser`.** `dateparser` is
  eager: it reads `monday.com` as a weekday and a bare `5pm` as a filter. Phase 3
  needs the opposite, and the supported phrase set is small and fixed, so a
  deterministic ~150-line regex parser (`search/timeparse.py`) makes every phrase
  an explicit, testable rule - with no dependency and no locale data.
* **The date range is a pre-filter, not a re-rank.** The parsed `[start, end)` is
  pushed down into both retrievers' SQL, so ranking never even sees rows outside
  the window.
* **Hybrid scores are explained, not mysterious.** Every hit carries
  `semantic_score`, `keyword_score` and `final_score`, surfaced by `--explain`.
* **The index is cached by a `(row_count, max_id)` fingerprint** so the Phase 5
  UI can search on every keystroke without re-decoding 50,000 blobs each time.

---

## Setup

Requires **Python 3.11+**.

```bash
cd clipmind
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first embedding request downloads ``all-MiniLM-L6-v2`` (~90 MB) from
Hugging Face into ``~/.cache/huggingface`` (or ``$HF_HOME``). That is the only
network access ClipMind ever performs; everything afterwards runs offline.

After the weights are cached, loading the model into a fresh process took
**~11 s** on the Intel Mac this was developed on, so each `search`/`reindex`
invocation costs that once (the download is additional and one-time). The
Phase 5 app (`clipmind app`) keeps one process alive and loads the model once on
a background thread at startup, so it only pays this once.
`requirements.txt` pins a version combination that works with PyTorch 2.2.2 (the
last release with macOS x86_64 wheels) - see the comments there if you are on a
newer platform.

Optional, for the `clipmind` command on your PATH:

```bash
pip install -e .
```

---

## Usage (Phase 1: capture)

### Capture

```bash
# Run in the foreground. Ctrl-C to stop.
python -m clipmind watch

# Faster polling, and don't ignore what's already on the clipboard.
python -m clipmind watch --interval 0.2 --capture-initial

# Scripted / demo mode: poll a fixed number of times, then exit.
python -m clipmind watch --iterations 20
```

The monitor prints one line per new capture:

```
19:33:57 INFO clipmind.capture.monitor: captured code entry #2 (69 chars)
```

### Inspect

```bash
python -m clipmind list                    # 20 most recent, newest first
python -m clipmind list -n 50
python -m clipmind list --type links       # links | code | text | email | number | all
python -m clipmind list --json             # one JSON object per line
python -m clipmind stats                   # counts per type + DB location
```

```
   ID  TYPE           AGE  LOCAL TIME      SOURCE APP      PREVIEW
------------------------------------------------------------------
    4  text      just now  10-04 19:33  -               just a normal note about the meeting
    3  email     just now  10-04 19:33  -               andrew@example.com
    2  code      just now  10-04 19:33  -               SELECT o.id, o.total FROM orders o WHERE …
    1  link      just now  10-04 19:33  -               https://react.dev/learn/thinking-in-react
```

### Store text directly (handy for demos and scripts)

```bash
python -m clipmind add "https://react.dev/learn/thinking-in-react"
echo "SELECT id FROM orders;" | python -m clipmind add --stdin
```

`add` goes through the exact same pipeline as the clipboard poller — same
normalisation, truncation, type detection and dedup — so it is a faithful way to
exercise capture without a GUI.

### Point at a different database

```bash
python -m clipmind --db /tmp/demo.db list
```

or set `CLIPMIND_DB` / `CLIPMIND_DATA_DIR`. Default location:
`~/Library/Application Support/clipmind/clipmind.db` (macOS),
`%APPDATA%\clipmind\` (Windows), `$XDG_DATA_HOME/clipmind/` (Linux).

---

## Usage (Phase 2: semantic search)

### Index what you already have

New captures are embedded automatically while `watch` runs. Entries stored
before Phase 2, or captured with `--no-embed`, need a backfill:

```bash
python -m clipmind reindex
```

```
Database: /Users/you/Library/Application Support/clipmind/clipmind.db
  2/7
  7/7
Scanned 12 entr(y/ies): 7 without embeddings (7 eligible, 0 skipped as too short/typed).
Embedded 7, failed 0 in 1.4s.
```

`reindex` only touches rows whose `embedding` column is still `NULL`, so it is
safe - and cheap - to run at any time. Bare numbers and strings shorter than 3
characters are never embedded (they carry no useful meaning; keyword search
will still find them in Phase 3).

### Search

```bash
python -m clipmind search "database query"
python -m clipmind search "react hooks article" --top 3
python -m clipmind search "that long article" --type links
python -m clipmind search "orders" --min-score 0.1     # more recall
python -m clipmind search "orders" --json
```

```
  #   SCORE        AGE  TYPE    PREVIEW
--------------------------------------
  1   0.411     2m ago  code    SELECT id FROM orders;
  2   0.238     1h ago  text    meeting at 5pm with Rahul

2 result(s) from 12 indexed entr(y/ies).
```

Flags:

| Flag | Meaning |
| ---- | ------- |
| `--top N` | Maximum results (default 5). |
| `-t/--type links\|code\|text\|email\|number\|all` | Restrict to one content type. Filtering happens before scoring. |
| `--min-score S` | Minimum cosine similarity (default 0.25). Raise for precision, lower for recall. |
| `--json` | One JSON object per hit: `rank, score, id, content, content_type, created_at, source_app`. |

When nothing clears the threshold you get an explanation instead of junk:

```
No good matches above 0.25 (best was 0.099). Try a lower --min-score or different words.
```

or, if the database has no embeddings at all:

```
No embeddings found in /…/clipmind.db.
Run `clipmind reindex` to index existing entries; new captures are indexed automatically while `clipmind watch` runs.
```

The model is loaded on the first query in a process and reused for every later
query in that process. On the development machine (Intel Mac, CPU/MPS) that load
measured ~11 s; once loaded, embedding a query takes ~0.17 s.

### Capture with automatic embedding

```bash
python -m clipmind watch              # capture + embed in the background
python -m clipmind watch --no-embed   # capture only (Phase 1 behaviour)
```

### Tuning

`min_score` is the main knob: `all-MiniLM-L6-v2` cosine scores are compressed,
so genuinely-related short strings often land around `0.3-0.8`, while unrelated
ones sit near `0.0-0.1`. The default `0.25` deliberately favours precision.
Set it via `--min-score`, or the `CLIPMIND_MIN_SCORE` environment variable.

```bash
CLIPMIND_EMBED_MODEL=all-MiniLM-L6-v2   # any local sentence-transformers model
CLIPMIND_EMBED_DEVICE=cpu               # cpu | mps | cuda
CLIPMIND_MIN_SCORE=0.35
CLIPMIND_TOP_K=10
```

---

## Usage (Phase 3: hybrid search + time filters)

`search` is now **hybrid** by default. It fuses the semantic score with an FTS5
BM25 keyword score, so a query can be answered by meaning *or* by exact string,
and any time phrase in the query is applied as a date range **before** ranking.

### What you can type

| You type | What happens |
| -------- | ------------ |
| `python -m clipmind search "database query"` | semantic + keyword fusion ranks `SELECT id FROM orders;` first |
| `python -m clipmind search "ERR_4021"` | found by the keyword half even though the string embeds poorly; an exact token hit bypasses the threshold |
| `python -m clipmind search "react hooks article yesterday"` | "yesterday" becomes a date filter; only yesterday's entries are ranked for "react hooks article" |
| `python -m clipmind search "that SQL query about orders I copied yesterday"` | same, with the time phrase removed from the searchable text |
| `python -m clipmind search "monday.com link"` | *not* misread as "last Monday" - the phrase is searched as-is |
| `python -m clipmind search "yesterday"` | a bare time phrase lists everything in that range, newest first |

Supported time phrases: `today`, `yesterday`, `this morning` / `this afternoon`
/ `this evening`, `last night`, `N minutes` / `N hours` / `N days` / `N weeks
ago`, `last <weekday>`, `this week`, `last week`, and
`between 2pm and 4pm [today|yesterday]`. A **bare** weekday or clock time
(`monday`, `5pm`) is deliberately left in the query and reported as ambiguous,
because it may be part of the content.

### Flags

| Flag | Meaning |
| ---- | ------- |
| `--top N` | Maximum results (default 5). |
| `-t/--type links\|code\|text\|email\|number\|all` | Restrict to one content type. |
| `--min-score S` | Minimum **combined** score (default 0.25). |
| `--semantic-only` | Ignore the keyword half (pure vector search). |
| `--keyword-only` | Ignore the semantic half (pure FTS5 BM25; needs **no model** and finds unembedded rows such as bare numbers). |
| `--explain` | Show the `FINAL / SEM / KW / COV` breakdown for each hit, plus notes. |
| `--json` | JSON lines; with `--explain` the breakdown fields are included. |

### Explain a result

```bash
python -m clipmind search "react hooks article yesterday" --explain
```

```
Time filter: yesterday  (2026-10-06 00:00 → 2026-10-07 00:00)
  #   FINAL     SEM      KW    COV        AGE  TYPE    PREVIEW
----------------------------------------------------------------
  1   0.612   0.731   0.333   0.67     2m ago  link    https://react.dev/reference/react/hooks

1 result(s); weights sem=True kw=True; threshold 0.25
```

`FINAL = w_sem * SEM + w_kw * KW`, where `KW` is the coverage-weighted
normalised BM25. An exact keyword hit that would otherwise fall below the
threshold is kept and marked `*`.

### Tuning the hybrid

```bash
CLIPMIND_SEMANTIC_WEIGHT=0.7     # weight of the semantic score
CLIPMIND_KEYWORD_WEIGHT=0.3      # weight of the keyword score
CLIPMIND_CANDIDATE_POOL=20       # candidates each retriever contributes before fusion
```

The day-part windows and the week start are `Config` fields
(`day_part_morning=(5, 12)`, `day_part_afternoon=(12, 17)`,
`day_part_evening=(17, 21)`, `day_part_night=(21, 5)`, `week_starts_on=0`).
Edit them in `config.toml` (see the Privacy & configuration section below).

---

## Capture rules

| Rule | Behaviour |
| ---- | --------- |
| Empty/whitespace-only | Skipped. |
| Exact duplicate | Skipped (SHA-256 of the stored text, `UNIQUE` index). |
| Longer than 10,000 chars | Truncated by default; set `oversize_policy = "skip"` to drop instead. |
| NUL bytes | Stripped. |
| Surrounding blank lines | Stripped — leading indentation is preserved so copied code survives intact. |
| Content type | `url` / `email` / `number` / `code` / `text`, by heuristic (see `capture/detect.py`). |

`config.py` holds the defaults (`max_chars`, `oversize_policy`, `poll_interval`,
`dedup`). An optional `config.toml` overrides them, and `CLIPMIND_*` environment
variables override the file (see the Privacy & configuration section below).

---

## Example queries

| You type | Behaviour |
| -------- | --------- |
| `python -m clipmind search "database query"` | ✅ finds `SELECT id FROM orders;` (semantic + keyword) |
| `python -m clipmind search "react hooks article"` | ✅ finds the React docs link |
| `python -m clipmind search "meeting with Rahul" --type text` | ✅ type-filtered |
| `python -m clipmind search "that SQL query about orders I copied yesterday"` | ✅ date-filtered to yesterday, then searched |
| `python -m clipmind search "the link about React hooks from this morning"` | ✅ date-filtered to this morning |
| `python -m clipmind search "ERR_4021"` | ✅ exact-keyword hit (threshold bypassed) |
| `python -m clipmind search "monday.com link"` | ✅ *not* misread as "last Monday"; searched as-is |

Phase 3 completes the pipeline: the meaning-based half, exact keyword matching,
and time parsing all work together, so the full natural-language examples at the
top of this README work end to end.

---

## Privacy notes

ClipMind is local-first by construction:

* **Storage is local.** Everything lives in one SQLite file under your user data
  directory. Nothing is synced or uploaded.
* **No network calls after setup.** The only outbound request in the whole
  project is the one-time download of the `all-MiniLM-L6-v2` weights (~90 MB)
  from Hugging Face, handled by `sentence-transformers`. After that the model is
  read from local cache and every embed/search runs on this machine — no API
  keys, no accounts, no cloud inference, no telemetry. You can verify it: unplug
  the network and `search`/`reindex` still work.
* **You can inspect it.** It is a normal SQLite database:
  `sqlite3 ~/Library/Application\ Support/clipmind/clipmind.db "select * from items"`.

Phase 4 adds a privacy layer at the capture boundary. Everything runs locally,
and no detected value is ever written to a log.

### What is detected

| Category | `kind` | Default action |
| -------- | ------ | -------------- |
| PEM private-key blocks | `private_key` | **skip** |
| Credit-card numbers (13-19 digits, **Luhn-checked**) | `credit_card` | **skip** |
| Password-like assignments (`password=...`, `secret: ...`) | `password` | **skip** |
| AWS access key ids (`AKIA...` / `ASIA...`) | `aws_access_key` | mask |
| GitHub tokens (`ghp_...`, `github_pat_...`) | `github_token` | mask |
| Slack tokens (`xoxb-...`) | `slack_token` | mask |
| OpenAI/Anthropic keys (`sk-...`) | `openai_key` | mask |
| Google API keys (`AIza...`) | `google_api_key` | mask |
| JSON Web Tokens | `jwt` | mask |

**skip** means the entry is not stored at all. **mask** stores it with the
secret replaced by `[REDACTED:<kind>]`. The masked text is what gets hashed,
embedded and indexed, so the raw secret never reaches the database, the FTS5
index, or the embedding model.

Detection is **best-effort, not a guarantee**: it recognises the shapes above
and nothing more. Ordinary text - UUIDs, git hashes, phone numbers, order IDs,
timestamps - is deliberately not flagged; card numbers must pass Luhn and JWTs
must decode to a header with `alg`, which keeps false positives low.

```toml
# config.toml
privacy_enabled = true
[secret_actions]
aws_access_key = "mask"
credit_card = "skip"
```

### Source-application ignore list

Captures from a configurable list of apps are skipped entirely (defaults:
1Password, Bitwarden, KeePassXC, Keychain Access, LastPass). Matching is
case-insensitive and substring-based. This relies on knowing the frontmost app,
which on macOS needs `pyobjc` and is otherwise empty - so treat it as
defence-in-depth, not a guarantee (see Limitations).

### Pause, data management, retention

```bash
clipmind pause                 # stop capturing; a running watch honours it live
clipmind resume
clipmind status                # pause state + database summary
clipmind delete 42             # remove one entry (its FTS row goes with it)
clipmind clear --yes           # wipe everything (prompts without --yes)
clipmind purge --older-than 30d
```

Set `retention_days` in `config.toml` to auto-purge old entries on watcher
startup and once a day. Bulk deletes run `VACUUM`, and the FTS5 delete triggers
keep the keyword index exact.

---

## Usage (Phase 5: desktop app)

`clipmind app` is **one long-running process** that owns the clipboard watcher
(with the Phase 4 privacy filters), the embedding service (model loaded **once**
and kept warm), the tray icon and the global hotkey.

```bash
clipmind app                 # popup + tray + global hotkey
clipmind app --no-hotkey     # tray only (e.g. permission not granted yet)
clipmind app --no-tray
clipmind app --no-embed      # keyword-only search; torch is never imported
clipmind app --check         # build everything, print a summary, exit
```

A second instance refuses to start:

```text
$ clipmind app
ClipMind is already running (pid 4711).
```

That is enforced by an advisory lock file (`app.lock`, next to the database)
which the OS releases automatically when the process exits, so a crash never
leaves a stale lock behind.

### Global hotkey

| Platform | Default |
| -------- | ------- |
| macOS | `Cmd+Shift+V` |
| Windows / Linux | `Ctrl+Shift+V` |

Press it to toggle the popup. Set your own in `config.toml`:

```toml
hotkey = "cmd+shift+v"     # cmd / ctrl / alt / shift + a key
```

The listener runs on its own thread and only ever **emits a Qt signal**, so no
widget is ever touched off the Qt main thread.

### macOS permission

macOS gates global key monitoring behind **Input Monitoring**. Without it the
hotkey registers but receives nothing, so ClipMind checks on first start and
shows a dialog naming the exact steps:

1. **System Settings → Privacy & Security → Input Monitoring**.
2. Turn on the switch next to **ClipMind** (or your terminal, when you run
   ClipMind from a shell) — click **+** and add it if it is not listed.
3. Quit and reopen ClipMind.

Until the permission is granted the hotkey will not fire, but the tray /
menu-bar icon still opens the popup. With `pyobjc-framework-Quartz` installed
(it comes in as a `pynput` dependency on macOS), `clipmind app --check` reports
the current grant.

### Popup

* Search-on-type with a **150 ms debounce**; searches run on a worker thread so
  typing never blocks, and superseded searches are dropped by a generation
  counter (stale results never reach the UI).
* The same **hybrid + time-aware** search as the CLI, so `react hooks yesterday`
  works in the box. The detected window shows as a dismissible chip
  (`⏱ Yesterday ✕`), and ambiguous time shapes are reported rather than filtered.
* An empty query lists the most recent entries.
* Every row shows a **type badge**, an ellipsised single-line **preview**
  (monospace for `code`), the **relative age** (`2h ago`) and — with **Explain**
  on — the `final/semantic/keyword` scores.
* Keyboard: `↑`/`↓` move, `⏎` copies back and closes, `Esc` closes, `⌘⌫` (or
  `Del`) deletes with an **undo toast** (`⌘Z` undoes), and `Tab` cycles
  **All → Code → Links → Text**. Clicking away closes the popup.
* Light/dark follows the system; set `theme = "light" | "dark"` to pin it.
* **Copy-back is suppressed.** Copying an entry back updates the watcher's
  baseline, so the change is not re-captured, and the database's hash dedup
  guarantees an existing row is never re-inserted or bumped.
* If the model is still loading, the popup says **“Loading model…”** and keeps
  working; a locked database or missing model becomes an in-popup error state
  instead of a crash. Worker exceptions never take the app down.

### Tray / menu-bar

**Open ClipMind**, **Pause/Resume capture** (reflects the persisted flag live, so
a `clipmind pause` in a terminal is visible here within ~2 s), **Clear
history…** (with a confirmation), **Open config file**, **Quit ClipMind**. The
icon greys out and gains a slash while capture is paused.

On macOS ClipMind runs as a **menu-bar app with no Dock icon**: it calls
`NSApplication.setActivationPolicy_(Accessory)` at startup (best effort; needs
`pyobjc`). For a packaged `.app`, the equivalent is in `Info.plist`:

```xml
<key>LSUIElement</key>
<true/>
```

Set `hide_dock_icon = false` in `config.toml` to opt out.

---

## Demo script

A ~30-second recording sequence. Use a throwaway database so nothing mixes with
your real history:

```bash
export CLIPMIND_DB=/tmp/clipmind_demo.db
rm -f "$CLIPMIND_DB"*
clipmind app                 # the popup stays hidden until summoned
```

1. **Capture** — copy each of these in any app (browser, editor, terminal):
   * `https://react.dev/learn` (link)
   * `SELECT id FROM orders WHERE total > 100;` (code)
   * `Meeting notes for the Q3 roadmap planning session` (text)
   * `react hooks article`
2. **Open** — press **Cmd+Shift+V**. Four entries appear, newest first, search
   box focused.
3. **Search** — type `react hooks`. The article jumps to the top. Press **Tab**
   to filter to **Code**, again for **Links**, then back to **All**.
4. **Time filter** — type `react hooks yesterday`; a `⏱ Yesterday` chip appears.
   Click it to dismiss and watch the same query re-run without the filter.
5. **Explain** — click **Explain** to reveal `final/semantic/keyword` on each row.
6. **Copy back** — press `↓` and `⏎`. The popup closes and the text is on your
   clipboard. Reopen it: still one entry, no bumped timestamp.
7. **Delete + undo** — select a row, press **Cmd+Backspace**, then **Cmd+Z**.
8. **Pause** — open the tray menu and click **Pause capture**; the icon greys
   out and newly copied text is not stored. Click **Resume capture** to finish.

---

## Testing

```bash
pip install -r requirements.txt
python -m pytest              # 454 passed, 6 skipped, ~7s, no model, no network
python -m pytest --runslow    # + 6 integration tests against the real model
```

The fast suite uses a deterministic fake embedder (`tests/fakes.py`), so it
never loads torch or touches the network. It covers content-type detection,
normalisation/truncation/hashing, schema + FTS5 sync, dedup (including the
unique-index race path), date-range and type filters, the monitor's
baseline/dedup/stats behaviour, the BLOB codec, the numpy vector index, the
embedding worker (including a test that the capture loop does **not** block on a
slow embedder), reindex eligibility/idempotence/progress, the time-phrase parser
(including the ambiguous `monday.com` case), FTS5 query sanitising and BM25
normalisation, hybrid fusion (weights, merge-by-id, the exact-keyword threshold
bypass, and date pre-filtering), secret detection and the skip/mask policy
(including assertions that a masked secret never reaches the DB, the FTS index
or the embedding input), the source-app ignore list, pause/resume, retention
purge, `config.toml` loading/validation, and every CLI command.

The six `slow` integration tests load the real `all-MiniLM-L6-v2`. Three cover
semantic ranking: `"database query"` ranks `SELECT id FROM orders;` first,
`"react hooks article"` ranks the React link first, and an unrelated query
returns nothing above the threshold. Three cover the hybrid path end to end:
`"react hooks article yesterday"` returns only yesterday's React link,
`"ERR_4021"` is found by the keyword retriever, and `"database query"` still
finds the SQL entry.

Phase 5 adds display-free tests for the popup view-model (debounce coalescing,
stale-result cancellation, selection/wrap, filter cycling, the time chip,
copy-back, and delete/undo), copy-back suppression at both the monitor-baseline
and hash-dedup layers, hotkey parsing, the single-instance lock, and the
search-backend adapter. A `pytest-qt` smoke test builds the popup and renders
results under `QT_QPA_PLATFORM=offscreen` (it is skipped cleanly when PySide6 or
pytest-qt are missing), and a `clipmind app --check` test exercises the full
wiring headlessly. The base suite never imports PySide6 from the non-Qt modules.

### Manual test checklist (needs a real GUI/hotkey session)

The sandbox this was developed in has **no display and no Input Monitoring
grant**, so the following are written but **unverified** and must be checked by
hand:

- [ ] `clipmind app` shows a menu-bar icon and **no** Dock icon.
- [ ] `Cmd+Shift+V` toggles the popup from another app (incl. full-screen); the
      search box has focus immediately.
- [ ] Typing filters the list within ~150 ms without freezing the UI.
- [ ] `↑`/`↓`/`⏎`/`Esc`/`Tab`/`⌘⌫`/`⌘Z` behave as described above.
- [ ] Clicking another app closes the popup.
- [ ] `⏎` copies back; the reopened popup shows the same single entry (no new
      row, no bumped timestamp).
- [ ] Tray **Pause/Resume** toggles the icon and stops/starts capture, and a
      `clipmind pause` in a terminal updates the menu within ~2 s.
- [ ] Tray **Clear history…** confirms, then empties the list.
- [ ] Tray **Open config file** opens `config.toml` (creating it if missing).
- [ ] On a fresh machine the first run shows the Input Monitoring dialog with
      the right steps; granting it and reopening makes the hotkey work.
- [ ] Switching macOS appearance (and `theme = "dark"`) re-themes the popup.
- [ ] A second `clipmind app` prints "already running" and exits 2.

---

## Benchmark results

> Phase 6 deliverable: search latency over 10,000 synthetic entries
> (p50/p95, semantic vs keyword vs hybrid). Not yet measured — no numbers are
> reported here rather than estimated ones.
>
> What *is* measured (this machine: Intel Mac, Python 3.12, torch 2.2.2,
> sentence-transformers 3.4.1, `all-MiniLM-L6-v2`, model cached; a single
> process, wall clock, no benchmark harness):
>
> | Operation | Measured |
> | --------- | -------- |
> | Cold model load + first embed | 11.1 s |
> | Warm embed, batch of 32 | 0.336 s (~95 texts/s) |
> | Warm embed, single query | 0.167 s |
> | `reindex` of 3 entries (includes the load) | 12.1 s |
> | `search` CLI, fresh process (load-dominated) | 11.1 s |
>
> These are single runs on one machine, not a distribution, and they are **not**
> the Phase 6 deliverable. The 10,000-entry p50/p95 latency comparison of
> semantic vs keyword vs hybrid is still unmeasured, so no numbers are claimed
> for it.

---

## Limitations (through Phase 5)

* **Text only.** No images, files or rich content.
* **One process at a time.** `clipmind app` is the long-running process and
  refuses to start twice; `clipmind watch` is still a separate foreground
  command. Run one against a database, not both.
* **`source_app` is usually empty on macOS** unless you install `pyobjc`
  (`pip install pyobjc-framework-Cocoa`). The column is always populated when
  the frontmost app can be determined, and can be forced with
  `CLIPMIND_SOURCE_APP` for testing.
* **Type detection is heuristic.** A one-line SQL statement without
  punctuation may be labelled `text`; a prose sentence containing two uppercase
  SQL keywords may be labelled `code`.
* **The hybrid threshold applies to the combined score.** With weights 0.7/0.3 a
  purely semantic hit needs a cosine similarity of ~0.36 to clear the default
  `0.25`; lower `--min-score` when both halves are borderline.
* **Keyword tokens are OR-joined.** A query is reduced to alphanumeric tokens and
  any one of them can match, so a partial hit may be returned; its `COV` (token
  coverage) is shown under `--explain`, and it is only rescued from the threshold
  when *all* tokens are present.
* **The time parser supports a fixed phrase set** (`today`, `yesterday`,
  `this morning/afternoon/evening`, `last night`, `N minutes/hours/days/weeks
  ago`, `last <weekday>`, `this week`, `last week`, `between H and H
  [yesterday]`). Anything else stays in the query; ambiguous shapes such as
  `monday.com` or a bare `5pm` are reported rather than silently filtered.
* **Numbers and very short entries are never embedded**, so they are invisible to
  the *semantic* half of a search - the keyword half still finds them.
* **`min_score` is a global threshold.** It is tuned for `all-MiniLM-L6-v2`;
  changing `CLIPMIND_EMBED_MODEL` will need re-tuning. Scores are not calibrated
  probabilities.
* **Brute force does not scale forever.** Every query needs the whole matrix in
  memory (~75 MB at 50k entries) and decodes it after any change. Fine to
  ~50k entries; the `VectorIndex` seam exists for FAISS/Chroma beyond that.
* **The embedding model is a hard dependency for `search`/`reindex`.**
  `watch --no-embed` and all Phase 1 commands work without torch installed. When
  the model is missing, `search`/`reindex` print one actionable line to stderr
  and exit 1 (no traceback), and a running `watch` keeps capturing - it just
  reports the embeddings as failed instead of dying.
* **No encryption at rest.** The SQLite file is protected only by filesystem
  permissions.
* **Secret detection is best-effort.** The detectors match known credential
  shapes; a secret in an unlisted format is stored unmasked. Treat masking as a
  safety net, not a guarantee.
* **The source-app ignore list needs app detection.** On macOS without
  `pyobjc`, and on Linux/Windows generally, the frontmost app is usually
  unknown, so the ignore list cannot fire - the secret detectors still run.
* **The UI was verified headlessly, not visually.** The suite runs the popup
  under `QT_QPA_PLATFORM=offscreen`. Real rendering, the menu-bar app, the
  permission dialog and the global hotkey itself were **not** exercised in the
  development sandbox (no display). Use the manual checklist in the Testing
  section.
* **Windows/Linux UI is untested.** The hotkey default, the `pynput` backend and
  the tray are written portably, but only macOS was exercised. Menu-bar/Dock
  behaviour is macOS-only.
* **The tray icon needs a system tray.** Where none is available (headless,
  some Linux desktops) ClipMind logs a warning and the hotkey is the only way to
  summon the popup.
* **On macOS the hotkey needs Input Monitoring.** Without it the listener starts
  but receives no events; the tray menu is the fallback.
* **Undo re-inserts a deleted row with its original timestamp and embedding**,
  but it gets a new id.
* **`clear`/`purge` rewrite the database with `VACUUM`.** That is a hard delete
  as far as SQLite is concerned, but it is not a secure erase of free space on
  disk; use full-disk encryption if that matters to you.

---

## Project layout

```
clipmind/
├── clipmind/
│   ├── config.py            # Config dataclass + env overrides + data paths
│   ├── __main__.py          # `python -m clipmind`
│   ├── capture/
│   │   ├── detect.py        # url/email/number/code/text heuristics
│   │   ├── monitor.py       # polling loop, baseline + dedup, stats
│   │   ├── processor.py     # normalize → truncate → hash → classify
│   │   ├── source_app.py    # best-effort frontmost-app probe
│   │   └── cli.py           # watch / list / add / search / reindex / stats
│   ├── storage/
│   │   ├── db.py            # schema, FTS5, insert/dedup, queries
│   │   └── models.py        # ClipRecord, InsertResult
│   ├── embeddings/
│   │   ├── codec.py         # float32 BLOB encode/decode + cosine helpers
│   │   ├── encoder.py       # lazy sentence-transformers wrapper, eligibility
│   │   └── service.py       # background worker + reindex sweep
│   ├── search/
│   │   ├── vector_index.py  # VectorIndex protocol + numpy brute force
│   │   ├── semantic.py      # SemanticSearcher, SearchResult, min-score
│   │   ├── keyword.py       # FTS5 BM25 searcher + MATCH sanitising
│   │   ├── hybrid.py        # HybridSearcher, weighted fusion, --explain breakdown
│   │   └── timeparse.py     # deterministic time phrase -> local date window
│   ├── privacy/
│   │   ├── detectors.py     # regex detectors + Luhn/JWT validation
│   │   └── policy.py        # skip/mask policy, masking, ignore list
│   └── ui/                  # Phase 5 desktop UI
│       ├── view_model.py    # display-free controller (no Qt import)
│       ├── search_backend.py# hybrid + time-aware search adapter
│       ├── hotkey.py        # pynput listener + macOS permission helpers (no Qt)
│       ├── single_instance.py# app.lock (flock / msvcrt)
│       ├── clipboard.py     # copy-back that suppresses re-capture
│       ├── theme.py         # light/dark palettes -> Qt style sheet
│       ├── popup.py         # frameless Spotlight-style window + row delegate
│       ├── tray.py          # menu-bar icon + menu
│       └── app.py           # `clipmind app`: wires everything together
├── tests/                   # pytest suite (incl. offscreen pytest-qt smoke test)
├── requirements.txt
└── pyproject.toml           # packaging + `clipmind` entry point
```

---

## Quick verification

```bash
cd clipmind
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# --- Phase 1: capture + storage (fast, no model) ---
python -m pytest                       # 454 passed, 6 skipped
export CLIPMIND_DB=/tmp/clipmind_demo.db
python -m clipmind add "SELECT id FROM orders;"
python -m clipmind add "https://react.dev/learn"
python -m clipmind add "SELECT id FROM orders;"   # -> Duplicate of #1 - skipped.
python -m clipmind list                            # two rows, types code + link
python -m clipmind stats                           # Total entries: 2

# --- Phase 2: embeddings + semantic search ---
python -m clipmind reindex                         # first run downloads the model (~90s)
python -m clipmind search "database query"
#   #   SCORE       AGE  TYPE    PREVIEW
#   1   0.4xx   just now  code    SELECT id FROM orders;

python -m clipmind search "react hooks" --top 3    # only 2 entries, so 1 hit
python -m clipmind search "gardening tips"         # -> No good matches above 0.25 (...)
python -m clipmind search "order" --json           # machine-readable

# assert the Phase 2 behaviour end to end against the real model
python -m pytest --runslow -m slow -v              # 6 passed

# --- Phase 3: hybrid + time-aware search ---
python -m clipmind search "ERR_4021"                  # exact string, no reindex needed
python -m clipmind search "database query" --explain  # FINAL / SEM / KW / COV breakdown
python -m clipmind search "react hooks article yesterday"
python -m clipmind search "yesterday"                 # everything from yesterday, newest first
python -m clipmind search "monday.com link"           # reported as ambiguous, not filtered

# --- Phase 4: privacy + data management ---
python -m clipmind add "password=hunter2"           # -> Not stored (secret).
python -m clipmind add "my key AKIAIOSFODNN7EXAMPLE"
python -m clipmind list                             # key [REDACTED:aws_access_key]
python -m clipmind pause                            # a running watch stops storing
python -m clipmind status
python -m clipmind resume
python -m clipmind purge --older-than 30d
python -m clipmind config path                      # ~/.config/clipmind/config.toml
python -m clipmind config show
python -m clipmind config show --template           # documented defaults

# --- Phase 5: desktop app ---
python -m clipmind app --check        # headless wiring summary, exit 0
python -m clipmind app                # popup + tray + global hotkey
#   Cmd+Shift+V opens the popup; type "react hooks yesterday", press Enter
python -m pytest                      # 454 passed, 6 skipped

# --- live capture with automatic embedding ---
python -m clipmind watch     # in one tab; copy things in another app
python -m clipmind list      # then inspect what was captured
```

> Note: `CLIPMIND_DB` puts the demo database in `/tmp`. Without it, ClipMind uses
> `~/Library/Application Support/clipmind/clipmind.db` (macOS).
