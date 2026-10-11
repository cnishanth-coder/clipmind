# ClipMind benchmark results

_Measured on 2026-10-11 05:45 UTC by `benchmarks/bench_search.py` (seed `20261007`, corpus 10,000 entries, 200 latency queries, 45 labelled pairs). All figures below are measured on this machine, not estimates._

## Environment

| Item | Value |
| --- | --- |
| OS / platform | `macOS-26.6.2-x86_64-i386-64bit` |
| CPU | `Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz` |
| Architecture | `x86_64` |
| Python | `3.12.15` |
| PyTorch | `2.2.2` (8 threads) |
| sentence-transformers | `3.4.1` |
| NumPy | `1.26.4` |
| Device | `auto` |
| Models benchmarked | all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5, sentence-transformers/all-mpnet-base-v2 |
| Current default model | `all-MiniLM-L6-v2` |

## Corpus

10,000 synthetic clipboard entries, generated with a fixed seed (`20261007`) and timestamps spread uniformly over the previous 90 days. The 45 labelled expected entries are part of it.

| Content type | Count |
| --- | ---: |
| text | 3,649 |
| code | 1,793 |
| url | 2,224 |
| email | 1,878 |
| number | 456 |
| **total** | **10,000** |

## Search latency: before vs after the index-reuse fix (Task 1)

Before: every `hybrid_search` call built a fresh `SemanticSearcher`, so each query re-read all embedding BLOBs from SQLite. After: one long-lived in-memory index, updated incrementally. Both rows use `all-MiniLM-L6-v2` on the same corpus; before is the recorded pre-fix run (`benchmarks/results_before.json`), after is this run.

| Mode | before p50 | before p95 | after p50 | after p95 | p50 speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| semantic | 227.7 ms | 418.5 ms | 44.5 ms | 73.7 ms | 5.1x |
| keyword | 5.0 ms | 12.6 ms | 4.3 ms | 9.8 ms | 1.2x |
| hybrid | 234.3 ms | 368.7 ms | 49.6 ms | 62.0 ms | 4.7x |

## Model comparison (Task 2)

### Model size and embedding throughput

Weights (MB) is the parameter count at fp32 (the on-disk weight file is the same within a few percent). Throughput is the 10,000-text corpus pass in chunks of 256.

| Model | Dim | Params (M) | Weights (MB) | Load (s) | Pass (s) | Throughput (texts/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `all-MiniLM-L6-v2` | 384 | 22.7 | 91 | 7.46 | 22.2 | **450.9** |
| `BAAI/bge-small-en-v1.5` | 384 | 33.4 | 133 | 3.79 | 25.2 | **397.5** |
| `sentence-transformers/all-mpnet-base-v2` | 768 | 109.5 | 438 | 4.36 | 39.8 | **251.3** |

Prefixes applied (query / document): `all-MiniLM-L6-v2` → '' / ''; `BAAI/bge-small-en-v1.5` → 'Represent this sentence for searching relevant passages: ' / ''; `sentence-transformers/all-mpnet-base-v2` → '' / ''.

### Query latency after Task 1 (per model)

One long-lived searcher per model over the shared query set; wall-clock milliseconds per query.

| Model | semantic p50 | semantic p95 | keyword p50 | hybrid p50 | hybrid p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `all-MiniLM-L6-v2` | 44.5 ms | 73.7 ms | 4.3 ms | 49.6 ms | 62.0 ms |
| `BAAI/bge-small-en-v1.5` | 74.8 ms | 89.1 ms | 4.3 ms | 79.7 ms | 94.3 ms |
| `sentence-transformers/all-mpnet-base-v2` | 81.6 ms | 154.6 ms | 4.3 ms | 86.3 ms | 92.6 ms |

### Retrieval quality: all 45 labelled queries

| Model | semantic top-1 | semantic top-5 | hybrid top-1 | hybrid top-5 |
| --- | ---: | ---: | ---: | ---: |
| `all-MiniLM-L6-v2` | 51.1% | 68.9% | 51.1% | **71.1%** |
| `BAAI/bge-small-en-v1.5` | 42.2% | 57.8% | 48.9% | **57.8%** |
| `sentence-transformers/all-mpnet-base-v2` | 53.3% | 77.8% | 60.0% | **80.0%** |

### Retrieval quality: the 15 code-focused queries

| Model | semantic top-1 | semantic top-5 | hybrid top-1 | hybrid top-5 |
| --- | ---: | ---: | ---: | ---: |
| `all-MiniLM-L6-v2` | 52.0% | 60.0% | 48.0% | **60.0%** |
| `BAAI/bge-small-en-v1.5` | 40.0% | 44.0% | 48.0% | **48.0%** |
| `sentence-transformers/all-mpnet-base-v2` | 68.0% | 80.0% | 68.0% | **80.0%** |

## Default model detail (`all-MiniLM-L6-v2`)

The real `all-MiniLM-L6-v2` model was loaded once (7.46 s) and then used to embed the whole corpus in chunks of 256.

| Metric | Value |
| --- | ---: |
| Model load (first call) | 7.46 s |
| Texts embedded | 10,000 |
| Embedding pass | 22.18 s |
| Throughput | **450.9 texts/s** |
| Throughput (chars) | 22,896 chars/s |

| Mode | p50 | p95 | p99 | mean | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| semantic | 44.5 ms | 73.7 ms | 208.8 ms | 53.8 ms | 387.0 ms |
| keyword | 4.3 ms | 9.8 ms | 10.7 ms | 4.6 ms | 14.1 ms |
| hybrid | 49.6 ms | 62.0 ms | 73.3 ms | 50.5 ms | 98.4 ms |

Top-1 / top-5 hit rate on 45 labelled pairs, `min_score = 0` so the modes are compared purely on ranking. Code subset = 25 queries.

| Mode | all: top-1 | all: top-5 | code: top-1 | code: top-5 |
| --- | ---: | ---: | ---: | ---: |
| semantic | 51.1% | 68.9% | 52.0% | 60.0% |
| keyword | 48.9% | 60.0% | 48.0% | 60.0% |
| hybrid | 51.1% | 71.1% | 48.0% | 60.0% |

## Interpretation

* **The per-query rebuild is gone.** Semantic p50 fell from 228 ms to 44 ms (5.1x) for `all-MiniLM-L6-v2`; a query is now a masked matmul over the warm matrix plus one model forward pass. Keyword search is unchanged (FTS5/BM25 inside SQLite).
* **Retrieval quality (all 45 queries, hybrid top-5):** `all-MiniLM-L6-v2` 71%, `BAAI/bge-small-en-v1.5` 58%, `sentence-transformers/all-mpnet-base-v2` 80%. Measured, not assumed.
* **Code queries (15) hybrid top-5:** `all-MiniLM-L6-v2` 60%, `BAAI/bge-small-en-v1.5` 48%, `sentence-transformers/all-mpnet-base-v2` 80%.
* **Default model stayed `all-MiniLM-L6-v2`.** The best alternative, `sentence-transformers/all-mpnet-base-v2`, measured +8.9 points overall and +20.0 on code (hybrid top-5), but costs 4.8x the weights (438 MB vs 91 MB), 1.7x the hybrid latency and half the embedding throughput, and would need `min_score` re-tuned. That total cost is not clearly acceptable for a lightweight local tool, so the smaller model is kept; set `embedding_model = "sentence-transformers/all-mpnet-base-v2"` to opt in.
* **Reproducibility:** the corpus and labels are deterministic (fixed seed); the throughput and latency columns depend on the host and its load.

### Labelled queries used

Rank of the expected entry in each mode for `all-MiniLM-L6-v2` (lower is better; `—` means it did not appear in the top 5). The 15 code-focused queries are the last rows.

| # | Query | semantic | keyword | hybrid | Expected entry |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | python query the orders table by customer id | — | — | — | `SELECT order_id, total FROM orders WHERE customer_id = 4021;` |
| 2 | sql average order value per month | 1 | — | 1 | `SELECT strftime('%Y-%m', created_at) AS month, AVG(total) FROM orde…` |
| 3 | git command to squash the last three commits | 3 | — | 3 | `git rebase -i HEAD~3` |
| 4 | javascript fetch with async await | — | — | — | `async function loadUser(id) {   const res = await fetch(`/api/users…` |
| 5 | reverse a linked list in python | 1 | 1 | 1 | `def reverse_list(head):     prev = None     while head:         hea…` |
| 6 | docker build and tag an image | — | 2 | — | `docker build -t clipmind:latest .` |
| 7 | regex to validate an email address | — | — | — | `^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$` |
| 8 | bash loop over log files and gzip them | 1 | 1 | 1 | `for f in *.log; do gzip "$f"; done` |
| 9 | react hook that fetches data on mount | — | — | — | `useEffect(() => {   fetchData().then(setData); }, []);` |
| 10 | kubectl list pods across namespaces | 2 | 1 | 2 | `kubectl get pods -n production -o wide` |
| 11 | react refs documentation page | 1 | 1 | 1 | `https://react.dev/learn/referencing-values-with-refs` |
| 12 | rust book chapter about ownership | 1 | 1 | 1 | `https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html` |
| 13 | postgres window functions tutorial | 1 | 1 | 1 | `https://www.postgresql.org/docs/current/tutorial-window.html` |
| 14 | mdn array map reference | — | 1 | 1 | `https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/G…` |
| 15 | python pathlib documentation | 2 | 4 | 3 | `https://docs.python.org/3/library/pathlib.html` |
| 16 | sqlite full text search fts5 docs | 1 | 2 | 2 | `https://www.sqlite.org/fts5.html` |
| 17 | django migrations guide | 2 | — | 2 | `https://docs.djangoproject.com/en/stable/topics/migrations/` |
| 18 | email the design team about the landing page review | — | — | — | `landing-page-review@acme-corp.com` |
| 19 | invoice from the cloud hosting provider | 3 | — | 3 | `billing@cloudhost.io` |
| 20 | support address for the payments vendor | 1 | 1 | 1 | `support@stripe-payments.example` |
| 21 | HR contact for onboarding at acme | 1 | 1 | 1 | `onboarding@acme-corp.com` |
| 22 | sprint planning moved to Thursday afternoon | — | — | — | `Standup moves to 9:30; sprint planning is now Thursday at 3pm in ro…` |
| 23 | when does the wildcard ssl certificate expire | 1 | 1 | 1 | `The wildcard SSL certificate for *.example.com expires on 2026-11-30.` |
| 24 | grocery list with oat milk and sourdough | 1 | 1 | 1 | `Shopping: oat milk, sourdough bread, eggs, spinach, coffee beans.` |
| 25 | idea for a cli that summarises git history | 1 | 1 | 1 | `Idea: a CLI that summarises the last week of git history into a cha…` |
| 26 | how do I reset the wifi router | 1 | — | 1 | `To reset the router, hold the pinhole button for ten seconds until …` |
| 27 | client quote about the dashboard speed | 2 | — | 3 | `"The dashboard feels much faster now - great work on the caching la…` |
| 28 | conference talk about local-first software | 5 | 1 | 1 | `Talk idea: local-first software, and why your app should work offline.` |
| 29 | twenty minute tomato pasta recipe | 4 | — | 4 | `Recipe: boil 200g spaghetti, saute garlic and tomatoes, toss with b…` |
| 30 | flight booking confirmation for Lisbon | — | — | — | `Flight TP1042 LIS, 2026-12-18 07:40, seat 14C, confirmation XR7Q2P.` |
| 31 | sort a list of dicts by key in python | 1 | 1 | 1 | `rows.sort(key=lambda r: r['created_at'])` |
| 32 | typescript interface for a user object | — | 1 | — | `interface User {   id: number;   email: string; }` |
| 33 | go routine waited on with a wait group | 1 | 1 | 1 | `var wg sync.WaitGroup wg.Add(1) go func() { defer wg.Done() }() wg.…` |
| 34 | rust get a map value or a default | — | 1 | — | `let value = map.get("key").unwrap_or(&default_value);` |
| 35 | css flexbox center a div | 1 | 1 | 1 | `.center {   display: flex;   align-items: center;   justify-content…` |
| 36 | postgres delete duplicate rows keeping one | 1 | 2 | 2 | `DELETE FROM users a USING users b WHERE a.id > b.id AND a.email = b…` |
| 37 | python read a csv file with pandas | 1 | 1 | 1 | `import pandas as pd df = pd.read_csv('data.csv')` |
| 38 | java stream filter and collect to list | 1 | 1 | 1 | `List<String> names = users.stream().filter(u -> u.isActive()).map(U…` |
| 39 | c++ sort a vector in descending order | 1 | 2 | 1 | `std::sort(v.begin(), v.end(), std::greater<int>());` |
| 40 | sqlite create an index on a column | — | 1 | — | `CREATE INDEX idx_orders_customer ON orders(customer_id);` |
| 41 | flask route that returns json | 1 | 1 | 1 | `@app.route("/api/health") def health():     return jsonify(status="…` |
| 42 | makefile target that runs the tests | 1 | — | 1 | `test: 	pytest -q` |
| 43 | regular expression to match an ipv4 address | — | — | — | `^(?:\d{1,3}\.){3}\d{1,3}$` |
| 44 | shell script fail fast on error | — | — | — | `set -euo pipefail` |
| 45 | sqlalchemy filter by a date range | 1 | — | 1 | `session.query(Order).filter(Order.created_at >= start, Order.create…` |

## Reproduce

```bash
python benchmarks/bench_search.py            # regenerates this file
python benchmarks/bench_search.py --json out.json
python benchmarks/bench_search.py --models all-MiniLM-L6-v2
```

> Latency and throughput depend on the host CPU and on how busy it is; the corpus and the retrieval-quality figures are reproducible.
