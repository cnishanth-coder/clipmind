# ClipMind benchmark results

_Measured on 2026-10-07 08:32 UTC by `benchmarks/bench_search.py` (seed `20261007`, corpus 10,000 entries, 200 latency queries, 30 labelled pairs). All figures below are measured on this machine, not estimates._

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
| Embedding model | `all-MiniLM-L6-v2` |
| Device | `mps:0` |

## Corpus

10,000 synthetic clipboard entries, generated with a fixed seed (`20261007`) and timestamps spread uniformly over the previous 90 days.

| Content type | Count |
| --- | ---: |
| text | 3,651 |
| code | 1,782 |
| url | 2,227 |
| email | 1,883 |
| number | 457 |
| **total** | **10,000** |

## Embedding throughput

The real `all-MiniLM-L6-v2` model was loaded once (8.95 s) and then used to embed the whole corpus in chunks of 256.

| Metric | Value |
| --- | ---: |
| Model load (first call) | 8.95 s |
| Texts embedded | 10,000 |
| Embedding pass | 26.90 s |
| Throughput | **371.7 texts/s** |
| Throughput (chars) | 18,866 chars/s |

## Search latency

Each of the 200 queries was run through all three modes against the full corpus. Values are wall-clock milliseconds per query.

| Mode | p50 | p95 | p99 | mean | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| semantic | 227.7 ms | 418.5 ms | 613.0 ms | 265.5 ms | 901.0 ms |
| keyword | 5.0 ms | 12.6 ms | 16.5 ms | 5.6 ms | 30.9 ms |
| hybrid | 234.3 ms | 368.7 ms | 437.9 ms | 259.1 ms | 651.3 ms |

## Retrieval quality

Top-1 / top-5 hit rate on 30 hand-written `query -> expected entry` pairs (the expected contents are part of the corpus). Scoring used `min_score = 0` so the modes are compared purely on ranking.

| Mode | top-1 | top-5 |
| --- | ---: | ---: |
| semantic | 43.3% | 70.0% |
| keyword | 43.3% | 53.3% |
| hybrid | 46.7% | 73.3% |

## Interpretation

* **Semantic latency is dominated by index rebuilding, not the model.** Every call to `hybrid_search` constructs a fresh `SemanticSearcher`, whose decoded-index cache therefore never hits, so each query re-reads all 10,000 embedding blobs from SQLite and restacks them into a numpy matrix. That is why semantic p50 (228 ms) is ~40x keyword p50 (5 ms); the query itself is a single `all-MiniLM-L6-v2` forward pass. Reusing one searcher across queries is the obvious optimisation (see the README roadmap).
* **Keyword search is essentially free** (FTS5/BM25 inside SQLite) and scales with the corpus far better than the current numpy scan.
* **Retrieval quality is limited by distractors.** The corpus contains thousands of near-duplicate entries (generic URLs, role emails, `SELECT ... FROM orders` snippets), so an exact top-1 is often ambiguous by construction. Hybrid ranking beats both single modes at top-5 (73% vs 70% semantic and 53% keyword).
* **Reproducibility:** the corpus and labels are deterministic (fixed seed); the throughput and latency columns depend on the host and its load.

### Labelled queries used

Rank of the expected entry in each mode (lower is better; `—` means it did not appear in the top 5). This is the raw data behind the table above.

| # | Query | semantic | keyword | hybrid | Expected entry |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | python query the orders table by customer id | — | — | — | `SELECT order_id, total FROM orders WHERE customer_id = 4021;` |
| 2 | sql average order value per month | 1 | — | 1 | `SELECT strftime('%Y-%m', created_at) AS month, AVG(total) FROM orde…` |
| 3 | git command to squash the last three commits | 3 | — | 3 | `git rebase -i HEAD~3` |
| 4 | javascript fetch with async await | — | — | — | `async function loadUser(id) {   const res = await fetch(`/api/users…` |
| 5 | reverse a linked list in python | 1 | 1 | 1 | `def reverse_list(head):     prev = None     while head:         hea…` |
| 6 | docker build and tag an image | — | 1 | — | `docker build -t clipmind:latest .` |
| 7 | regex to validate an email address | — | — | — | `^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$` |
| 8 | bash loop over log files and gzip them | 1 | 1 | 1 | `for f in *.log; do gzip "$f"; done` |
| 9 | react hook that fetches data on mount | — | — | — | `useEffect(() => {   fetchData().then(setData); }, []);` |
| 10 | kubectl list pods across namespaces | 2 | 2 | 2 | `kubectl get pods -n production -o wide` |
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

## Reproduce

```bash
python benchmarks/bench_search.py            # regenerates this file
python benchmarks/bench_search.py --json out.json
```

> Latency and throughput depend on the host CPU and on how busy it is; the corpus and the retrieval-quality figures are reproducible.
