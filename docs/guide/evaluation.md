# Evaluation

Marvin ships with a reproducible retrieval benchmark so changes to chunking,
embeddings, ranking, or fusion can be measured against an external
reference rather than vibes.

## LongMemEval-S

[LongMemEval](https://arxiv.org/abs/2410.10813) (ICLR 2025) is a
public benchmark for long-term chat memory. The "small" variant
(`LongMemEval-S`) contains 500 questions, each paired with a haystack
of ~50 prior chat sessions; the gold answer is contained in one or two
of those sessions. We use the
[`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
release, the same one that `agentmemory` reports against, so numbers are
directly comparable.

The harness measures **retrieval only**: for each question we build a
fresh in-memory index from that question's haystack, run a search with
the question text, and check whether any gold session id appears in the
top-K results.

### Metrics

For each question (and aggregated across all questions):

- **`recall_any@K`** — 1.0 if any gold session is in the top-K results.
- **`NDCG@10`** — normalised DCG with binary session-level relevance.
- **`MRR`** — reciprocal rank of the first gold session.

The published `agentmemory` headline is `recall_any@5`.

### Quick start

```bash
# 1. Download the cleaned dataset (~270 MB, one-off).
python scripts/download_longmemeval.py

# 2. Run the BM25 baseline on all 500 questions (~1 minute).
python -m marvin.eval \
    --dataset experiments/data/longmemeval_s_cleaned.json \
    --mode bm25 \
    --output experiments/results/bm25.json

# 3. Run hybrid retrieval (BM25 + dense vectors via fastembed).
python -m marvin.eval \
    --dataset experiments/data/longmemeval_s_cleaned.json \
    --mode hybrid \
    --output experiments/results/hybrid.json

# 4. Quick sanity check without downloading any embedding model.
python -m marvin.eval --dataset PATH --mode bm25 \
    --embedding-provider hash --limit 20
```

### Modes

| Mode     | Index streams                  | Notes                              |
| -------- | ------------------------------ | ---------------------------------- |
| `bm25`   | SQLite FTS5 only               | Fastest; deterministic; no model.  |
| `vector` | `sqlite-vec` only              | Pure dense retrieval ablation.     |
| `hybrid` | FTS5 + sqlite-vec, RRF fused   | Default. Mirrors `service.search`. |

`--rerank` composes orthogonally with any mode. The first stage fetches a
pool of chunks (controlled by `--rerank-depth`), the cross-encoder scores
each `(query, chunk_text)` pair, and the session-level result is the
max-pool of chunk scores. See [Reranking](#reranking) below.

#### First-stage over-fetch

Each first-stage ranker (FTS5 and `sqlite-vec`) pulls
`max(limit * first_stage_overfetch, first_stage_overfetch_min)` chunks
before RRF fusion. The defaults (multiplier `5`, floor `20`) preserve
the previous hardcoded behaviour and are now exposed via
`MARVIN_FIRST_STAGE_OVERFETCH` / `MARVIN_FIRST_STAGE_OVERFETCH_MIN` (or
the matching fields on `MarvinSettings`). Increase the multiplier for
deeper recall before reranking; decrease it to cut first-stage SQL
work. When `--rerank` is enabled, `--rerank-depth` is the second-stage
pool size; if it exceeds what RRF can produce given the per-stream
limit, the reranker gets fewer candidates than requested.

#### K-Lines graph stream

The `hybrid` mode is actually a three-stream pipeline: FTS5 +
`sqlite-vec` (chunk-level, max-pooled to notes) and a third *graph*
stream that ranks notes by edge weight to query entities. Wikilinks
extracted from note bodies (`[[Apple Card]]`) populate an
`entity_edges` table during indexing; query terms are resolved to
entities via case-fold word-boundary matching, and notes linking to
those entities are ranked by total edge weight. The graph note
ranking is RRF-fused with the chunk-tier note ranking before reranking.

The stream is silently a no-op on vaults that have not yet been
wikilink-consolidated (such as raw LongMemEval-S haystacks where chat
sessions have no `[[wikilinks]]`); enable at-ingest extraction in a
follow-up to populate the graph for unconsolidated content. Toggle
via `MARVIN_KG_ENABLED=false` if you want to ablate it; tune the
fusion damping with `MARVIN_KG_RRF_K` (default `60.0`).

### Baseline numbers

Run on commit `feature/eval-longmemeval`, multi-core CPU, no GPU,
default chunking (1200 / 200), full LongMemEval-S (500 questions).

| Mode   | Embedder            | R@5    | R@10  | R@20  | NDCG@10 | MRR   | Wall   |
| ------ | ------------------- | ------ | ----- | ----- | ------- | ----- | ------ |
| BM25   | n/a (FTS5)          | **95.6%** | 98.2% | 99.2% | 87.3%   | 89.2% | 55 s   |
| Hybrid | hash (deterministic)| 88.8%  | 92.2% | 98.2% | 74.7%   | 77.1% | 73 s   |

For reference, `agentmemory` reports `recall_any@5 = 95.2%` on the same
dataset using BM25 + dense vectors with cross-encoder reranking. Marvin's
BM25 alone matches that headline number; closing the remaining gap on the
hardest question types (especially `single-session-preference`) is the
target for follow-up work on dense retrieval and reranking.

The "hybrid + hash" row is informative rather than aspirational: when the
real embedder isn't available, Marvin currently falls back to a feature-
hashing backend whose vectors are essentially random. Mixing them into
RRF *hurts* relative to BM25-only, so production deployments should
always have `fastembed` (or a real embedding API) installed.

#### Cross-encoder reranker lift

Measured on a 100-question LongMemEval-S subset, BM25 first-stage (`hash`
embedder), `--rerank-depth 50`, CPU under heavy concurrent load.

| Mode           | R@5       | R@10  | NDCG@10   | MRR       | Median latency |
| -------------- | --------- | ----- | --------- | --------- | -------------- |
| BM25           | 96.0%     | 99.0% | 88.9%     | 89.5%     | 1.5 s          |
| BM25 + rerank  | **98.0%** | 99.0% | **94.6%** | **95.3%** | 125 s          |

Per question type:

| Type (n)                   | R@5 BM25 | R@5 +rerank | NDCG@10 BM25 | NDCG@10 +rerank | MRR BM25 | MRR +rerank |
| -------------------------- | -------- | ----------- | ------------ | --------------- | -------- | ----------- |
| single-session-user (n=70) | 97.1%    | 98.6%       | 96.0%        | 98.6%           | 94.7%    | 98.1%       |
| multi-session (n=30)       | 93.3%    | **96.7%**   | 72.4%        | **85.5%**       | 77.2%    | **88.6%**   |

The reranker's value is concentrated where it should be: the harder
`multi-session` slice jumps **+13.1 pp NDCG@10** and **+11.4 pp MRR**. On
the easier single-session slice it still delivers a clean +2.6 pp /
+3.4 pp. Recall@10 was already at ceiling so top-K movement is dominated
by ordering metrics rather than R@5.

The latency figure reflects the host this run was executed on (load
average ~40 during the whole run). On an idle workstation the same
depth-50 rerank runs in single-digit seconds per query, and on a GPU
it's milliseconds. Production MCP queries only pay the cost when
`rerank_enabled=true`.

#### Hybrid with `fastembed` on CPU

`fastembed`'s ONNX backend has roughly linear-then-superlinear cost in
sequence length on CPU. With the default 512-char embedding cap, a full
500-question hybrid run takes several hours on a typical workstation.
For interactive iteration:

- Cap embedding text aggressively: `--max-embed-chars 128` is ~5× faster
  than the default 512 with little recall impact in our spot checks.
- Use `--limit N` to evaluate on a subset.
- Use `--mode bm25` for changes that don't touch dense retrieval.

Reducing this cost is on the roadmap (smaller models, batched embedding
service, optional GPU/Metal backends).

### Reranking

Hybrid retrieval is strong at *finding* the right chunk but RRF only uses
rank order — it ignores query-document interactions. A cross-encoder
reranker reads the query and each candidate jointly and typically lifts
top-K precision by 5–15 points on open-domain QA tasks at the cost of a
few hundred ms per query on CPU.

The harness (and `MarvinService.search`) ship with an optional reranking
pass backed by [`fastembed`'s](https://qdrant.github.io/fastembed/)
`TextCrossEncoder`. The default model is
[`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3):
multilingual, 568M params, Apache-2.0. BAAI does not publish ONNX weights
directly, so Marvin registers the community
[`onnx-community/bge-reranker-v2-m3-ONNX`](https://huggingface.co/onnx-community/bge-reranker-v2-m3-ONNX)
int8 port (~570 MB, single file) via `TextCrossEncoder.add_custom_model`
the first time the reranker is constructed. Any reranker listed by
`TextCrossEncoder.list_supported_models()` works too — pass e.g.
`--rerank-model Xenova/ms-marco-MiniLM-L-6-v2` for a faster, English-only
alternative.

```bash
# BM25 retrieval + cross-encoder reranking on the first 50 questions.
python -m marvin.eval \
    --dataset experiments/data/longmemeval_s_cleaned.json \
    --mode bm25 \
    --rerank \
    --rerank-depth 50 \
    --limit 50 \
    --output experiments/results/bm25_rerank.json
```

Flags:

- `--rerank` — enable the cross-encoder.
- `--rerank-model` — HF model id (default: `BAAI/bge-reranker-v2-m3`).
- `--rerank-depth` — first-stage chunk pool size (default: 50). Chunks,
  not sessions: several chunks from the same session are scored
  independently and max-pooled back.
- `--rerank-max-chars` — per-document truncation before tokenisation
  (default: 1024). Keeps CPU cost bounded.

Why *chunk* reranking rather than session-level? LongMemEval sessions are
long conversations (commonly 10–20 KB of raw turns). The reranker's input
window is effectively 512 tokens, so naively prefixing a whole session
discards the very signal we need. Scoring the chunks that first-stage
retrieval already matched, then max-pooling to sessions, recovers the
signal cleanly.

Performance: with `bge-reranker-v2-m3` quantized to int8 on CPU, 50
`(query, chunk)` pairs take roughly 2–4 seconds on an idle workstation
and much more on a heavily-loaded box (see the
[measured lift table](#cross-encoder-reranker-lift) for a worst-case
number). Budget a few minutes of wall time per 100 questions in the
harness, or pick a smaller model (e.g.
`Xenova/ms-marco-MiniLM-L-6-v2`) for interactive iteration. MCP gateway
queries pay the reranker cost once per `search()` call and only when
`rerank_enabled` is set.

### Output

The CLI prints a per-question-type breakdown and writes a JSON dump:

```json
{
  "mode": "bm25",
  "embedding_provider": "hash",
  "questions": 500,
  "recall_at_5": 0.956,
  "recall_at_10": 0.982,
  "ndcg_at_10": 0.873,
  "mrr": 0.892,
  "median_latency_ms": 111.1,
  "total_seconds": 55.4,
  "per_type": { "...": "..." },
  "per_question": [ "...", "..." ]
}
```

`per_question` includes the retrieved session ids, gold ids, and per-question
metrics — useful for digging into failure cases.

### Programmatic API

```python
from pathlib import Path
from marvin.embeddings import EmbeddingService
from marvin.eval.longmemeval import load_dataset, run_benchmark
from marvin.reranker import RerankerService

entries = load_dataset(Path("experiments/data/longmemeval_s_cleaned.json"))
summary = run_benchmark(
    entries[:50],
    mode="hybrid",
    embedder=EmbeddingService(),
    reranker=RerankerService(provider="fastembed"),
    rerank_depth=50,
)
print(summary.recall_at_5, summary.mrr)
```
