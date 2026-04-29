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

An optional regex-based fallback extractor (`MARVIN_KG_EXTRACT_AT_INGEST=true`,
default `false`) populates entities from the body of unconsolidated
notes (chat sessions, freshly-pasted markdown). Enabling it adds a
multi-word capitalised noun phrase pass (`Apple Card`, `Western Australia`)
behind a stop-word filter that drops sentence-fragment phrases like
`But I` or `Once I`. We default it off because LongMemEval-S is
chat-style data: capitalised tokens are dominated by sentence-starter
imperatives (`Remember`, `However`, `Did`) rather than entities, and
the few real entities a query references rarely line up with what the
regex finds in the haystack. The empirical impact on the 100-question
slice (hash embedder, hybrid mode):

| | kg off | kg + wikilinks-only (default) | kg + wikilinks + at-ingest |
|---|---|---|---|
| R@5 | 86.0% | 86.0% | 84.0% (-2pp) |
| R@10 | 90.0% | 90.0% | 90.0% |
| NDCG@10 | 69.3% | 69.3% | 68.9% (-0.4pp) |
| MRR | 68.1% | 68.1% | 67.7% (-0.4pp) |
| multi-session R@5 | 90.0% | 90.0% | 83.3% (-7pp) |
| single-session R@5 | 84.3% | 84.3% | 84.3% |

Wikilinks-only graph (Phase 1A) is silent on raw chat data and
matches the chunk-only baseline exactly. Enable at-ingest when:

* the vault is wikilink-consolidated and you want freshly-ingested
  notes to contribute graph signal before the next consolidation
  pass;
* the source text is curated (docs, research notes) where
  capitalisation reliably marks proper nouns.

Toggles, all `MARVIN_KG_*` env vars or fields on `MarvinSettings`:

| setting | default | purpose |
|---|---|---|
| `kg_enabled` | `true` | toggle the third stream entirely |
| `kg_rrf_k` | `60.0` | RRF damping constant |
| `kg_fusion_weight` | `0.5` | graph-stream weight in fusion (`< 1` keeps strong chunk matches from being displaced) |
| `kg_extract_at_ingest` | `false` | regex fallback entity extraction |
| `kg_ingest_min_length` | `3` | drop short capitalised tokens |
| `kg_ingest_multiword_only` | `true` | drop single-word at-ingest entities |

The graph ranker is IDF-weighted (`log((N+1)/(df+0.5))`) so common
entities (the speaker's name, the platform brand) contribute little
and rare entities a lot, mirroring BM25's term weighting.

### Baseline numbers

Run on commit `feature/p2-real-embedder-bench`, full LongMemEval-S (500
questions), default chunking (1200 / 200), `fastembed` 0.8 with
`onnxruntime-gpu` 1.25 on a single RTX 4090 (CUDA 12.9, FP16 reranker
ONNX). Embedder: `BAAI/bge-small-en-v1.5` (384 dim, the latest in the
BGE-small line — there is no v2.0 yet). Reranker:
`BAAI/bge-reranker-v2-m3` (FP16 ONNX, depth 50).

| Mode               | R@5      | R@10  | R@20  | NDCG@10  | MRR      | Median latency | Wall  |
| ------------------ | -------- | ----- | ----- | -------- | -------- | -------------- | ----- |
| BM25 (FTS5 only)   | 95.6%    | 98.2% | 99.2% | 87.3%    | 89.2%    | 144 ms         | 88 s  |
| Hybrid (BM25+vec)  | 98.0%    | 99.0% | 99.6% | 91.9%    | 92.7%    | 853 ms         | 456 s |
| Hybrid + rerank    | **99.6%** | 99.6% | 99.8% | **95.3%** | **95.5%** | 1 178 ms       | 663 s |

Per question type (Hybrid + rerank, n=500):

| Type (n)                          | R@5     | R@10   | NDCG@10 | MRR    |
| --------------------------------- | ------- | ------ | ------- | ------ |
| knowledge-update (n=78)           | 100.0%  | 100.0% | 99.8%   | 100.0% |
| single-session-assistant (n=56)   | 100.0%  | 100.0% | 100.0%  | 100.0% |
| single-session-user (n=70)        | 100.0%  | 100.0% | 98.1%   | 97.5%  |
| multi-session (n=133)             | 100.0%  | 100.0% | 92.1%   | 92.5%  |
| temporal-reasoning (n=133)        | 99.2%   | 99.2%  | 93.7%   | 95.2%  |
| single-session-preference (n=30)  | 96.7%   | 96.7%  | 88.8%   | 86.2%  |

For reference, `agentmemory` reports `recall_any@5 = 95.2%` on the same
dataset using BM25 + dense vectors with cross-encoder reranking. Marvin
hits that headline with BM25 alone (95.6%) and pulls **+4.4 pp** ahead
with the full hybrid + reranker stack (99.6%). The historically-hardest
slice (`single-session-preference`) climbs from 73.3% on BM25 → 90.0%
hybrid → **96.7%** with reranking.

#### Cross-encoder reranker lift (full 500q)

Going from `Hybrid` to `Hybrid + rerank` on the full benchmark with the
real `bge-small-en-v1.5` embedder:

| Slice                              | Δ R@5    | Δ NDCG@10 | Δ MRR    |
| ---------------------------------- | -------- | --------- | -------- |
| All 500                            | +1.6 pp  | +3.4 pp   | +2.8 pp  |
| `single-session-preference` (n=30) | **+6.7 pp** | **+14.7 pp** | **+19.6 pp** |
| `single-session-user` (n=70)       | +2.9 pp  | +4.2 pp   | +4.4 pp  |
| `temporal-reasoning` (n=133)       | +2.2 pp  | +5.0 pp   | +3.5 pp  |
| `multi-session` (n=133)            | +0.8 pp  | +0.5 pp   | -1.2 pp  |

Reranking pays off most on the slices where order matters (preferences,
user-question recall). On `multi-session` the first stage already has
the gold session in the top-1 most of the time; the cross-encoder mostly
breaks ties.

#### Embedder size A/B (100q slice)

`bge-base-en-v1.5` (768 dim, ~210 MB) compared head-to-head with
`bge-small-en-v1.5` on the first 100 questions, no rerank:

| Embedder                   | R@5   | R@10   | NDCG@10 | MRR   | Median latency |
| -------------------------- | ----- | ------ | ------- | ----- | -------------- |
| `BAAI/bge-small-en-v1.5`   | 97.0% | 98.0%  | 91.6%   | 91.5% | 1 033 ms       |
| `BAAI/bge-base-en-v1.5`    | 97.0% | 100.0% | 93.7%   | 93.2% | 1 918 ms       |

`bge-base` improves NDCG/MRR by ~2 pp at roughly 2× the latency. On this
benchmark the cross-encoder reranker is a much bigger lever than going
from `bge-small` to `bge-base`, so the default keeps `bge-small-en-v1.5`
(67 MB, 384 dim) and recommends turning on reranking when ranking
quality matters.

#### Hybrid with `fastembed` on GPU vs CPU

`fastembed` autodetects CUDA when `onnxruntime-gpu` is installed
alongside the matching NVIDIA wheels. On the same RTX 4090 host, the
500q hybrid run takes ~7 minutes (median 853 ms / question); on a
32-core Threadripper CPU only, it takes 16+ hours because the ~512-char
chunks hit the bge-small attention quadratically.

To enable GPU inference on Linux + NVIDIA (CUDA 12.x, cuDNN 9.x), install
the `gpu` extra:

```bash
uv pip install 'marvin[gpu]'

# Optional: reranker uses the FP16 ONNX file (5x faster on GPU than the
# int8 CPU default).
export MARVIN_RERANK_MODEL_FILE=onnx/model_fp16.onnx
```

The `gpu` extra pulls in `onnxruntime-gpu` and the matching
`nvidia-*-cu12` wheels (~2.5 GB on disk). At first use,
:func:`marvin.gpu.bootstrap` ``ctypes``-preloads the bundled CUDA / cuDNN
libraries into the running process so onnxruntime's CUDA execution
provider can find them without `LD_LIBRARY_PATH` plumbing. The preload
is idempotent and can be disabled with `MARVIN_DISABLE_GPU_BOOTSTRAP=1`
if the host CUDA toolkit is older than the wheel-bundled libs.
`MarvinService.health()` reports `gpu_active` and `gpu_lib_count` so
you can verify a deployment is using the GPU.

For CPU-only iteration:

- Cap embedding text aggressively: `--max-embed-chars 128` is ~5× faster
  than the default 512 with little recall impact in our spot checks.
- Use `--limit N` to evaluate on a subset.
- Use `--mode bm25` for changes that don't touch dense retrieval.

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
port via `TextCrossEncoder.add_custom_model` the first time the reranker
is constructed. The default file is the int8-quantised variant
(`onnx/model_quantized.onnx`, ~570 MB) which runs well on CPU; GPU users
can switch to the FP16 variant with
`MARVIN_RERANK_MODEL_FILE=onnx/model_fp16.onnx`. Any reranker listed by
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

Performance: with `bge-reranker-v2-m3` (FP16 ONNX) on a single RTX 4090,
the full 500-question hybrid+rerank run takes ~11 minutes (median
1.18 s / query, +325 ms on top of the no-rerank hybrid). On an int8
ONNX/CPU path, budget a few minutes of wall time per 100 questions, or
pick a smaller model (e.g. `Xenova/ms-marco-MiniLM-L-6-v2`) for
interactive iteration. MCP gateway queries pay the reranker cost once
per `search()` call and only when `rerank_enabled` is set.

### Output

The CLI prints a per-question-type breakdown and writes a JSON dump:

```json
{
  "mode": "hybrid",
  "embedding_provider": "fastembed",
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "questions": 500,
  "recall_at_5": 0.980,
  "recall_at_10": 0.990,
  "ndcg_at_10": 0.919,
  "mrr": 0.927,
  "median_latency_ms": 852.6,
  "total_seconds": 455.9,
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
