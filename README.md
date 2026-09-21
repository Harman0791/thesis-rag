# thesis-rag

**Optimizing Retrieval-Augmented Generation for Low-Resource Settings: A Comparative Study of Retrieval and Chunking Strategies**

Master's thesis project (Gisma University of Applied Sciences, MSc Data Science, AI & Digital Business — M598R Master Thesis) investigating how chunking strategy, retrieval method, and generator model size interact in a fully CPU-only RAG pipeline.

## Overview

RAG pipelines are usually benchmarked on GPU infrastructure. This project instead builds and evaluates a modular RAG system entirely on a standard laptop (no GPU) to answer two questions:

1. How do sparse, dense, and hybrid retrieval techniques behave in a resource-constrained environment?
2. Which retrieval technique, paired with which generator, gives the best end-to-end results?

The system is tested across:

- **Retrieval strategies (3):** BM25 (sparse/lexical), Dense (`intfloat/e5-small-v2` embeddings + FAISS HNSW), Hybrid (weighted fusion of both, α = 0.5)
- **Chunk sizes (3):** 128 / 256 / 512 tokens
- **Overlap ratios (3):** 0% / 12.5% / 25%
- **Generator models (3):** LLAMA, Qwen-3B, Qwen-7B (GGUF, quantized, CPU-only)

Evaluation is run on the [TriviaQA](https://arxiv.org/abs/1705.03551) validation split.

## Key findings

- **Chunk size 256 is the sweet spot.** Larger chunks (512) improve raw retrieval metrics (Recall, MRR) but generator quality (EM/F1) peaks at moderate chunk size — beyond that, extra context introduces noise that hurts answer precision more than it helps recall.
- **Overlap mostly isn't worth it.** On the retrieval side, 0% overlap performs as well as or better than 12.5–25%, while overlap consistently increases latency and memory. It helps a *little* on the generation side in some configurations, but the retrieval-side story doesn't carry over cleanly to the generator.
- **BM25 is the strongest all-round retriever for CPU-only deployment.** It's dramatically faster than Dense or Hybrid (which pay real costs for embedding/similarity computation) and, at larger chunk sizes, closes most of the quality gap with Dense retrieval.
- **Hybrid retrieval underperforms.** It inherits the computational overhead of both sparse and dense methods without a clear quality payoff — the added complexity isn't justified in a resource-constrained setting.
- **Bigger generator ≠ better.** Qwen-3B matches or beats Qwen-7B in most configurations. On a CPU, the larger model's extra capacity is offset by higher latency/memory cost and by "context dilution" from longer retrieved passages.
- **Best overall configuration:** BM25 retrieval + chunk size 256 + Qwen-3B generator — achieves the highest Exact Match / F1 scores of any tested combination while remaining the cheapest to run.

## Architecture

```
TriviaQA
   │
   ▼
Data Preparation ── HTML/punctuation cleanup, sliding-window chunking
   │                (configurable chunk size + overlap), saved as Parquet
   ▼
Embedding Generation ── intfloat/e5-small-v2 (384-dim), batched, memory-mapped
   │
   ▼
Retrieval Mechanisms ── Sparse (BM25) | Dense (FAISS HNSW) | Hybrid (weighted fusion)
   │                       Top-k passages
   ▼
Generator ── Format retrieved passages into a strict, context-grounded prompt
   │            (LLAMA / Qwen-3B / Qwen-7B, GGUF quantized)
   ▼
Metrics Analysis ── Retrieval: Recall, MRR, latency, memory
                     Generation: Exact Match (EM), token-level F1
```

- **Retrieval layer:** three interchangeable retriever classes exposing the same interface, so techniques can be swapped without touching the rest of the pipeline.
  - *Sparse (BM25):* CountVectorizer-based term-document matrix, classic BM25 scoring — fast, no embeddings required.
  - *Dense:* passages and queries encoded with `e5-small-v2`, indexed with FAISS HNSW, cosine similarity via inner product on L2-normalized vectors.
  - *Hybrid:* dense candidates (FAISS) + lexical candidates (TF-IDF dot product), each min-max normalized, combined via `S_hybrid = α·S_dense + (1-α)·S_lex` (α = 0.5).
- **Generation layer:** builds a strict, context-grounded prompt (system instruction + top-ranked passages within token budget + question), truncating/falling back gracefully if context doesn't fit. Output is post-processed (strip punctuation/prefixes/quotes) before scoring.
- **Evaluation:**
  - Retrieval: **Recall** (proportion of relevant docs retrieved), **MRR** (how early the first relevant doc ranks), plus latency and memory.
  - Generation: **Exact Match** (strict, normalized string match against gold answers) and **token-level F1** (precision/recall overlap, more forgiving of paraphrasing).

## Repo structure

```
data/                    # TriviaQA subset, chunked and normalized
indexes/                  # FAISS HNSW indexes per embedder/chunk/overlap config
notebooks/                # Master notebook + pipeline stages (data prep, embedding,
                           # retrieval, generation, evaluation)
results/                   # Retrieval + generator output tables per configuration
requirements.txt          # Python dependencies
```

## Tech stack

- Python 3.11, `sentence-transformers` (`intfloat/e5-small-v2`), FAISS (HNSW index)
- BM25 via `CountVectorizer` + custom scoring; TF-IDF via `TfidfVectorizer`
- GGUF-quantized generator models (LLAMA, Qwen-3B, Qwen-7B) — CPU-only inference, no GPU required
- Jupyter notebooks for pipeline orchestration and evaluation

## Setup

```bash
git clone https://github.com/Harman0791/thesis-rag.git
cd thesis-rag
pip install -r requirements.txt
```

Run the notebooks in `notebooks/` in order (data prep → embedding/indexing → retrieval → generation → evaluation). A shared config file at the top controls chunk size, overlap ratio, retriever choice, and generator model for each experiment run.

**Environment used for experiments:** Windows 11, 16 GB RAM, Intel Core i5-1340P — CPU only, no GPU.

## Author

Harmanpreet Kaur — MSc Data Science, AI & Digital Business, Gisma University of Applied Sciences (M598R Master Thesis, January 2026)
Supervised by Prof. Dr. Mohammad Mahdavi
