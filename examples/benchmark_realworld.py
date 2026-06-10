#!/usr/bin/env python3
"""
Real-world benchmark: FAISS vs QuantaIndex on real sentence embeddings
Corpus : 1000 news articles from AG News (title + description, 4 categories)
Queries: 50 article titles used as search queries (title→article retrieval task)
Model  : sentence-transformers/all-mpnet-base-v2  (dim=768)
Metrics: Recall@K, index memory (theoretical), search latency, qualitative top-K

Memory note: theoretical calculation is used (see benchmark_faiss_vs_quanta.py for details).
  Quanta N-bit : n × dim × N/8 bytes  (via idx.memory_bytes)
  FAISS float32: n × dim × 4  bytes  (exact)

Requirements:
    pip install sentence-transformers faiss-cpu matplotlib datasets
"""

import gc
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers not installed.\n  Run: pip install sentence-transformers")
    sys.exit(1)

try:
    import faiss
except ImportError:
    print("ERROR: faiss not installed.\n  Run: pip install faiss-cpu")
    sys.exit(1)

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: datasets not installed.\n  Run: pip install datasets")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from quanta import QuantaIndex

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL      = "sentence-transformers/all-mpnet-base-v2"
DIM              = 768
N_CORPUS         = 1000   # articles indexed
N_QUERIES        = 100    # titles used as queries (from items 1000–1049)
TOP_K            = 5
QUANTA_BIT_WIDTHS = (2, 3, 4)
FAISS_NPROBE     = 10
QUALITATIVE_IDX  = [0, 12, 31]   # query indices shown in qualitative section
PLOT_OUTPUT      = "benchmark_realworld_results.png"

AG_NEWS_CATEGORIES = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}


# ── Result container ─────────────────────────────────────────────────────────
@dataclass
class BenchResult:
    system: str
    index_time_s: float
    index_mb: float        # theoretical index-data size (MB)
    avg_search_ms: float
    p50_search_ms: float
    p95_search_ms: float
    recall_at_k: float
    top_k_per_query: list[list[str]] = field(default_factory=list)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(norms, 1e-12, None)


def _ground_truth(corpus_emb: np.ndarray, query_emb: np.ndarray, k: int) -> np.ndarray:
    c = _normalize(corpus_emb)
    q = _normalize(query_emb)
    return np.argsort(-(q @ c.T), axis=1)[:, :k]


def _recall(
    retrieved: list[list[str]],
    gt: np.ndarray,
    id_to_idx: dict[str, int],
    k: int,
) -> float:
    recalls = []
    for qi, ret_ids in enumerate(retrieved):
        true_set = set(gt[qi].tolist())
        ret_set  = {id_to_idx.get(rid, -1) for rid in ret_ids[:k]}
        recalls.append(len(true_set & ret_set) / k)
    return float(np.mean(recalls))


def _latency_stats(ms: list[float]) -> tuple[float, float, float]:
    a = np.array(ms)
    return float(a.mean()), float(np.percentile(a, 50)), float(np.percentile(a, 95))


# ── Data loading ─────────────────────────────────────────────────────────────
def load_ag_news() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Returns (corpus, queries) as lists of (doc_id, text).

    Corpus : first N_CORPUS train items  → full 'title. description'
    Queries: next N_QUERIES train items  → title only  (title→article retrieval)
    """
    print("  Loading AG News dataset from HuggingFace ...", end=" ", flush=True)
    ds = load_dataset("fancyzhx/ag_news", split="train")
    print(f"Done  ({len(ds):,} train items available)")

    corpus: list[tuple[str, str]] = []
    for i in range(N_CORPUS):
        row = ds[i]
        cat = AG_NEWS_CATEGORIES.get(row["label"] + 1, "Unknown")
        doc_id = f"{cat[:3].lower()}_{i:04d}"
        text   = f"{row['text']}"
        corpus.append((doc_id, text))

    queries: list[tuple[str, str]] = []
    for i in range(N_CORPUS, N_CORPUS + N_QUERIES):
        row = ds[i]
        # Strip the description — keep only the title portion before " - " or the first sentence
        full = row["text"]
        title = full.split(" - ")[0].strip() if " - " in full else full[:120].strip()
        cat = AG_NEWS_CATEGORIES.get(row["label"] + 1, "Unknown")
        queries.append((f"q_{i:04d}_{cat[:3].lower()}", title))

    return corpus, queries


# ── Embed ─────────────────────────────────────────────────────────────────────
def embed(model: SentenceTransformer, texts: list[str], desc: str) -> np.ndarray:
    print(f"  Embedding {len(texts)} {desc} ...", end=" ", flush=True)
    t0 = time.perf_counter()
    emb = model.encode(texts, show_progress_bar=False, batch_size=64,
                       convert_to_numpy=True).astype(np.float32)
    print(f"Done  ({time.perf_counter() - t0:.1f}s)  shape={emb.shape}")
    return emb


# ── Benchmarkers ─────────────────────────────────────────────────────────────
def bench_quanta(
    corpus_emb: np.ndarray,
    query_emb: np.ndarray,
    ids: list[str],
    id_to_idx: dict[str, int],
    gt: np.ndarray,
    tmpdir: str,
    bit_width: int,
) -> BenchResult:
    t0 = time.perf_counter()
    idx = QuantaIndex(name=f"rw_{bit_width}b", dim=DIM, bit_width=bit_width, index_dir=tmpdir)
    idx.add(corpus_emb, ids)
    index_time = time.perf_counter() - t0

    index_mb = idx.memory_bytes / 1024 / 1024

    _ = idx.search(query_emb[0], k=TOP_K)   # warmup

    latencies: list[float] = []
    all_results: list[list[str]] = []
    for q in query_emb:
        t0 = time.perf_counter()
        hits = idx.search(q, k=TOP_K)
        latencies.append((time.perf_counter() - t0) * 1000)
        all_results.append([h.id for h in hits])

    result = BenchResult(
        system=f"Quanta {bit_width}-bit",
        index_time_s=index_time,
        index_mb=index_mb,
        avg_search_ms=0, p50_search_ms=0, p95_search_ms=0,
        recall_at_k=_recall(all_results, gt, id_to_idx, TOP_K),
        top_k_per_query=all_results,
    )
    result.avg_search_ms, result.p50_search_ms, result.p95_search_ms = _latency_stats(latencies)
    del idx
    gc.collect()
    return result


def bench_faiss_flat(
    corpus_emb: np.ndarray,
    query_emb: np.ndarray,
    ids: list[str],
    id_to_idx: dict[str, int],
    gt: np.ndarray,
) -> BenchResult:
    corpus_n = _normalize(corpus_emb).astype(np.float32)
    query_n  = _normalize(query_emb).astype(np.float32)

    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(DIM)
    index.add(corpus_n)
    index_time = time.perf_counter() - t0

    index_mb = index.ntotal * DIM * 4 / 1024 / 1024
    _ = index.search(query_n[:1], TOP_K)   # warmup

    latencies: list[float] = []
    all_results: list[list[str]] = []
    for q in query_n:
        t0 = time.perf_counter()
        _, I = index.search(q.reshape(1, -1), TOP_K)
        latencies.append((time.perf_counter() - t0) * 1000)
        all_results.append([ids[i] for i in I[0] if i >= 0])

    result = BenchResult(
        system="FAISS FlatIP (exact)",
        index_time_s=index_time,
        index_mb=index_mb,
        avg_search_ms=0, p50_search_ms=0, p95_search_ms=0,
        recall_at_k=_recall(all_results, gt, id_to_idx, TOP_K),
        top_k_per_query=all_results,
    )
    result.avg_search_ms, result.p50_search_ms, result.p95_search_ms = _latency_stats(latencies)
    del index
    gc.collect()
    return result


def bench_faiss_ivf(
    corpus_emb: np.ndarray,
    query_emb: np.ndarray,
    ids: list[str],
    id_to_idx: dict[str, int],
    gt: np.ndarray,
) -> BenchResult:
    corpus_n = _normalize(corpus_emb).astype(np.float32)
    query_n  = _normalize(query_emb).astype(np.float32)
    nlist = max(4, min(int(np.sqrt(len(corpus_emb))), 64))
    nprobe = min(FAISS_NPROBE, nlist)

    t0 = time.perf_counter()
    quantizer = faiss.IndexFlatIP(DIM)
    index = faiss.IndexIVFFlat(quantizer, DIM, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(corpus_n)
    index.add(corpus_n)
    index.nprobe = nprobe
    index_time = time.perf_counter() - t0

    index_mb = index.ntotal * DIM * 4 / 1024 / 1024
    _ = index.search(query_n[:1], TOP_K)   # warmup

    latencies: list[float] = []
    all_results: list[list[str]] = []
    for q in query_n:
        t0 = time.perf_counter()
        _, I = index.search(q.reshape(1, -1), TOP_K)
        latencies.append((time.perf_counter() - t0) * 1000)
        all_results.append([ids[i] for i in I[0] if i >= 0])

    result = BenchResult(
        system=f"FAISS IVFFlat (nlist={nlist}, np={nprobe})",
        index_time_s=index_time,
        index_mb=index_mb,
        avg_search_ms=0, p50_search_ms=0, p95_search_ms=0,
        recall_at_k=_recall(all_results, gt, id_to_idx, TOP_K),
        top_k_per_query=all_results,
    )
    result.avg_search_ms, result.p50_search_ms, result.p95_search_ms = _latency_stats(latencies)
    del index, quantizer
    gc.collect()
    return result


# ── Reporting ─────────────────────────────────────────────────────────────────
_W = 80


def print_results_table(results: list[BenchResult]) -> None:
    sep = "─" * _W
    print(f"\n┌{sep}┐")
    print(f"│{'REAL-WORLD BENCHMARK RESULTS':^{_W}}│")
    print(f"│{f'AG News · n={N_CORPUS} articles · {N_QUERIES} queries · k={TOP_K} · dim={DIM}':^{_W}}│")
    print(f"│{'Task: title → article retrieval  ·  Memory: theoretical index-data size':^{_W}}│")
    print(f"├{sep}┤")
    print(f"│  {'System':<32} {'Build(s)':>8}  {'Mem(MB)':>7}  {'avg ms':>7}  {'p50 ms':>7}  {'p95 ms':>7}  {'R@' + str(TOP_K):>4}  │")
    print(f"├{sep}┤")
    for r in results:
        print(
            f"│  {r.system:<32} {r.index_time_s:>8.4f}  {r.index_mb:>7.2f}  "
            f"{r.avg_search_ms:>7.4f}  {r.p50_search_ms:>7.4f}  {r.p95_search_ms:>7.4f}  "
            f"{r.recall_at_k:>4.3f}  │"
        )
    print(f"└{sep}┘")


def print_summary(results: list[BenchResult]) -> None:
    faiss_r = next((r for r in results if "FlatIP" in r.system), None)
    if not faiss_r:
        return

    print(f"\n  Memory savings vs FAISS FlatIP  (n={N_CORPUS}, dim={DIM}):")
    for r in results:
        if "Quanta" not in r.system:
            continue
        if faiss_r.index_mb > 0 and r.index_mb > 0:
            ratio = faiss_r.index_mb / r.index_mb
            print(f"    {r.system}: {ratio:.1f}x smaller  ({faiss_r.index_mb:.2f} MB → {r.index_mb:.2f} MB)")
        else:
            print(f"    {r.system}: index too small to show meaningful difference at n={N_CORPUS}")

    print(f"\n  Recall@{TOP_K} — quality impact of quantization:")
    for r in results:
        filled = int(r.recall_at_k * 30)
        bar = "█" * filled + "░" * (30 - filled)
        print(f"    {r.system:<32}  {r.recall_at_k:.4f}  {bar}")


def print_qualitative(
    results: list[BenchResult],
    queries: list[tuple[str, str]],
    id_to_text: dict[str, str],
) -> None:
    sep = "─" * _W
    print(f"\n{'═' * _W}")
    print(f"  QUALITATIVE COMPARISON  (top-{TOP_K} articles per query)")
    print(f"  Task: query = article title  →  retrieve full article from corpus")
    print(f"{'═' * _W}")

    for qi in QUALITATIVE_IDX:
        qid, qtext = queries[qi]
        print(f"\n  Query [{qid}]")
        print(f"  \"{qtext}\"")
        print(f"  {sep}")
        for r in results:
            print(f"\n  ▶ {r.system}")
            for rank, doc_id in enumerate(r.top_k_per_query[qi], 1):
                text = id_to_text.get(doc_id, "?")
                snippet = textwrap.shorten(text, width=70, placeholder="…")
                print(f"    {rank}. [{doc_id}] {snippet}")
        print()


def plot_results(results: list[BenchResult]) -> None:
    if not HAS_MATPLOTLIB:
        print("\n  [plot skipped — pip install matplotlib]")
        return

    systems = [r.system for r in results]
    palette = ["#1A237E", "#1565C0", "#42A5F5", "#B71C1C", "#E53935"]
    colors  = [palette[i % len(palette)] for i in range(len(systems))]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        f"Real-world Benchmark · AG News · n={N_CORPUS}, dim={DIM}, k={TOP_K}\n"
        f"Task: title→article retrieval  ·  Model: {EMBED_MODEL}",
        fontsize=11, fontweight="bold",
    )

    panels = [
        ([r.recall_at_k     for r in results], f"Recall@{TOP_K}",         f"Recall@{TOP_K}  ↑ better"),
        ([r.index_mb    for r in results],  "Index size (MB)",         "Index Memory  (theoretical)  ↓ better"),
        ([r.avg_search_ms   for r in results],  "Avg latency (ms)",        "Avg Search Latency  ↓ better"),
    ]

    for ax, (vals, ylabel, title) in zip(axes, panels):
        bars = ax.bar(range(len(systems)), vals, color=colors, alpha=0.87, width=0.55)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9,
            )
        ax.set_xticks(range(len(systems)))
        ax.set_xticklabels(systems, rotation=18, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        if "Recall" in ylabel:
            ax.set_ylim(0, 1.12)

    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {PLOT_OUTPUT}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    sep = "=" * _W
    print(sep)
    print("  Real-world Benchmark: FAISS vs QuantaIndex")
    print(f"  Dataset: AG News (World / Sports / Business / Sci·Tech)")
    print(f"  Model  : {EMBED_MODEL}")
    print(f"  Corpus : {N_CORPUS} articles  ·  Queries: {N_QUERIES} titles  ·  k={TOP_K}")
    print(f"  Memory : theoretical index-data size (Quanta: idx.memory_bytes · FAISS: n×dim×4 B)")
    print(sep)

    # ── Load data ─────────────────────────────────────────────────────────────
    corpus, queries = load_ag_news()

    corpus_ids   = [doc_id for doc_id, _ in corpus]
    corpus_texts = [text   for _, text   in corpus]
    query_texts  = [text   for _, text   in queries]
    id_to_text   = {doc_id: text for doc_id, text in corpus}
    id_to_idx    = {doc_id: i for i, doc_id in enumerate(corpus_ids)}

    # ── Embed ─────────────────────────────────────────────────────────────────
    print(f"\n  Loading model: {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)

    corpus_emb = embed(model, corpus_texts, "corpus articles")
    query_emb  = embed(model, query_texts,  "query titles")

    # ── Ground truth ──────────────────────────────────────────────────────────
    print(f"  Computing ground truth (brute-force cosine) ...", end=" ", flush=True)
    gt = _ground_truth(corpus_emb, query_emb, TOP_K)
    print("Done")

    # ── Benchmarks ────────────────────────────────────────────────────────────
    all_results: list[BenchResult] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for bw in QUANTA_BIT_WIDTHS:
            print(f"\n  [Quanta {bw}-bit] ...", end=" ", flush=True)
            r = bench_quanta(corpus_emb, query_emb, corpus_ids, id_to_idx, gt, tmpdir, bw)
            all_results.append(r)
            print(f"R@{TOP_K}={r.recall_at_k:.4f}  p50={r.p50_search_ms:.4f}ms  mem={r.index_mb:.2f}MB")

    print(f"\n  [FAISS FlatIP] ...", end=" ", flush=True)
    r = bench_faiss_flat(corpus_emb, query_emb, corpus_ids, id_to_idx, gt)
    all_results.append(r)
    print(f"R@{TOP_K}={r.recall_at_k:.4f}  p50={r.p50_search_ms:.4f}ms  mem={r.index_mb:.2f}MB")

    print(f"\n  [FAISS IVFFlat] ...", end=" ", flush=True)
    r = bench_faiss_ivf(corpus_emb, query_emb, corpus_ids, id_to_idx, gt)
    all_results.append(r)
    print(f"R@{TOP_K}={r.recall_at_k:.4f}  p50={r.p50_search_ms:.4f}ms  mem={r.index_mb:.2f}MB")

    # ── Output ────────────────────────────────────────────────────────────────
    print_results_table(all_results)
    print_summary(all_results)
    print_qualitative(all_results, queries, id_to_text)
    plot_results(all_results)


if __name__ == "__main__":
    main()
