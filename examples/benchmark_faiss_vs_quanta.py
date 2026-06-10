#!/usr/bin/env python3
"""
Benchmark: FAISS vs QuantaIndex (2-bit, 3-bit, 4-bit)
Metrics: Recall@K, index memory (theoretical), build time, search latency.

Memory note: theoretical calculation is used because RSS-delta is unreliable
for sequential in-process benchmarks — the OS reuses already-allocated pages
and the delta collapses to 0 after the first run.
  Quanta N-bit : n × dim × N/8 bytes  (via idx.memory_bytes)
  FAISS float32: n × dim × 4  bytes  (exact, IndexFlatIP stores float32)

Requirements:
    pip install faiss-cpu matplotlib
"""

import gc
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import faiss
except ImportError:
    print("ERROR: faiss not installed.\n  Run: pip install faiss-cpu")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from quanta import QuantaIndex

# ── Configuration ────────────────────────────────────────────────────────────
DIM        = 768
TOP_K      = 10
NUM_QUERIES = 200
FAISS_NPROBE = 10
SIZES      = [1_000, 10_000, 50_000]
RANDOM_SEED = 42
PLOT_OUTPUT = "benchmark_results.png"


# ── Result container ─────────────────────────────────────────────────────────
@dataclass
class BenchResult:
    system: str
    n_vectors: int
    index_time_s: float
    index_mb: float        # theoretical index-data size (MB)
    avg_search_ms: float
    p50_search_ms: float
    p95_search_ms: float
    recall_at_k: float


# ── Helpers ──────────────────────────────────────────────────────────────────
def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(norms, 1e-12, None)


def _ground_truth(corpus: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    c = _normalize(corpus)
    q = _normalize(queries)
    return np.argsort(-(q @ c.T), axis=1)[:, :k]


def _recall(
    retrieved: list[list[str]],
    gt: np.ndarray,
    id_to_idx: dict[str, int],
    k: int,
) -> float:
    recalls = []
    for q_idx, ret_ids in enumerate(retrieved):
        true_set = set(gt[q_idx].tolist())
        ret_set  = {id_to_idx.get(rid, -1) for rid in ret_ids[:k]}
        recalls.append(len(true_set & ret_set) / k)
    return float(np.mean(recalls))


def _latency_stats(ms: list[float]) -> tuple[float, float, float]:
    a = np.array(ms)
    return float(a.mean()), float(np.percentile(a, 50)), float(np.percentile(a, 95))


# ── Benchmarkers ─────────────────────────────────────────────────────────────
def bench_quanta(
    corpus: np.ndarray,
    queries: np.ndarray,
    ids: list[str],
    id_to_idx: dict[str, int],
    gt: np.ndarray,
    tmpdir: str,
    bit_width: int,
) -> BenchResult:
    t0 = time.perf_counter()
    idx = QuantaIndex(name=f"bench_{bit_width}b", dim=corpus.shape[1], bit_width=bit_width, index_dir=tmpdir)
    idx.add(corpus, ids)
    index_time = time.perf_counter() - t0

    # Exact theoretical size: n × dim × bits/8 + ID-mapping overhead
    index_mb = idx.memory_bytes / 1024 / 1024

    _ = idx.search(queries[0], k=TOP_K)  # warmup

    latencies: list[float] = []
    all_results: list[list[str]] = []
    for q in queries:
        t0 = time.perf_counter()
        hits = idx.search(q, k=TOP_K)
        latencies.append((time.perf_counter() - t0) * 1000)
        all_results.append([h.id for h in hits])

    result = BenchResult(
        system=f"Quanta {bit_width}-bit",
        n_vectors=len(ids),
        index_time_s=index_time,
        index_mb=index_mb,
        avg_search_ms=0, p50_search_ms=0, p95_search_ms=0,
        recall_at_k=_recall(all_results, gt, id_to_idx, TOP_K),
    )
    result.avg_search_ms, result.p50_search_ms, result.p95_search_ms = _latency_stats(latencies)
    del idx
    gc.collect()
    return result


def bench_faiss_flat(
    corpus: np.ndarray,
    queries: np.ndarray,
    ids: list[str],
    id_to_idx: dict[str, int],
    gt: np.ndarray,
) -> BenchResult:
    corpus_n  = _normalize(corpus).astype(np.float32)
    queries_n = _normalize(queries).astype(np.float32)
    dim = corpus.shape[1]

    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(dim)
    index.add(corpus_n)
    index_time = time.perf_counter() - t0

    # Exact: IndexFlatIP stores n × dim float32 values
    index_mb = index.ntotal * dim * 4 / 1024 / 1024

    _ = index.search(queries_n[:1], TOP_K)  # warmup

    latencies: list[float] = []
    all_results: list[list[str]] = []
    for q in queries_n:
        t0 = time.perf_counter()
        _, I = index.search(q.reshape(1, -1), TOP_K)
        latencies.append((time.perf_counter() - t0) * 1000)
        all_results.append([ids[i] for i in I[0] if i >= 0])

    result = BenchResult(
        system="FAISS FlatIP (exact)",
        n_vectors=len(ids),
        index_time_s=index_time,
        index_mb=index_mb,
        avg_search_ms=0, p50_search_ms=0, p95_search_ms=0,
        recall_at_k=_recall(all_results, gt, id_to_idx, TOP_K),
    )
    result.avg_search_ms, result.p50_search_ms, result.p95_search_ms = _latency_stats(latencies)
    del index
    gc.collect()
    return result


def bench_faiss_ivf(
    corpus: np.ndarray,
    queries: np.ndarray,
    ids: list[str],
    id_to_idx: dict[str, int],
    gt: np.ndarray,
) -> Optional[BenchResult]:
    if len(corpus) < 1000:
        return None

    corpus_n  = _normalize(corpus).astype(np.float32)
    queries_n = _normalize(queries).astype(np.float32)
    dim   = corpus.shape[1]
    nlist = max(4, min(int(np.sqrt(len(corpus))), 256))
    nprobe = min(FAISS_NPROBE, nlist)

    t0 = time.perf_counter()
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(corpus_n)
    index.add(corpus_n)
    index.nprobe = nprobe
    index_time = time.perf_counter() - t0

    # Vectors stored as float32; centroids negligible vs ntotal
    index_mb = index.ntotal * dim * 4 / 1024 / 1024

    _ = index.search(queries_n[:1], TOP_K)  # warmup

    latencies: list[float] = []
    all_results: list[list[str]] = []
    for q in queries_n:
        t0 = time.perf_counter()
        _, I = index.search(q.reshape(1, -1), TOP_K)
        latencies.append((time.perf_counter() - t0) * 1000)
        all_results.append([ids[i] for i in I[0] if i >= 0])

    result = BenchResult(
        system=f"FAISS IVFFlat (nlist={nlist}, np={nprobe})",
        n_vectors=len(ids),
        index_time_s=index_time,
        index_mb=index_mb,
        avg_search_ms=0, p50_search_ms=0, p95_search_ms=0,
        recall_at_k=_recall(all_results, gt, id_to_idx, TOP_K),
    )
    result.avg_search_ms, result.p50_search_ms, result.p95_search_ms = _latency_stats(latencies)
    del index, quantizer
    gc.collect()
    return result


# ── Reporting ─────────────────────────────────────────────────────────────────
_W = 88


def print_results_table(results: list[BenchResult]) -> None:
    sep = "─" * _W
    print(f"\n┌{sep}┐")
    print(f"│{'BENCHMARK RESULTS':^{_W}}│")
    print(f"│{'Memory = theoretical index-data size  (Quanta: idx.memory_bytes · FAISS: n×dim×4 B)':^{_W}}│")
    print(f"├{sep}┤")
    print(f"│  {'System':<34} {'N':>7}  {'Build(s)':>8}  {'Mem(MB)':>7}  {'p50ms':>6}  {'p95ms':>6}  {'R@10':>5}  │")
    print(f"├{sep}┤")

    prev_n = None
    for r in results:
        if prev_n is not None and r.n_vectors != prev_n:
            print(f"├{sep}┤")
        prev_n = r.n_vectors
        print(
            f"│  {r.system[:34]:<34} {r.n_vectors:>7,}  {r.index_time_s:>8.3f}  "
            f"{r.index_mb:>7.1f}  {r.p50_search_ms:>6.3f}  {r.p95_search_ms:>6.3f}  "
            f"{r.recall_at_k:>5.3f}  │"
        )

    print(f"└{sep}┘")


def print_summary(results: list[BenchResult]) -> None:
    sizes = sorted({r.n_vectors for r in results})

    print("\n  Memory savings vs FAISS FlatIP (theoretical):")
    for n in sizes:
        faiss_r = next((r for r in results if r.n_vectors == n and "FlatIP" in r.system), None)
        if not faiss_r:
            continue
        for tag in ("2-bit", "3-bit", "4-bit"):
            q = next((r for r in results if r.n_vectors == n and tag in r.system), None)
            if q:
                ratio = faiss_r.index_mb / q.index_mb
                print(
                    f"    N={n:>7,}  Quanta {tag}: {ratio:.1f}x smaller  "
                    f"({faiss_r.index_mb:.1f} MB → {q.index_mb:.1f} MB)"
                )

    print(f"\n  Recall@{TOP_K} — accuracy vs quantization level:")
    for n in sizes:
        for r in [r for r in results if r.n_vectors == n]:
            bar = "█" * int(r.recall_at_k * 20) + "░" * (20 - int(r.recall_at_k * 20))
            print(f"    N={n:>7,}  {r.system:<36}  {r.recall_at_k:.4f}  {bar}")


def plot_results(results: list[BenchResult]) -> None:
    if not HAS_MATPLOTLIB:
        print("\n  [plot skipped — pip install matplotlib]")
        return

    sizes   = sorted({r.n_vectors for r in results})
    systems = list(dict.fromkeys(r.system for r in results))
    palette = ["#1565C0", "#42A5F5", "#90CAF9", "#C62828", "#EF9A9A"]
    colors  = {s: palette[i % len(palette)] for i, s in enumerate(systems)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"FAISS vs QuantaIndex  (dim={DIM}, top_k={TOP_K}, {NUM_QUERIES} queries)\n"
        f"Memory = theoretical index-data size",
        fontsize=13, fontweight="bold",
    )

    metrics = [
        ("p50_search_ms", "Search latency p50 (ms)", f"Search Latency p50  [Recall@{TOP_K} shown]"),
        ("index_mb",      "Index size (MB)",          "Index Memory  (theoretical)"),
        ("index_time_s",  "Build time (s)",            "Index Build Time"),
    ]

    x     = np.arange(len(sizes))
    width = 0.75 / max(len(systems), 1)

    for ax_idx, (metric, ylabel, title) in enumerate(metrics):
        ax = axes[ax_idx]
        for i, sys_name in enumerate(systems):
            vals = []
            for n in sizes:
                r = next((r for r in results if r.n_vectors == n and r.system == sys_name), None)
                vals.append(getattr(r, metric, 0.0) if r else 0.0)

            xs   = x + (i - len(systems) / 2 + 0.5) * width
            bars = ax.bar(xs, vals, width * 0.88, label=sys_name, color=colors[sys_name], alpha=0.87)

            if ax_idx == 0:
                for bar, n in zip(bars, sizes):
                    r = next((r for r in results if r.n_vectors == n and r.system == sys_name), None)
                    if r:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() * 1.02,
                            f"R={r.recall_at_k:.2f}",
                            ha="center", va="bottom", fontsize=7, rotation=45, color=colors[sys_name],
                        )

        ax.set_xticks(x)
        ax.set_xticklabels([f"{n:,}" for n in sizes], fontsize=9)
        ax.set_xlabel("Corpus size (vectors)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {PLOT_OUTPUT}")


# ── Per-size runner ──────────────────────────────────────────────────────────
def run_size(n: int, tmpdir: str) -> list[BenchResult]:
    sep = "─" * _W
    print(f"\n{sep}")
    print(f"  N={n:,}  dim={DIM}  k={TOP_K}  queries={NUM_QUERIES}")
    print(sep)

    rng     = np.random.default_rng(RANDOM_SEED)
    corpus  = rng.standard_normal((n, DIM)).astype(np.float32)
    queries = rng.standard_normal((NUM_QUERIES, DIM)).astype(np.float32)
    ids       = [f"doc_{i}" for i in range(n)]
    id_to_idx = {v: i for i, v in enumerate(ids)}

    print("  Ground truth (brute-force cosine)... ", end="", flush=True)
    gt = _ground_truth(corpus, queries, TOP_K)
    print("done")

    results: list[BenchResult] = []

    for bw in (2, 3, 4):
        print(f"  [Quanta {bw}-bit]        ...", end=" ", flush=True)
        r = bench_quanta(corpus, queries, ids, id_to_idx, gt, tmpdir, bit_width=bw)
        results.append(r)
        print(f"R@{TOP_K}={r.recall_at_k:.4f}  p50={r.p50_search_ms:.3f}ms  mem={r.index_mb:.1f}MB")

    print(f"  [FAISS FlatIP (exact)] ...", end=" ", flush=True)
    r = bench_faiss_flat(corpus, queries, ids, id_to_idx, gt)
    results.append(r)
    print(f"R@{TOP_K}={r.recall_at_k:.4f}  p50={r.p50_search_ms:.3f}ms  mem={r.index_mb:.1f}MB")

    print(f"  [FAISS IVFFlat (approx)]...", end=" ", flush=True)
    r = bench_faiss_ivf(corpus, queries, ids, id_to_idx, gt)
    if r:
        results.append(r)
        print(f"R@{TOP_K}={r.recall_at_k:.4f}  p50={r.p50_search_ms:.3f}ms  mem={r.index_mb:.1f}MB")
    else:
        print("skipped (N < 1000)")

    return results


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * _W)
    print("  Quanta vs FAISS Benchmark")
    print(f"  dim={DIM}  k={TOP_K}  queries={NUM_QUERIES}  sizes={SIZES}")
    print(f"  Quanta bit widths: 2, 3, 4  (turbovec supports 2/3/4 only)")
    print(f"  Memory: theoretical index-data size (not RSS — see module docstring)")
    print("=" * _W)

    all_results: list[BenchResult] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for n in SIZES:
            all_results.extend(run_size(n, tmpdir))

    print_results_table(all_results)
    print_summary(all_results)
    plot_results(all_results)


if __name__ == "__main__":
    main()
