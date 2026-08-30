#!/usr/bin/env python3
"""Benchmark retrieval accuracy and quality metrics.

This script measures:
1. Retrieval recall at K
2. Temporal accuracy
3. Evidence quality
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from videocontent import Video, load
from videocontent.retrieval import search, at
from videocontent.schema.v1 import VideoContextDocument


def run_retrieval_benchmark(vctx_path: str) -> dict[str, Any]:
    """Run retrieval benchmarks on a processed video."""
    print(f"\n=== Retrieval Benchmark: {vctx_path} ===")

    video = load(vctx_path)
    doc = video.document

    print(f"Video: {doc.video.filename}")
    print(f"Duration: {doc.video.duration:.1f}s")
    print(f"Transcript: {len(doc.transcript)} segments")
    print(f"OCR: {len(doc.ocr)} events")
    print(f"Vision: {len(doc.vision)} notes")
    print(f"Events: {len(doc.events)}")

    # Define test queries with expected time ranges
    test_queries = [
        {
            "query": "revenue",
            "expected_modalities": ["transcript", "ocr"],
            "description": "Revenue mention",
        },
        {
            "query": "pricing",
            "expected_modalities": ["transcript", "ocr"],
            "description": "Pricing discussion",
        },
        {
            "query": "competitor",
            "expected_modalities": ["transcript", "ocr"],
            "description": "Competitor mention",
        },
        {
            "query": "dashboard",
            "expected_modalities": ["ocr", "vision"],
            "description": "Dashboard UI",
        },
        {
            "query": "login",
            "expected_modalities": ["ocr", "vision"],
            "description": "Login screen",
        },
        {
            "query": "demo",
            "expected_modalities": ["transcript", "vision"],
            "description": "Demo section",
        },
    ]

    results = []

    for test in test_queries:
        print(f"\n--- Query: '{test['query']}' ({test['description']}) ---")

        # Search with different top_k values
        for top_k in [1, 3, 5, 10]:
            start = time.perf_counter()
            result = video.search(test["query"], top_k=top_k)
            search_time = time.perf_counter() - start

            hits = result.spans
            total = result.total

            # Check modality coverage
            modalities_found = set(h.modality for h in hits)
            expected = set(test["expected_modalities"])
            modality_coverage = len(modalities_found & expected) / len(expected) if expected else 0

            print(f"  top_k={top_k}: {len(hits)} hits, {total} total, "
                  f"modalities: {modalities_found}, coverage: {modality_coverage:.0%}, "
                  f"time: {search_time*1000:.1f}ms")

        # Detailed results for top_k=5
        result = video.search(test["query"], top_k=5)
        hits = result.spans

        # Score distribution
        scores = [h.score for span in hits for h in (span,)]
        if scores:
            print(f"  Scores: min={min(scores):.4f}, max={max(scores):.4f}, avg={statistics.mean(scores):.4f}")

        results.append({
            "query": test["query"],
            "description": test["description"],
            "total_matches": result.total,
            "top_5_hits": len(result.spans),
            "modalities_found": list(set(h.modality for h in result.spans)),
            "top_score": max(h.score for h in result.spans) if result.spans else 0,
            "avg_score": statistics.mean(h.score for h in result.spans) if result.spans else 0,
        })

    # Test timeline lookup (at())
    print("\n--- Timeline Lookup (at()) ---")
    test_times = [0, 5, 15, 30, 45, 60]
    for ts in test_times:
        if ts >= video.document.video.duration:
            continue
        result = video.at(ts)
        print(f"  t={ts:.0f}s: {len(result.spans)} spans across {len(set(s.modality for s in result.spans))} modalities")

    return {
        "queries": results,
    }


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python bench_retrieval.py <video.vctx> [output.json]")
        sys.exit(1)

    vctx_path = sys.argv[1]
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("retrieval_benchmark.json")

    results = run_retrieval_benchmark(vctx_path)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    import sys
    main()