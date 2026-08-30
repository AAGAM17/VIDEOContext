#!/usr/bin/env python3
"""Benchmark context compression and quality metrics.

This script measures:
1. Context compression ratio (raw vs optimized)
2. Token reduction
3. Evidence recall
4. Profile generation time
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from videocontent import Video, load
from videocontent.config import ProcessingConfig
from videocontent.profiles import get_profile_builder, ProfileContext
from videocontent.routing import classify_task, select_context, ContextBudget, pack_context
from videocontent.schema.v1 import VideoContextDocument


def estimate_raw_tokens(doc: VideoContextDocument) -> int:
    """Estimate tokens if we sent all raw evidence to an LLM."""
    tokens = 0
    # Transcript
    for utt in doc.transcript:
        tokens += len(utt.text) // 4
    # OCR
    for ocr in doc.ocr:
        tokens += len(ocr.text) // 4
    # Vision
    for vision in doc.vision:
        tokens += len(vision.description) // 4
    # Events
    for evt in doc.events:
        tokens += len(evt.description or "") // 4
    return tokens


def run_compression_benchmark(video_path: str, output_dir: Path) -> dict[str, Any]:
    """Run compression benchmark on a video."""
    print(f"\n=== Benchmarking: {video_path} ===")

    # Load or process video
    vctx_path = Path(video_path).with_suffix(".vctx")
    if vctx_path.exists():
        video = load(vctx_path)
    else:
        print("Processing video...")
        video = Video(video_path, config=ProcessingConfig())
        video.process()

    doc = video.document
    print(f"Video duration: {doc.video.duration:.1f}s")
    print(f"Frames: {len(doc.frames)}")
    print(f"Transcript segments: {len(doc.transcript)}")
    print(f"OCR events: {len(doc.ocr)}")
    print(f"Vision notes: {len(doc.vision)}")
    print(f"Events: {len(doc.events)}")

    # Estimate raw tokens
    raw_tokens = estimate_raw_tokens(doc)
    print(f"\nRaw evidence tokens (est.): {raw_tokens}")

    # Test different task types
    task_types = [
        ("factual_retrieval", "What was the revenue mentioned?"),
        ("global_understanding", "What is this video about?"),
        ("ui_recreation", "Recreate the website design language"),
        ("application_understanding", "How does this application work?"),
        ("interaction_analysis", "Describe the animations and transitions"),
    ]

    results = []

    for task_type, query in task_types:
        print(f"\n--- Task: {task_type} ---")
        print(f"Query: {query}")

        # Classify
        from videocontent.routing import TaskType
        classification = type('obj', (object,), {
            'task_type': getattr(__import__('videocontent.routing', fromlist=['TaskType']).TaskType, task_type.upper()),
            'confidence': 0.8,
            'suggested_profiles': [],
            'requires_evidence': True,
            'requires_frames': task_type in ['ui_recreation', 'interaction_analysis', 'global_understanding'],
            'requires_global': task_type in ['global_understanding', 'ui_recreation'],
        })()

        # Select context
        budget = ContextBudget(max_tokens=4000)
        start = time.perf_counter()
        selection = select_context(video.document, classification, budget, query)
        selection_time = time.perf_counter() - start

        # Pack context
        packed = pack_context(selection, task_type, query)
        token_estimate = len(packed) // 4

        compression_ratio = (1 - token_estimate / raw_tokens) * 100 if raw_tokens > 0 else 0

        print(f"  Selection time: {selection_time*1000:.1f}ms")
        print(f"  Packed tokens: {token_estimate}")
        print(f"  Compression: {compression_ratio:.1f}%")
        print(f"  Profiles: {list(selection['profiles'].keys())}")
        print(f"  Evidence spans: {len(selection['evidence'])}")
        print(f"  Frames: {len(selection['frames'])}")

        results.append({
            "task_type": task_type,
            "query": query,
            "raw_tokens": raw_tokens,
            "optimized_tokens": token_estimate,
            "compression_ratio_pct": compression_ratio,
            "selection_time_ms": selection_time * 1000,
            "profiles": list(selection['profiles'].keys()),
            "evidence_count": len(selection['evidence']),
            "frame_count": len(selection['frames']),
        })

    # Profile generation benchmark
    print("\n--- Profile Generation ---")
    from videocontent.profiles import get_profile_builder, ProfileContext

    profile_results = []
    profile_names = ["ui_design", "application", "product_demo", "tutorial"]
    ctx = ProfileContext(doc=doc, config={})

    for pname in ["ui_design", "application", "product_demo", "tutorial"]:
        try:
            builder_cls = get_profile_builder(pname)
            builder = builder_cls()
            profile_ctx = __import__('videocontent.profiles.base', fromlist=['ProfileContext']).ProfileContext(doc=doc, config={})
            if builder.supports(profile_ctx):
                start = time.perf_counter()
                profile = builder.build(profile_ctx)
                gen_time = time.perf_counter() - start
                print(f"  {pname}: {gen_time*1000:.1f}ms")
                profile_results.append({
                    "profile": pname,
                    "generation_time_ms": gen_time * 1000,
                    "supported": True,
                })
            else:
                print(f"  {pname}: Not supported for this video")
                profile_results.append({
                    "profile": pname,
                    "generation_time_ms": 0,
                    "supported": False,
                })
        except Exception as e:
            print(f"  {pname}: Error - {e}")
            profile_results.append({
                "profile": pname,
                "generation_time_ms": 0,
                "supported": False,
                "error": str(e),
            })

    # Summary
    avg_compression = statistics.mean(r["compression_ratio_pct"] for r in results) if results else 0
    avg_selection = statistics.mean(r["selection_time_ms"] for r in results) if results else 0

    print(f"\n=== SUMMARY ===")
    print(f"Raw tokens: {raw_tokens}")
    print(f"Average compression: {avg_compression:.1f}%")
    print(f"Average selection time: {avg_selection:.1f}ms")

    return {
        "video": video_path,
        "video_duration": doc.video.duration,
        "raw_tokens": raw_tokens,
        "tasks": results,
        "profiles": profile_results,
        "summary": {
            "avg_compression_pct": avg_compression,
            "avg_selection_time_ms": avg_selection,
        },
    }


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python bench_context.py <video_path> [output.json]")
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("benchmark_results.json")

    results = run_compression_benchmark(video_path, Path("."))

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()