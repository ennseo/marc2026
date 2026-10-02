"""Latency/memory benchmark for the NLU parser -- separate from accuracy testing.

Measures what test_parser.py doesn't: model load time, per-sentence latency
distribution, and memory footprint, specifically forced to CPU-only (n_gpu_layers=0,
the default -- and the only mode available in the submission Docker image, since
Dockerfile installs plain `pip install llama-cpp-python` with no CUDA build flags).

Usage: python3 benchmark_latency.py
"""

import os
import resource
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # parser.py etc. live in nlu/

from parser import NLUParser
from test_data import KNOWN_EXAMPLES
from paraphrases import PARAPHRASE_EXAMPLES
from held_out_sentences import HELD_OUT_EXAMPLES

ALL_SENTENCES = (
    [c["sentence"] for c in KNOWN_EXAMPLES]
    + [c["sentence"] for c in PARAPHRASE_EXAMPLES]
    + [c["sentence"] for c in HELD_OUT_EXAMPLES]
)


def rss_mb() -> float:
    # ru_maxrss is KB on Linux, bytes on macOS
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 if os.uname().sysname == "Linux" else r / (1024 * 1024)


if __name__ == "__main__":
    print(f"CPU count (os.cpu_count()): {os.cpu_count()}")
    print(f"RSS before load: {rss_mb():.0f} MB")

    t0 = time.perf_counter()
    parser = NLUParser(n_gpu_layers=0)  # force CPU -- matches the Docker submission (no CUDA build)
    load_s = time.perf_counter() - t0
    print(f"Model load time: {load_s:.2f}s")
    print(f"RSS after load: {rss_mb():.0f} MB")

    # warm up (first call sometimes has extra one-time init cost)
    parser.parse(ALL_SENTENCES[0])

    latencies = []
    for s in ALL_SENTENCES:
        t0 = time.perf_counter()
        parser.parse(s)
        latencies.append(time.perf_counter() - t0)

    latencies.sort()
    n = len(latencies)
    print(f"\nRSS after {n} calls: {rss_mb():.0f} MB")
    print(f"\n--- per-sentence latency over {n} calls (CPU, n_gpu_layers=0) ---")
    print(f"  min:    {latencies[0]:.2f}s")
    print(f"  median: {latencies[n//2]:.2f}s")
    print(f"  mean:   {sum(latencies)/n:.2f}s")
    print(f"  p95:    {latencies[int(n*0.95)]:.2f}s")
    print(f"  max:    {latencies[-1]:.2f}s")
