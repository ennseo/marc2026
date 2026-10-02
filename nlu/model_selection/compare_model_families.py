"""Same prompt, same test sets, different model family: Qwen2.5 vs Llama-3.2 vs
Gemma-2 vs SmolLM2 vs Phi-3.5.

Purpose: answer "why Qwen and not some other LLM family?" with actual data instead
of a general reputation-based justification. Compares our chosen Qwen2.5-1.5B against
the most popular alternative small-LLM families, roughly bracketing its size (1-4B) --
all via llama-cpp-python's create_chat_completion, which applies each model's own
embedded chat template, so each model gets a fair prompt in its native format.

Usage: python3 compare_model_families.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # parser.py etc. live in nlu/

from parser import NLUParser
from test_data import KNOWN_EXAMPLES
from paraphrases import PARAPHRASE_EXAMPLES
from held_out_sentences import HELD_OUT_EXAMPLES

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
_MODELS = {
    "Qwen2.5-1.5B": os.path.join(_MODELS_DIR, "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"),
    "Llama-3.2-1B": os.path.join(_MODELS_DIR, "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    "Llama-3.2-3B": os.path.join(_MODELS_DIR, "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
    "Gemma-2-2B": os.path.join(_MODELS_DIR, "gemma-2-2b-it-Q4_K_M.gguf"),
    "SmolLM2-1.7B": os.path.join(_MODELS_DIR, "SmolLM2-1.7B-Instruct-Q4_K_M.gguf"),
    "Phi-3.5-mini": os.path.join(_MODELS_DIR, "Phi-3.5-mini-instruct-Q4_K_M.gguf"),
}

ALL_SETS = [
    ("Known 39", KNOWN_EXAMPLES, ""),
    ("Paraphrase 66", PARAPHRASE_EXAMPLES, ""),
    ("Held-out 24", HELD_OUT_EXAMPLES, "expected_"),
]


def score_and_time(parser, cases, prefix):
    hits = {"target_type": 0, "landmark": 0, "relation": 0, "situation": 0}
    latencies = []
    for case in cases:
        expected = {k: case.get(f"{prefix}{k}") for k in hits}
        t0 = time.perf_counter()
        got = parser.parse(case["sentence"])
        latencies.append(time.perf_counter() - t0)
        for k in hits:
            if str(got.get(k)).lower() == str(expected[k]).lower():
                hits[k] += 1
    return hits, latencies


if __name__ == "__main__":
    results = {}
    for label, path in _MODELS.items():
        print(f"\nLoading {label} ({path})...")
        t0 = time.perf_counter()
        parser = NLUParser(model_path=path, n_gpu_layers=0)  # CPU, matches Docker submission
        load_s = time.perf_counter() - t0
        print(f"  load time: {load_s:.2f}s")

        all_hits = {"target_type": 0, "landmark": 0, "relation": 0, "situation": 0}
        all_latencies = []
        total_n = 0
        for name, cases, prefix in ALL_SETS:
            hits, latencies = score_and_time(parser, cases, prefix)
            n = len(cases)
            print(f"  [{name}] " + " ".join(f"{k}={100*v/n:.0f}%" for k, v in hits.items()))
            for k in all_hits:
                all_hits[k] += hits[k]
            all_latencies += latencies
            total_n += n

        results[label] = {
            "load_s": load_s,
            "hits": all_hits,
            "n": total_n,
            "latencies": sorted(all_latencies),
        }

    print("\n=== TOTAL (129 sentences) ===")
    print(f"{'field':14s}" + "".join(f"{label:>16s}" for label in _MODELS))
    for k in ("target_type", "landmark", "relation", "situation"):
        row = f"{k:14s}"
        for label in _MODELS:
            r = results[label]
            row += f"{100*r['hits'][k]/r['n']:>15.0f}%"
        print(row)

    print(f"\n{'latency':14s}" + "".join(f"{label:>16s}" for label in _MODELS))
    for stat_name, fn in (("median", lambda L: L[len(L)//2]), ("p95", lambda L: L[int(len(L)*0.95)]), ("max", lambda L: L[-1])):
        row = f"{stat_name:14s}"
        for label in _MODELS:
            row += f"{fn(results[label]['latencies']):>15.2f}s"
        print(row)

    row = f"{'load':14s}"
    for label in _MODELS:
        row += f"{results[label]['load_s']:>15.2f}s"
    print(row)
