"""Compare the LLM parser (parser.py) against the rule-based baseline (rule_based_baseline.py)
on all three test sets. This is the ablation for the paper's "why an LLM" section.

Usage: python3 compare_baseline.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # parser.py etc. live in nlu/

from parser import NLUParser
from rule_based_baseline import RuleBasedParser
from test_data import KNOWN_EXAMPLES
from paraphrases import PARAPHRASE_EXAMPLES
from held_out_sentences import HELD_OUT_EXAMPLES


def score(parse_fn, cases, expected_prefix=""):
    hits = {"target_type": 0, "landmark": 0, "relation": 0, "situation": 0}
    for case in cases:
        expected = {k: case.get(f"{expected_prefix}{k}") for k in hits}
        got = parse_fn(case["sentence"])
        for k in hits:
            if str(got.get(k)).lower() == str(expected[k]).lower():
                hits[k] += 1
    n = len(cases)
    return {k: (v, n, round(100 * v / n)) for k, v in hits.items()}


def print_table(name, llm_scores, rule_scores):
    print(f"\n=== {name} ===")
    print(f"{'field':12s} {'rule-based':>12s} {'LLM (Qwen)':>12s} {'diff':>8s}")
    for k in llm_scores:
        rv, n, rp = rule_scores[k]
        lv, _, lp = llm_scores[k]
        print(f"{k:12s} {rp:>10d}% {lp:>11d}% {lp - rp:>+7d}p")


if __name__ == "__main__":
    print("Loading models (LLM load is slow)...")
    llm = NLUParser()
    rule = RuleBasedParser()

    sets = [
        ("Known 39", KNOWN_EXAMPLES, ""),
        ("Paraphrase 66", PARAPHRASE_EXAMPLES, ""),
        ("Held-out 56 (generalization)", HELD_OUT_EXAMPLES, "expected_"),
    ]

    all_llm_hits = {"target_type": 0, "landmark": 0, "relation": 0, "situation": 0}
    all_rule_hits = {"target_type": 0, "landmark": 0, "relation": 0, "situation": 0}
    total_n = 0

    for name, cases, prefix in sets:
        llm_scores = score(llm.parse, cases, prefix)
        rule_scores = score(rule.parse, cases, prefix)
        print_table(name, llm_scores, rule_scores)
        for k in all_llm_hits:
            all_llm_hits[k] += llm_scores[k][0]
            all_rule_hits[k] += rule_scores[k][0]
        total_n += len(cases)

    print(f"\n=== TOTAL ({total_n} sentences) ===")
    print(f"{'field':12s} {'rule-based':>12s} {'LLM (Qwen)':>12s} {'diff':>8s}")
    for k in all_llm_hits:
        rp = round(100 * all_rule_hits[k] / total_n)
        lp = round(100 * all_llm_hits[k] / total_n)
        print(f"{k:12s} {rp:>10d}% {lp:>11d}% {lp - rp:>+7d}p")
