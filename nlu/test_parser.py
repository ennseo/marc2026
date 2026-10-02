"""Run the NLU parser against known + paraphrased + held-out sentences and report accuracy.

Usage:
    cd nlu
    python3 test_parser.py                # known 39 only
    python3 test_parser.py --paraphrases   # 66 paraphrased variants only
    python3 test_parser.py --held-out      # held-out (unseen vocabulary) only -- the most
                                            # important number: this is the real generalization check
    python3 test_parser.py --all           # all three sets (176 sentences)
"""

import argparse
import time

# works both as `cd nlu && python3 test_parser.py` and `python3 -m nlu.test_parser` from /agent
try:
    from nlu.parser import NLUParser
    from nlu.test_data import KNOWN_EXAMPLES
    from nlu.paraphrases import PARAPHRASE_EXAMPLES
    from nlu.held_out_sentences import HELD_OUT_EXAMPLES
except ModuleNotFoundError:
    from parser import NLUParser
    from test_data import KNOWN_EXAMPLES
    from paraphrases import PARAPHRASE_EXAMPLES
    from held_out_sentences import HELD_OUT_EXAMPLES


def run(parser: NLUParser, cases: list[dict], expected_prefix: str = "") -> None:
    field_hits = {"target_type": 0, "landmark": 0, "relation": 0, "situation": 0}
    total = len(cases)
    t0 = time.time()

    for i, case in enumerate(cases, 1):
        sentence = case["sentence"]
        expected = {
            "target_type": case.get(f"{expected_prefix}target_type"),
            "landmark": case.get(f"{expected_prefix}landmark"),
            "relation": case.get(f"{expected_prefix}relation"),
            "situation": case.get(f"{expected_prefix}situation"),
        }
        got = parser.parse(sentence)

        ok = all(str(got.get(k)).lower() == str(expected[k]).lower() for k in field_hits)
        for k in field_hits:
            if str(got.get(k)).lower() == str(expected[k]).lower():
                field_hits[k] += 1

        mark = "OK  " if ok else "FAIL"
        print(f"[{mark}] {i}/{total} \"{sentence[:60]}...\"" if len(sentence) > 60 else f"[{mark}] {i}/{total} \"{sentence}\"")
        if not ok:
            print(f"       expected: {expected}")
            print(f"       got:      {got}")

    elapsed = time.time() - t0
    print()
    print(f"--- {total} sentences in {elapsed:.1f}s ({elapsed/total:.1f}s/sentence) ---")
    for k, hits in field_hits.items():
        print(f"  {k:12s}: {hits}/{total} ({100*hits/total:.0f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--held-out", action="store_true", help="run only the held-out (unseen vocabulary) sentences")
    ap.add_argument("--paraphrases", action="store_true", help="run only the paraphrased sentences")
    ap.add_argument("--all", action="store_true", help="run known + paraphrases + held-out")
    args = ap.parse_args()
    any_flag = args.held_out or args.paraphrases or args.all

    print("Loading model (first load is slow)...")
    parser = NLUParser()

    if args.held_out or args.all:
        print("\n=== Held-out (unseen vocabulary) -- the real generalization check ===")
        run(parser, HELD_OUT_EXAMPLES, expected_prefix="expected_")

    if args.paraphrases or args.all:
        print("\n=== Paraphrases (same 33, reworded) -- wording-robustness check ===")
        run(parser, PARAPHRASE_EXAMPLES, expected_prefix="")

    if not any_flag or args.all:
        print("\n=== Known 39 examples ===")
        run(parser, KNOWN_EXAMPLES, expected_prefix="")
