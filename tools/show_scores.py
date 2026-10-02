#!/usr/bin/env python3
"""Per-round Stage 1 score breakdown (cam / type / lmk / rel / anc / tgt / total).

The platform prints these per round in its OWN container log the moment each round is
scored -- the agent's msg 401 only carries the round total. Two sources, same table:

  # live, while a run is going (parses the platform container log):
  python3 tools/show_scores.py --log
  python3 tools/show_scores.py --watch          # follow, one row per round as it scores

  # after a run, full detail incl. expected vs submitted camera and coord error:
  python3 tools/show_scores.py                   # newest results JSON
  python3 tools/show_scores.py path/to/team_YYYYMMDD_HHMMSS.json

Override the platform container name with --container and the results dir with --results.
"""
import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys

COMPONENTS = ("camera", "object_type", "landmark", "relation", "anchor", "target")
HDR = f"{'rnd':>3}  {'problem':<26} {'cam':>4} {'type':>4} {'lmk':>4} {'rel':>4} {'anc':>4} {'tgt':>4} {'total':>6}"
_LOG_RE = re.compile(
    r"Stage1 score \[(?P<pid>[^\]]+)\] kind=(?P<kind>\w+): "
    r"camera=(?P<camera>[\d.]+), type/situation=(?P<object_type>[\d.]+), "
    r"landmark=(?P<landmark>[\d.]+)(?:\(n/a\))?, relation=(?P<relation>[\d.]+)(?:\(n/a\))?, "
    r"anchor=(?P<anchor>[\d.]+), target=(?P<target>[\d.]+), total=(?P<total>[\d.]+)"
)


def _row(rnd, pid, comps, total, kind=None):
    na = kind == "person"
    def c(k):
        v = comps.get(k)
        if v is None:
            return "  - "
        if na and k in ("landmark", "relation"):
            return " n/a"
        return f"{v:4.2f}"
    return (f"{rnd:>3}  {pid:<26} {c('camera')} {c('object_type')} {c('landmark')} "
            f"{c('relation')} {c('anchor')} {c('target')} {total:6.1f}")


def from_results(path):
    d = json.load(open(path))
    rd = d.get("round_details", [])
    print(f"# {os.path.basename(path)}   stage1_avg={d['scores'].get('stage1_average')}   "
          f"stage2={d['scores'].get('stage2')}   total={d['scores'].get('total')}")
    print(HDR + f" {'GT cam':>9} {'sub cam':>9} {'tgt err':>8}")
    sums = {k: 0.0 for k in COMPONENTS}
    lost_cam = []
    for i, r in enumerate(rd, 1):
        comps = {k: r.get(k) for k in COMPONENTS}
        e, s = r.get("expected", {}), r.get("submission", {})
        gt_cam, sub_cam = e.get("camera_id", "?"), (s.get("camera_id") or "-")
        err = ""
        et, st_ = e.get("target_coord"), s.get("target_coord")
        if et and st_ and any(st_):
            err = f"{math.dist(et[:3], st_[:3]):8.2f}"
        line = _row(i, r.get("problem_id", "?"), comps, r.get("total", 0.0), e.get("target_kind"))
        flag = "  <-- cam" if comps.get("camera") == 0.0 else ""
        print(f"{line} {gt_cam:>9} {sub_cam:>9} {err:>8}{flag}")
        for k in COMPONENTS:
            if comps.get(k) is not None:
                sums[k] += comps[k]
        if comps.get("camera") == 0.0:
            lost_cam.append(i)
    n = len(rd) or 1
    print("-" * len(HDR))
    print(f"{'avg':>3}  {'(fraction of max)':<26} " +
          " ".join(f"{sums[k]/n:4.2f}" for k in COMPONENTS))
    print(f"\ncamera lost on {len(lost_cam)}/{len(rd)} rounds: {lost_cam}")


def from_log(container, follow):
    cmd = ["docker", "logs"] + (["-f"] if follow else []) + [container]
    if follow:
        print(HDR)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        rnd = 0
        try:
            for ln in p.stdout:
                m = _LOG_RE.search(ln)
                if not m or m.group("pid").startswith("rt"):
                    continue
                rnd += 1
                g = m.groupdict()
                comps = {k: float(g[k]) for k in COMPONENTS}
                print(_row(rnd, g["pid"], comps, float(g["total"]), g["kind"]))
        except KeyboardInterrupt:
            pass
        return
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
    rows = [m.groupdict() for m in _LOG_RE.finditer(out) if not m.group("pid").startswith("rt")]
    if not rows:
        sys.exit(f"no 'Stage1 score [...]' lines in `docker logs {container}` "
                 f"(wrong --container? try: docker ps --format '{{{{.Names}}}}')")
    print(HDR)
    sums = {k: 0.0 for k in COMPONENTS}
    for i, g in enumerate(rows, 1):
        comps = {k: float(g[k]) for k in COMPONENTS}
        print(_row(i, g["pid"], comps, float(g["total"]), g["kind"]))
        for k in COMPONENTS:
            sums[k] += comps[k]
    n = len(rows)
    print("-" * len(HDR))
    print(f"{'avg':>3}  {'(fraction of max)':<26} " + " ".join(f"{sums[k]/n:4.2f}" for k in COMPONENTS))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_json", nargs="?", help="a specific results JSON (default: newest)")
    ap.add_argument("--log", action="store_true", help="read the platform container log instead")
    ap.add_argument("--watch", action="store_true", help="follow the platform log, a row per round")
    ap.add_argument("--container", default="simulation-platform-platform-1")
    ap.add_argument("--results", default=os.path.join(
        os.path.dirname(__file__), "..", "results", "marc2026_chungmu"))
    a = ap.parse_args()

    if a.log or a.watch:
        from_log(a.container, a.watch)
        return
    path = a.results_json
    if not path:
        cands = [p for p in glob.glob(os.path.join(a.results, "*.json"))
                 if not p.endswith("_summary.json")]
        if not cands:
            sys.exit(f"no results JSON under {a.results} -- pass one explicitly or use --log")
        path = max(cands, key=os.path.getmtime)
    from_results(path)


if __name__ == "__main__":
    main()
