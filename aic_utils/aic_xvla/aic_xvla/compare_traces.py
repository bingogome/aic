"""Diff an offline replay trace against a closed-loop RunXVLA trace.

Both traces are JSONL where each line records one inference call.
Schema (a superset across the two sources):
  {
    "frame" or "step": int,
    "live_pos": [x, y, z],
    "live_quat_xyzw": [...],
    "instruction": str,
    "image_paths": [left, center, right],
    "pred_actions": [[x, y, z, qx, qy, qz, qw], ...],
    "gt_actions": [[...], ...]   # offline only
  }

We line them up by sequence (offline frame 0 ↔ closed step 0, etc.) and
report how the *predictions* differ given the *live_pos* drift between
the two runs.

Usage:
    python -m aic_xvla.compare_traces \\
        --offline /home/yifeng/aic_xvla_overfit/replay_trace.jsonl \\
        --closed  /home/yifeng/aic_xvla_overfit/closedloop_trace.jsonl
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def _load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _key(rec: dict) -> int:
    return rec.get("frame", rec.get("step", -1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offline", required=True)
    p.add_argument("--closed", required=True)
    p.add_argument("--n", type=int, default=10, help="number of pairs to print")
    args = p.parse_args()

    off = _load(args.offline)
    clo = _load(args.closed)
    n = min(len(off), len(clo), args.n)
    print(
        f"loaded offline={len(off)} closed-loop={len(clo)}; comparing first {n} entries\n"
    )

    # Header row.
    print(
        f"{'i':>3}  {'off frame':>9}  {'clo step':>8}  "
        f"{'off live xyz':>32}  {'clo live xyz':>32}  "
        f"{'|Δlive|':>7}  {'off pred[0] xyz':>32}  {'clo pred[0] xyz':>32}  "
        f"{'|Δpred[0]|':>9}"
    )
    live_diffs = []
    pred_diffs = []
    for i in range(n):
        a, b = off[i], clo[i]
        a_live = np.array(a["live_pos"])
        b_live = np.array(b["live_pos"])
        a_pred = np.array(a["pred_actions"][0])
        b_pred = np.array(b["pred_actions"][0])
        live_diff = np.linalg.norm(a_live - b_live)
        pred_diff = np.linalg.norm(a_pred[:3] - b_pred[:3])
        live_diffs.append(live_diff)
        pred_diffs.append(pred_diff)
        print(
            f"{i:>3}  {_key(a):>9}  {_key(b):>8}  "
            f"{str(a_live.round(4).tolist()):>32}  {str(b_live.round(4).tolist()):>32}  "
            f"{live_diff:>7.4f}  "
            f"{str(a_pred[:3].round(4).tolist()):>32}  {str(b_pred[:3].round(4).tolist()):>32}  "
            f"{pred_diff:>9.4f}"
        )

    print("\n=== summary ===")
    print(f"  mean |Δlive|     : {np.mean(live_diffs):.4f} m   (input drift)")
    print(f"  mean |Δpred[0]|  : {np.mean(pred_diffs):.4f} m   (output divergence)")
    print(f"  off instruction  : {off[0]['instruction']!r}")
    print(f"  clo instruction  : {clo[0]['instruction']!r}")
    print(f"  off image paths  : {off[0]['image_paths']}")
    print(f"  clo image paths  : {clo[0]['image_paths']}")


if __name__ == "__main__":
    main()
