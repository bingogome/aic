"""Stitch RunXVLA's per-step camera images into a tiled MP4.

RunXVLA saves frames to <out_dir>/images/{left,center,right}/NNNNNN.jpg every
replan (one set per chunk). This script sorts them by step number and
ffmpeg-encodes a 3-camera tiled video.

Usage:
    pixi run python -m aic_xvla.images_to_mp4 \
      --image-root /home/yifeng/aic_xvla_overfit_abs \
      --out /home/yifeng/aic_xvla_overfit_abs/closedloop_pose_r15.mp4 \
      --fps 5

Notes:
- Frames are sparse (one per replan). At fps=5, an 18-replan run plays in
  ~3.6s. Bump --fps to slow it down, or use --hold-frames to repeat each.
- The image dir is shared across runs. Use --min-step / --max-step to clip
  to a specific run's step range (read from its closedloop_trace_*.jsonl).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

CAMS = ("left", "center", "right")


def collect_frames(image_root: Path, min_step: int, max_step: int) -> list[int]:
    pat = re.compile(r"^(\d{6})\.jpg$")
    common: set[int] | None = None
    for cam in CAMS:
        d = image_root / "images" / cam
        if not d.is_dir():
            raise SystemExit(f"missing camera dir: {d}")
        steps = {int(m.group(1)) for f in d.iterdir() if (m := pat.match(f.name))}
        common = steps if common is None else (common & steps)
    sel = sorted(s for s in (common or set()) if min_step <= s <= max_step)
    if not sel:
        raise SystemExit("no frames matched (check --min-step/--max-step and image-root).")
    return sel


def tile(left: np.ndarray, center: np.ndarray, right: np.ndarray) -> np.ndarray:
    h = max(im.shape[0] for im in (left, center, right))
    resized = []
    for im in (left, center, right):
        scale = h / im.shape[0]
        w = int(im.shape[1] * scale)
        resized.append(cv2.resize(im, (w, h)))
    return np.concatenate(resized, axis=1)


def annotate(im: np.ndarray, text: str) -> np.ndarray:
    out = im.copy()
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-root", required=True, type=Path,
                    help="dir containing images/{left,center,right}/NNNNNN.jpg")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--min-step", type=int, default=0)
    ap.add_argument("--max-step", type=int, default=10**9)
    ap.add_argument("--hold-frames", type=int, default=1,
                    help="repeat each frame N times (slow-mo)")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH")

    steps = collect_frames(args.image_root, args.min_step, args.max_step)
    print(f"matched {len(steps)} frames (steps {steps[0]}..{steps[-1]})")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        idx = 0
        for s in steps:
            left = cv2.imread(str(args.image_root / "images/left" / f"{s:06d}.jpg"))
            center = cv2.imread(str(args.image_root / "images/center" / f"{s:06d}.jpg"))
            right = cv2.imread(str(args.image_root / "images/right" / f"{s:06d}.jpg"))
            tiled = annotate(tile(left, center, right), f"step {s}")
            for _ in range(max(1, args.hold_frames)):
                cv2.imwrite(str(tmpdir / f"{idx:06d}.png"), tiled)
                idx += 1

        args.out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-framerate", str(args.fps),
            "-i", str(tmpdir / "%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            str(args.out),
        ]
        subprocess.run(cmd, check=True)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
