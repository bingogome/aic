"""Generate X-VLA `meta.json` for an aic flat-parquet dataset.

Discovers episodes either by:
  - one parquet per episode (passed as `--parquet-glob`), or
  - a single multi-episode parquet split by `episode_id` column.

Writes a meta file with `dataset_name="aic"` and a `datalist` of episodes.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

DEFAULT_INSTRUCTION = "insert the SFP cable into the port"


def build_from_per_episode_parquets(
    parquet_paths: list[str], image_root: str, instruction: str, fps: int
) -> dict:
    datalist = []
    for p in sorted(parquet_paths):
        datalist.append(
            {
                "parquet_path": str(Path(p).resolve()),
                "image_root": str(Path(image_root).resolve()),
                "instruction": instruction,
                "fps": fps,
            }
        )
    return {
        "dataset_name": "aic",
        "fps": fps,
        "datalist": datalist,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Build X-VLA meta.json for aic dataset.")
    p.add_argument(
        "--parquet-glob", required=True, help="glob for per-episode parquet files"
    )
    p.add_argument(
        "--image-root",
        required=True,
        help="root dir under which image_path_* are resolved",
    )
    p.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--out", required=True, help="output meta.json path")
    args = p.parse_args()

    paths = glob.glob(args.parquet_glob)
    if not paths:
        raise SystemExit(f"no parquet files matched glob: {args.parquet_glob}")
    meta = build_from_per_episode_parquets(
        paths, args.image_root, args.instruction, args.fps
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {args.out} with {len(meta['datalist'])} episodes")


if __name__ == "__main__":
    main()
