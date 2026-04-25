"""Thin wrapper around X-VLA's official `train.py` that:

  1. Registers the aic domain handler with X-VLA's registry.
  2. Optionally enables W&B (via tensorboard sync — see README).
  3. Forwards all CLI args to X-VLA's `main()`.

Usage (from inside X-VLA's environment, with both repos on PYTHONPATH):
    PYTHONPATH=$XVLA_REPO:$AIC_XVLA_PKG \
    accelerate launch -m aic_xvla.train \
        --models 2toINF/X-VLA-Pt \
        --train_metas_path /path/to/aic_meta.json \
        --output_dir runnings/aic_xvla \
        --batch_size 16 --learning_rate 1e-4 \
        --iters 100000 --save_interval 10000
"""

from __future__ import annotations

import argparse
import sys

from aic_xvla.handler import register as register_aic_handler


def main() -> None:
    register_aic_handler()
    # Defer X-VLA imports until after registration so the module-level
    # registry is populated before train.py captures it.
    from train import get_args_parser
    from train import main as xvla_main

    parser = argparse.ArgumentParser("aic-xvla training", parents=[get_args_parser()])
    args = parser.parse_args()
    xvla_main(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
