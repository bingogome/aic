"""Thin wrapper around X-VLA's official training scripts that:

  1. Registers the aic domain handler with X-VLA's registry.
  2. Selects between X-VLA's `train.py` (full FT) and `peft_train.py` (LoRA FT).
  3. Forwards all remaining CLI args to the chosen entry point.

Usage (from inside X-VLA's environment, with both repos on PYTHONPATH):
    PYTHONPATH=$XVLA_REPO:$AIC_XVLA_PKG \\
    accelerate launch -m aic_xvla.train --mode peft \\
        --models 2toINF/X-VLA-Pt \\
        --train_metas_path /path/to/aic_meta.json \\
        --output_dir runnings/aic_xvla \\
        --batch_size 1 --learning_rate 1e-4 \\
        --iters 2000 --save_interval 1000

Both `train.py` (full) and `peft_train.py` (LoRA) are official X-VLA
fine-tune recipes. Pick `peft` when GPU memory is tight; `full` when you
have the budget to update all weights.
"""

from __future__ import annotations

import argparse
import sys

from aic_xvla.handler import register as register_aic_handler


def main() -> None:
    register_aic_handler()

    mode_parser = argparse.ArgumentParser(add_help=False)
    mode_parser.add_argument(
        "--mode",
        choices=["full", "peft"],
        default="full",
        help="X-VLA training entry point: 'full' = train.py, 'peft' = peft_train.py (LoRA)",
    )
    mode_args, remaining = mode_parser.parse_known_args()

    if mode_args.mode == "peft":
        from peft_train import get_args_parser
        from peft_train import main as xvla_main
    else:
        from train import get_args_parser
        from train import main as xvla_main

    parser = argparse.ArgumentParser("aic-xvla training", parents=[get_args_parser()])
    args = parser.parse_args(remaining)
    xvla_main(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
