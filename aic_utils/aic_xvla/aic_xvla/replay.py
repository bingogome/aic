"""Offline replay eval: predict action chunks for sampled frames of an aic
episode and compare against ground-truth actions.

Reports:
  - per-axis mean absolute error on TCP position (xyz, in meters)
  - quaternion-distance angle error (radians)
  - per-step trajectory error for the first action of each chunk

Use this as a sanity check before closed-loop rollouts in the engine.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pyarrow.parquet as pq
from aic_xvla.eval import DEFAULT_INSTRUCTION, AICXVLAPolicy
from aic_xvla.handler import _state_to_proprio
from PIL import Image
from scipy.spatial.transform import Rotation as R


def _quat_angle_deg(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """Per-row angle between two unit quaternions in degrees."""
    dot = np.clip(np.abs(np.sum(qa * qb, axis=-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def replay_episode(
    policy: AICXVLAPolicy,
    parquet_path: str,
    image_root: str,
    instruction: str,
    sample_every: int,
    horizon: int,
) -> dict:
    table = pq.read_table(parquet_path).to_pydict()
    n_frames = len(table["frame_index"])
    state_cols = [np.asarray(table[f"state_{i}"], dtype=np.float64) for i in range(26)]
    action_cols = [np.asarray(table[f"action_{i}"], dtype=np.float64) for i in range(7)]
    state = np.stack(state_cols, axis=-1)  # [T, 26]
    action_gt = np.stack(action_cols, axis=-1)  # [T, 7]
    left_paths = list(table["image_path_left_camera"])
    center_paths = list(table["image_path_center_camera"])
    right_paths = list(table["image_path_right_camera"])

    pos_errors = []
    quat_errors = []
    sampled_idx = list(range(0, n_frames - horizon, sample_every))
    print(
        f"replaying {len(sampled_idx)} frames (every {sample_every}, horizon {horizon}) of {n_frames}"
    )

    for n, idx in enumerate(sampled_idx):
        images = [
            Image.open(os.path.join(image_root, left_paths[idx])),
            Image.open(os.path.join(image_root, center_paths[idx])),
            Image.open(os.path.join(image_root, right_paths[idx])),
        ]
        proprio = _state_to_proprio(state[idx])  # [10]
        pred = policy.predict(images, proprio, instruction)  # [num_actions, 7]
        # Compare first `horizon` predictions to next `horizon` GT actions.
        h = min(horizon, pred.shape[0], n_frames - idx)
        pred_h = pred[:h]
        gt_h = action_gt[idx : idx + h]
        pos_err = np.abs(pred_h[:, :3] - gt_h[:, :3])  # [h, 3]
        # Normalize quats before angle distance (model output may drift slightly).
        pred_q = pred_h[:, 3:7] / np.linalg.norm(
            pred_h[:, 3:7], axis=-1, keepdims=True
        ).clip(1e-8)
        gt_q = gt_h[:, 3:7] / np.linalg.norm(gt_h[:, 3:7], axis=-1, keepdims=True).clip(
            1e-8
        )
        ang_err = _quat_angle_deg(pred_q, gt_q)  # [h]
        pos_errors.append(pos_err)
        quat_errors.append(ang_err)
        if n < 3 or n == len(sampled_idx) - 1:
            print(
                f"  frame {idx:4d}: pred[0] pos={pred[0, :3].round(4).tolist()} "
                f"gt[0] pos={action_gt[idx, :3].round(4).tolist()} "
                f"|Δpos|={pos_err[0].sum():.4f}m  Δang={ang_err[0]:.2f}°"
            )

    pos_errors = np.concatenate(pos_errors, axis=0)  # [N*h, 3]
    quat_errors = np.concatenate(quat_errors, axis=0)
    return {
        "n_frames_evaluated": int(pos_errors.shape[0]),
        "pos_mae_m": pos_errors.mean(axis=0).tolist(),
        "pos_mae_total_m": float(pos_errors.mean()),
        "pos_max_m": float(pos_errors.max()),
        "ang_mae_deg": float(quat_errors.mean()),
        "ang_max_deg": float(quat_errors.max()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Offline replay eval for aic-xvla.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--meta", required=True, help="aic_meta.json from aic_xvla.build_meta"
    )
    p.add_argument("--episode", type=int, default=0, help="datalist index")
    p.add_argument("--sample-every", type=int, default=20, help="frame stride")
    p.add_argument("--horizon", type=int, default=10, help="actions per chunk to score")
    p.add_argument("--instruction", default=None)
    p.add_argument("--out", default=None, help="optional JSON results path")
    args = p.parse_args()

    with open(args.meta) as f:
        meta = json.load(f)
    item = meta["datalist"][args.episode]
    instruction = args.instruction or item.get("instruction", DEFAULT_INSTRUCTION)

    policy = AICXVLAPolicy(args.checkpoint)
    results = replay_episode(
        policy,
        parquet_path=item["parquet_path"],
        image_root=item["image_root"],
        instruction=instruction,
        sample_every=args.sample_every,
        horizon=args.horizon,
    )
    print("\n=== Replay results ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
