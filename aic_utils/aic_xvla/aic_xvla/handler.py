"""X-VLA domain handler for aic LeRobot-flat dataset.

Schema (per-frame parquet row):
    state_0..25  : 26D proprio (TCP pose+vel+err, 6 joints, 1 gripper)
    action_0..6  : 7D TCP target (xyz + quat xyzw)
    image_path_{left,center,right}_camera : relative JPG paths
    episode_id, frame_index, timestamp, success, plug_port_dist

Mapping to X-VLA `ee6d` action space (dim_action=20):
    proprio[0:3]  = TCP pos (state_0..2)
    proprio[3:9]  = TCP rot6d from quat (state_3..6 -> rot6d)
    proprio[9]    = gripper (state_25)
    proprio[10:20]= 0
    action[0:3]   = target pos
    action[3:9]   = target rot6d
    action[9]     = gripper (held = current proprio[9])
    action[10:20] = 0
"""

from __future__ import annotations

import os
import random
from typing import Iterable

import numpy as np
import torch
from datasets.domain_handler.base import DomainHandler
from datasets.domain_handler.registry import _REGISTRY
from datasets.utils import quat_to_rotate6d, read_parquet
from PIL import Image
from scipy.interpolate import interp1d

DATASET_NAME = "aic"
DEFAULT_INSTRUCTION = "insert the SFP cable into the port"

# "delta": position predicted relative to proprio (X-VLA's action_slice
# subtracts proprio[idx] at training; eval adds it back).
# "absolute": position predicted in base_link directly (matches dataset).
ACTION_ENCODING = os.environ.get("AIC_XVLA_ACTION_ENCODING", "delta").lower()
if ACTION_ENCODING not in ("delta", "absolute"):
    raise ValueError(
        f"AIC_XVLA_ACTION_ENCODING must be delta|absolute, got {ACTION_ENCODING!r}"
    )
_IDX_FOR_DELTA = [0, 1, 2] if ACTION_ENCODING == "delta" else []


def _state_to_proprio(state: np.ndarray) -> np.ndarray:
    """state: [26] -> proprio: [10] = pos(3) + rot6d(6) + gripper(1)."""
    pos = state[0:3]
    quat_xyzw = state[3:7]
    rot6d = quat_to_rotate6d(quat_xyzw, scalar_first=False)
    gripper = state[25:26]
    return np.concatenate([pos, rot6d, gripper], axis=-1).astype(np.float32)


def _action_to_xvla(action7: np.ndarray, gripper: float) -> np.ndarray:
    """action7: [7] = pos(3)+quat(4) -> [10] = pos(3)+rot6d(6)+gripper(1)."""
    pos = action7[0:3]
    quat_xyzw = action7[3:7]
    rot6d = quat_to_rotate6d(quat_xyzw, scalar_first=False)
    return np.concatenate([pos, rot6d, [gripper]], axis=-1).astype(np.float32)


def _build_abs_trajectory(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """state: [T, 26], action: [T, 7] -> abs_trajectory: [T, 10] in xvla 9D+gripper format.

    Row format matches proprio layout so action_slice + delta indices work consistently.
    Action gripper held from proprio (single-arm, no commanded gripper in 7D action).
    """
    T = state.shape[0]
    out = np.zeros((T, 10), dtype=np.float32)
    pos = action[:, 0:3]
    quat = action[:, 3:7]
    rot6d = quat_to_rotate6d(quat, scalar_first=False)
    gripper = state[:, 25:26]
    out[:, 0:3] = pos
    out[:, 3:9] = rot6d
    out[:, 9:10] = gripper
    return out


class AICHandler(DomainHandler):
    """Streams X-VLA training samples from aic flat-parquet episodes.

    Each meta entry in `meta['datalist']` is:
        {"parquet_path": str, "image_root": str, "instruction": str, "fps": int}
    """

    CAMERA_KEYS = [
        "image_path_left_camera",
        "image_path_center_camera",
        "image_path_right_camera",
    ]
    QDUR = 1.0  # seconds of future window
    # Driven by AIC_XVLA_ACTION_ENCODING env var (read at module load).
    idx_for_delta = _IDX_FOR_DELTA
    idx_for_mask_proprio: list[int] = []

    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        lang_aug_map: dict | None,
        **kwargs,
    ) -> Iterable[dict]:
        item = self.meta["datalist"][traj_idx]
        parquet_path = item["parquet_path"]
        image_root = item.get("image_root", os.path.dirname(parquet_path))
        instruction = item.get("instruction", DEFAULT_INSTRUCTION)
        fps = float(item.get("fps", self.meta.get("fps", 20)))

        data = read_parquet(parquet_path)
        state = np.stack(
            [np.asarray(data[f"state_{i}"], dtype=np.float64) for i in range(26)],
            axis=-1,
        )
        action = np.stack(
            [np.asarray(data[f"action_{i}"], dtype=np.float64) for i in range(7)],
            axis=-1,
        )
        T = state.shape[0]
        if T < num_actions + 2:
            return

        abs_traj = _build_abs_trajectory(state, action)  # [T, 10]
        # Pad to xvla ee6d width (20).
        abs_traj_pad = np.concatenate(
            [abs_traj, np.zeros((T, 20 - abs_traj.shape[1]), dtype=np.float32)], axis=-1
        )
        t = np.arange(T, dtype=np.float64) / fps
        interp = interp1d(
            t,
            abs_traj_pad,
            axis=0,
            bounds_error=False,
            fill_value=(abs_traj_pad[0], abs_traj_pad[-1]),
        )

        image_paths_per_view = []
        for cam in self.CAMERA_KEYS[: self.num_views]:
            paths = list(data[cam]) if cam in data else []
            image_paths_per_view.append(paths)
        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        image_mask[: len([v for v in image_paths_per_view if v])] = True

        idxs = list(range(0, T - num_actions - 1))
        if training:
            random.shuffle(idxs)

        for idx in idxs:
            cur = t[idx]
            q = np.linspace(
                cur,
                min(cur + self.QDUR, float(t[-1])),
                num_actions + 1,
                dtype=np.float32,
            )
            cur_action = torch.tensor(interp(q))  # [num_actions+1, 20]
            if (cur_action[1] - cur_action[0]).abs().max() < 1e-5:
                continue

            imgs = []
            for v, paths in enumerate(image_paths_per_view):
                if not paths:
                    continue
                rel = paths[idx]
                pil = Image.open(os.path.join(image_root, rel)).convert("RGB")
                imgs.append(image_aug(pil))
            while len(imgs) < self.num_views:
                imgs.append(torch.zeros_like(imgs[0]))
            image_input = torch.stack(imgs, dim=0)

            ins = instruction
            if training and lang_aug_map and ins in lang_aug_map:
                ins = random.choice(lang_aug_map[ins])

            yield {
                "language_instruction": ins,
                "image_input": image_input,
                "image_mask": image_mask,
                "abs_trajectory": cur_action.float(),
                "idx_for_delta": self.idx_for_delta,
                "idx_for_mask_proprio": self.idx_for_mask_proprio,
            }


def register() -> None:
    """Register AICHandler with X-VLA's domain registry. Idempotent."""
    _REGISTRY[DATASET_NAME] = AICHandler
