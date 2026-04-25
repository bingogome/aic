"""Eval / inference wrapper for a fine-tuned X-VLA checkpoint on aic.

Supports both full-FT checkpoints (`XVLA.from_pretrained(ckpt)`) and LoRA
adapter checkpoints (load base from `adapter_config.json:base_model_name_or_path`,
then `PeftModel.from_pretrained(base, ckpt)`).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from datasets.utils import rotate6d_to_quat
from models.modeling_xvla import XVLA
from models.processing_xvla import XVLAProcessor
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

DEFAULT_INSTRUCTION = "insert the SFP cable into the port"

_IMAGE_TF = transforms.Compose(
    [
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


def _is_lora_ckpt(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def _load_model(checkpoint: str, device: str) -> tuple[torch.nn.Module, XVLAProcessor]:
    if _is_lora_ckpt(checkpoint):
        from peft import PeftModel

        with open(os.path.join(checkpoint, "adapter_config.json")) as f:
            adapter_cfg = json.load(f)
        base_id = adapter_cfg["base_model_name_or_path"]
        base = XVLA.from_pretrained(base_id)
        model = PeftModel.from_pretrained(base, checkpoint)
        processor = XVLAProcessor.from_pretrained(base_id)
    else:
        model = XVLA.from_pretrained(checkpoint)
        processor = XVLAProcessor.from_pretrained(checkpoint)
    return model.to(device).eval(), processor


class AICXVLAPolicy:
    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        domain_id: int = 0,
        steps: int = 10,
    ):
        self.model, self.processor = _load_model(checkpoint, device)
        self.device = device
        self.domain_id = domain_id
        self.steps = steps

    @torch.no_grad()
    def predict(
        self,
        images: Sequence[Image.Image],
        proprio_10d: np.ndarray,
        instruction: str = DEFAULT_INSTRUCTION,
    ) -> np.ndarray:
        if proprio_10d.shape[-1] < 20:
            proprio = np.concatenate(
                [proprio_10d, np.zeros(20 - proprio_10d.shape[-1], dtype=np.float32)]
            )
        else:
            proprio = proprio_10d.astype(np.float32)

        image_input = (
            torch.stack([_IMAGE_TF(im.convert("RGB")) for im in images], dim=0)
            .unsqueeze(0)
            .to(self.device)
        )
        image_mask = torch.ones(1, len(images), dtype=torch.bool, device=self.device)
        proprio_t = torch.from_numpy(proprio).unsqueeze(0).to(self.device)
        domain_id = torch.tensor([self.domain_id], dtype=torch.long, device=self.device)
        lang = self.processor.encode_language([instruction])
        lang = {k: v.to(self.device) for k, v in lang.items()}

        # Find the underlying XVLA module (PeftModel wraps it).
        base = (
            self.model.get_base_model()
            if hasattr(self.model, "get_base_model")
            else self.model
        )
        actions = base.generate_actions(
            image_input=image_input,
            image_mask=image_mask,
            proprio=proprio_t,
            domain_id=domain_id,
            steps=self.steps,
            **lang,
        )  # [1, num_actions, 20]
        return self._xvla_to_aic_actions(actions[0].cpu().float().numpy(), proprio[:3])

    @staticmethod
    def _xvla_to_aic_actions(act20: np.ndarray, proprio_pos: np.ndarray) -> np.ndarray:
        """[T, 20] -> [T, 7] = pos(3) + quat_xyzw(4).

        Position is predicted as a delta vs. proprio (matching `idx_for_delta=[0,1,2]`
        in the training handler), so we add proprio_pos back here to recover absolute.
        """
        pos = act20[:, 0:3] + proprio_pos.reshape(1, 3)
        rot6d = act20[:, 3:9]
        quat = rotate6d_to_quat(rot6d, scalar_first=False)
        return np.concatenate([pos, quat], axis=-1).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description="One-shot inference smoke test.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--left", required=True)
    p.add_argument("--center", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    args = p.parse_args()

    images = [Image.open(args.left), Image.open(args.center), Image.open(args.right)]
    proprio = np.zeros(10, dtype=np.float32)
    policy = AICXVLAPolicy(args.checkpoint)
    actions = policy.predict(images, proprio, args.instruction)
    print(f"predicted action chunk shape: {actions.shape}")
    print(f"first action (pos+quat): {actions[0]}")


if __name__ == "__main__":
    main()
