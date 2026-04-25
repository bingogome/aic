"""Eval / inference wrapper for a fine-tuned X-VLA checkpoint on aic.

Loads the checkpoint via X-VLA's native API, predicts a chunk of actions for
a single observation, and converts the 20D X-VLA output back to aic 7D TCP
(pos + quat).
"""

from __future__ import annotations

import argparse
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


class AICXVLAPolicy:
    def __init__(
        self, checkpoint: str, device: str = "cuda", domain_id: int = 0, steps: int = 10
    ):
        self.model = XVLA.from_pretrained(checkpoint).to(device).eval()
        self.processor = XVLAProcessor.from_pretrained(checkpoint)
        self.device = device
        self.domain_id = domain_id
        self.steps = steps

    @torch.no_grad()
    def predict(
        self,
        images: Sequence[Image.Image],  # length 3, order: left, center, right
        proprio_10d: np.ndarray,  # [10] = pos(3) + rot6d(6) + gripper(1)
        instruction: str = DEFAULT_INSTRUCTION,
    ) -> np.ndarray:
        # Pad proprio to 20D to match real_action_dim.
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

        actions = self.model.generate_actions(
            image_input=image_input,
            image_mask=image_mask,
            proprio=proprio_t,
            domain_id=domain_id,
            steps=self.steps,
            **lang,
        )  # [1, num_actions, 20]
        return self._xvla_to_aic_actions(actions[0].cpu().numpy())

    @staticmethod
    def _xvla_to_aic_actions(act20: np.ndarray) -> np.ndarray:
        """[T, 20] -> [T, 7] = pos(3) + quat_xyzw(4)."""
        pos = act20[:, 0:3]
        rot6d = act20[:, 3:9]
        quat = rotate6d_to_quat(rot6d, scalar_first=False)
        return np.concatenate([pos, quat], axis=-1).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(
        description="One-shot inference smoke test for fine-tuned X-VLA on aic."
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--left", required=True)
    p.add_argument("--center", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    args = p.parse_args()

    images = [Image.open(args.left), Image.open(args.center), Image.open(args.right)]
    proprio = np.zeros(
        10, dtype=np.float32
    )  # placeholder — caller provides real proprio
    policy = AICXVLAPolicy(args.checkpoint)
    actions = policy.predict(images, proprio, args.instruction)
    print(f"predicted action chunk shape: {actions.shape}")
    print(f"first action (pos+quat): {actions[0]}")


if __name__ == "__main__":
    main()
