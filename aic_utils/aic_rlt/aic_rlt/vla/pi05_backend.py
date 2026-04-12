#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""Pi0.5 (openpi) VLA backend for RLT.

Wraps Pi05VLA so RunRLT can use it via the VLABackend interface.
Pi0.5 accepts ROS Observation directly and supports an efficient single
forward pass for both embeddings and actions.

Requires:
  - openpi repository at /home/yifeng/workspace/openpi (or override OPENPI_ROOT)
  - Downloaded pi05_base checkpoint
  - JAX installed (via openpi venv)
"""

import logging

import numpy as np
import torch

from .base import VLABackend

logger = logging.getLogger(__name__)


class Pi05Backend(VLABackend):
    """Pi0.5 (PaliGemma-based) VLA backend.

    Uses a single forward pass for both embedding extraction and action
    generation — more efficient than calling them separately.

    Parameters
    ----------
    checkpoint_dir: path to the downloaded pi05_base checkpoint directory
    device:         torch device (Pi0.5 runs in JAX but tensors are moved here)
    chunk_length:   number of actions to return per chunk
    openpi_config:  openpi config name (default "pi05_aloha")
    instruction:    language prompt for the task
    """

    def __init__(
        self,
        checkpoint_dir: str,
        device: torch.device,
        chunk_length: int = 10,
        openpi_config: str = "pi05_aloha",
        instruction: str = "insert the cable into the port",
    ):
        # Import lazily to avoid mandatory JAX dependency when using XVLA
        from aic_rlt.vla_pi05 import Pi05VLA, Pi05Config

        cfg = Pi05Config(
            checkpoint_dir=checkpoint_dir,
            openpi_config_name=openpi_config,
            chunk_length=chunk_length,
            default_prompt=instruction,
        )
        self._vla = Pi05VLA(cfg, device=device)
        self._vla._ensure_loaded()
        self.device = device
        self.chunk_length = chunk_length

        # Dimensions come from the model after loading
        self.embed_dim: int = self._vla.embed_dim
        self.num_tokens: int = self._vla.num_tokens
        logger.info(
            "Pi05Backend ready: num_tokens=%d, embed_dim=%d", self.num_tokens, self.embed_dim
        )

    # ------------------------------------------------------------------
    # VLABackend interface
    # ------------------------------------------------------------------

    def get_embeddings(self, obs) -> torch.Tensor:
        """(1, num_tokens, embed_dim) on device."""
        return self._vla.get_embeddings(obs)  # already (1, N, D) on device

    def get_action_chunk(self, obs) -> np.ndarray:
        """(chunk_length, action_dim) float32."""
        return self._vla.get_action_chunk(obs)

    def get_embeddings_and_actions(self, obs) -> tuple:
        """Single Pi0.5 forward pass — more efficient than two separate calls."""
        return self._vla.get_embeddings_and_actions(obs)
