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

"""Abstract VLA backend interface for RLT.

All VLA backends must accept AIC ROS Observation messages and return:
  - get_embeddings(obs)  -> torch.Tensor (1, num_tokens, embed_dim) on device
  - get_action_chunk(obs) -> np.ndarray (chunk_length, action_dim)

Backends also expose:
  - embed_dim:   int   (per-token embedding dimensionality)
  - num_tokens:  int   (number of tokens per observation)

These dimensions are used to construct RLTokenConfig at init time.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch


class VLABackend(ABC):
    """Abstract base class for VLA backends used by RLT.

    Subclasses wrap a specific VLA model (XVLA, Pi0.5, ACT, …) and expose a
    uniform interface so RunRLT and the training scripts are backend-agnostic.
    """

    # Subclasses must set these after the model is loaded.
    embed_dim: int
    num_tokens: int

    @abstractmethod
    def get_embeddings(self, obs) -> torch.Tensor:
        """Extract internal VLA embeddings from a ROS Observation.

        Args:
            obs: aic_model_interfaces.msg.Observation

        Returns:
            torch.Tensor of shape (1, num_tokens, embed_dim) on the backend's device.
        """

    @abstractmethod
    def get_action_chunk(self, obs) -> np.ndarray:
        """Run VLA inference to produce the reference action chunk.

        Args:
            obs: aic_model_interfaces.msg.Observation

        Returns:
            np.ndarray of shape (chunk_length, action_dim), float32.
        """

    def get_embeddings_and_actions(self, obs) -> tuple:
        """Get embeddings and action chunk in one call.

        Default: two separate calls.  Override when a single forward pass
        returns both (e.g. Pi0.5 — avoids redundant computation).

        Returns:
            (embeddings, action_chunk):
                embeddings:   torch.Tensor (1, num_tokens, embed_dim)
                action_chunk: np.ndarray   (chunk_length, action_dim)
        """
        return self.get_embeddings(obs), self.get_action_chunk(obs)
