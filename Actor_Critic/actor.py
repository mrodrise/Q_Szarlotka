from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .quantum_layer import build_actor_layer
from .config import TrainingConfig, DEFAULT_CONFIG


class Actor(nn.Module):
    """Quantum-enhanced policy network returning a categorical distribution."""

    def __init__(self, config: TrainingConfig = DEFAULT_CONFIG) -> None:
        super().__init__()
        self.config = config
        self.num_qubits = config.num_qubits
        self.num_actions = config.num_actions
        # Quantum layer mirrors the notebook circuit but lets us swap parameters from the config.
        self.quantum_policy_head = build_actor_layer(
            num_qubits=self.num_qubits,
            num_layers=config.num_quantum_layers,
        )

    def forward(self, state: torch.Tensor) -> Categorical:
        """Return the action distribution for the provided environment state."""
        state = torch.atan(state)
        raw_outputs = self.quantum_policy_head(state)
        # The original training pipeline rescales the outputs into [-1, 1] before softmax.
        rescaled_outputs = -1 + (raw_outputs - raw_outputs.min()) * 2 / (
            raw_outputs.max() - raw_outputs.min()
        )
        probabilities = F.softmax(rescaled_outputs, dim=-1)
        return Categorical(probabilities)
