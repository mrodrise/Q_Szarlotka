from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    """State-value estimator mirroring the original notebook."""

    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size  # Kept for parity with notebook signature.
        self.input_layer = nn.Linear(self.state_size, 128)
        self.hidden_layer = nn.Linear(128, 256)
        self.output_layer = nn.Linear(256, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Estimate the scalar value for the provided state."""
        output = F.relu(self.input_layer(state))
        output = F.relu(self.hidden_layer(output))
        value = self.output_layer(output)
        return value
