from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class DQNValueNetwork(nn.Module):
    """Wrapper around the quantum (or classical) model exposing DQN utilities."""

    def __init__(
        self,
        env,
        model: nn.Module,
        *,
        learning_rate: float,
        device: torch.device,
        classic: bool = False,
        input_layer_learning_rate: float = 0.01,
        output_layer_learning_rate: float = 0.01,
    ) -> None:
        super().__init__()
        self.env = env
        self.device = device
        self.action_space = np.arange(env.action_space.n)
        self.learning_rate = learning_rate
        self.input_layer_learning_rate = input_layer_learning_rate
        self.output_layer_learning_rate = output_layer_learning_rate
        self.model = model.to(device)

        if classic:
            parameter_groups = [{"params": self.parameters()}]
        elif hasattr(self.model, "q_layers") and hasattr(self.model, "input_scale"):
            parameter_groups = [
                {"params": self.model.q_layers.parameters()},
                {"params": [self.model.input_scale], "lr": self.input_layer_learning_rate},
                {"params": [self.model.output_scale], "lr": self.output_layer_learning_rate},
            ]
        else:
            parameter_groups = [{"params": self.model.parameters()}]
        self.optimizer = torch.optim.RMSprop(parameter_groups, lr=self.learning_rate)

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(state_tensor)

    def q_values(self, state) -> torch.Tensor:
        """Return the Q-values predicted by the underlying model."""
        if isinstance(state, tuple):
            state = np.array([np.ravel(sample) for sample in state])
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        return self.model(state_tensor)

    def select_action(self, state, epsilon: float) -> int:
        """Epsilon-greedy action selection identical to the legacy agent."""
        if np.random.random() < epsilon:
            return np.random.choice(self.action_space)
        q_values = self.q_values(state)
        action = torch.argmax(q_values, dim=-1).item()
        return int(action)
