from __future__ import annotations

import math
from typing import Sequence

try:
    import pennylane as qml
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "PennyLane is required for the quantum DQN agent. "
        "Install it with 'pip install pennylane'."
    ) from exc

import torch
import torch.nn as nn

from .config import EnvironmentSettings


def _encode_inputs(num_qubits: int, inputs) -> None:
    """Apply RX rotations that embed the (scaled) classical inputs."""
    for wire in range(num_qubits):
        qml.RX(inputs[wire], wires=wire)


def _variational_layer(num_qubits: int, y_weights, z_weights) -> None:
    """Single layer combining RY/RZ rotations and entangling CZ gates."""
    for wire, y_weight in enumerate(y_weights):
        qml.RY(y_weight, wires=wire)
    for wire, z_weight in enumerate(z_weights):
        qml.RZ(z_weight, wires=wire)
    for wire in range(num_qubits):
        qml.CZ(wires=[wire, (wire + 1) % num_qubits])


def _measure_cartpole(num_qubits: int):
    """Observable readouts matching the original CartPole quantum circuit."""
    return [
        qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
        qml.expval(qml.PauliZ(2) @ qml.PauliZ(3)),
    ]


def build_cartpole_layer(
    *,
    num_qubits: int,
    num_layers: int,
    quantum_device: str,
) -> qml.qnn.TorchLayer:
    """Create the PennyLane TorchLayer used by the legacy CartPole DQN."""
    device = qml.device(quantum_device, wires=num_qubits)
    trainable_shapes = {
        "y_weights": (num_layers, num_qubits),
        "z_weights": (num_layers, num_qubits),
    }

    @qml.qnode(device, interface="torch")
    def circuit(inputs, y_weights, z_weights):
        for layer_index in range(num_layers):
            _encode_inputs(num_qubits, inputs)
            _variational_layer(num_qubits, y_weights[layer_index], z_weights[layer_index])
        return _measure_cartpole(num_qubits)

    return qml.qnn.TorchLayer(circuit, trainable_shapes)


def build_lunar_lander_layer(
    *,
    num_qubits: int,
    num_layers: int,
    num_actions: int,
    quantum_device: str,
) -> qml.qnn.TorchLayer:
    device = qml.device(quantum_device, wires=num_qubits)
    trainable_shapes = {
        "y_weights": (num_layers, num_qubits),
        "z_weights": (num_layers, num_qubits),
    }

    @qml.qnode(device, interface="torch")
    def circuit(inputs, y_weights, z_weights):
        for layer_index in range(num_layers):
            _encode_inputs(num_qubits, inputs)
            _variational_layer(num_qubits, y_weights[layer_index], z_weights[layer_index])
        return [qml.expval(qml.PauliZ(wire)) for wire in range(num_actions)]

    return qml.qnn.TorchLayer(circuit, trainable_shapes)


class QuantumCartPoleNetwork(nn.Module):
    """Quantum circuit plus classical scaling layers mirroring the legacy code."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_qubits: int,
        num_actions: int,
        quantum_device: str,
    ) -> None:
        super().__init__()
        if num_actions > num_qubits:
            raise ValueError("Number of actions cannot exceed number of qubits.")

        self.num_qubits = num_qubits
        self.num_actions = num_actions

        self.input_scale = nn.Parameter(torch.empty(self.num_qubits))
        nn.init.normal_(self.input_scale, mean=0.0, std=1.0)

        self.output_scale = nn.Parameter(torch.empty(self.num_actions))
        nn.init.normal_(self.output_scale, mean=90.0, std=1.0)

        self.q_layers = build_cartpole_layer(
            num_qubits=self.num_qubits,
            num_layers=num_layers,
            quantum_device=quantum_device,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() == 1:
            processed = torch.atan(inputs)
            scaled_inputs = processed * self.input_scale
            outputs = self.q_layers(scaled_inputs)
            outputs = (1.0 + outputs) / 2.0
            return outputs * self.output_scale

        outputs = []
        for sample in inputs:
            processed = torch.atan(sample)
            scaled_inputs = processed * self.input_scale
            sample_output = self.q_layers(scaled_inputs)
            sample_output = (1.0 + sample_output) / 2.0
            outputs.append(sample_output * self.output_scale)
        return torch.stack(outputs, dim=0)


class QuantumValueEncoder(nn.Module):
    """Encode classical observations into qubit rotation angles for Lunar Lander."""

    def __init__(
        self,
        obs_dim: int,
        *,
        num_layers: int,
        num_qubits: int,
        num_actions: int,
        quantum_device: str,
        feature_groups: Sequence[Sequence[int]],
        feature_bounds: Sequence[tuple[float, float]] | None,
        directional_mask: Sequence[bool],
        directional_threshold: float,
    ) -> None:
        super().__init__()
        if num_actions > num_qubits:
            raise ValueError("Number of actions cannot exceed number of qubits.")
        if feature_bounds is not None and len(feature_bounds) != obs_dim:
            raise ValueError("feature_bounds must match observation dimension.")
        if len(directional_mask) != obs_dim:
            raise ValueError("directional_mask must match observation dimension.")
        if len(feature_groups) != num_qubits:
            raise ValueError("feature_groups length must equal number of qubits.")

        self.feature_bounds = feature_bounds
        self.feature_groups = feature_groups
        self.directional_mask = directional_mask
        self.directional_threshold = directional_threshold
        self.quantum_layers = build_lunar_lander_layer(
            num_qubits=num_qubits,
            num_layers=num_layers,
            num_actions=num_actions,
            quantum_device=quantum_device,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() != 1:
            raise ValueError("QuantumValueEncoder expects a 1D tensor per call.")
        encoded = self._encode(inputs)
        bounded = torch.clamp(encoded, 0.0, 2.0 * math.pi)
        return self.quantum_layers(bounded)

    def regularization_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def _encode(self, inputs: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        angles = []
        for group in self.feature_groups:
            group_angles = []
            for index in group:
                value = inputs[index]
                if self.directional_mask[index]:
                    high = torch.tensor(math.pi, device=inputs.device, dtype=inputs.dtype)
                    low = torch.tensor(0.0, device=inputs.device, dtype=inputs.dtype)
                    angle = torch.where(value >= self.directional_threshold, high, low)
                else:
                    if self.feature_bounds is None:
                        angle = value
                    else:
                        lo, hi = self.feature_bounds[index]
                        span = max(hi - lo, eps)
                        angle = (value - lo) / span * (2.0 * math.pi)
                group_angles.append(angle)
            stacked = torch.stack(group_angles)
            angles.append(torch.mean(stacked))
        return torch.stack(angles)


class HybridQuantumValueNetwork(nn.Module):
    """Wrapper that vectorises the lunar encoder for batched evaluation."""

    def __init__(self, encoder: QuantumValueEncoder) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() == 1:
            return self.encoder(inputs)
        outputs = [self.encoder(sample) for sample in inputs]
        return torch.stack(outputs, dim=0)

    def regularization_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters()).device)


def build_quantum_model(settings: EnvironmentSettings, quantum_device: str) -> nn.Module:
    env_id = settings.env_id.lower()
    if "cartpole" in env_id:
        return QuantumCartPoleNetwork(
            num_layers=settings.num_layers,
            num_qubits=settings.num_qubits,
            num_actions=settings.num_actions,
            quantum_device=quantum_device,
        )

    obs_dim = len(settings.directional_features)
    encoder = QuantumValueEncoder(
        obs_dim=obs_dim,
        num_layers=settings.num_layers,
        num_qubits=settings.num_qubits,
        num_actions=settings.num_actions,
        quantum_device=quantum_device,
        feature_groups=settings.feature_groups,
        feature_bounds=settings.feature_bounds,
        directional_mask=settings.directional_features,
        directional_threshold=settings.directional_threshold,
    )
    return HybridQuantumValueNetwork(encoder)
