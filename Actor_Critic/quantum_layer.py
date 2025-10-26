from __future__ import annotations

try:
    import pennylane as qml
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "PennyLane is required for the quantum actor. Install it with 'pip install pennylane'."
    ) from exc


def build_actor_layer(*, num_qubits: int, num_layers: int):
    """Construct the parameterised quantum layer used by the actor network."""
    device = qml.device("lightning.qubit", wires=num_qubits)
    trainable_shapes = {
        "x_weights": (num_layers, num_qubits),
        "z_weights": (num_layers, num_qubits),
    }

    @qml.qnode(device, interface="torch")
    def circuit(inputs, x_weights, z_weights):
        for layer_index in range(num_layers):
            if layer_index == 0:
                for wire in range(num_qubits):
                    qml.RX(inputs[wire], wires=wire)
            for wire, x_weight in enumerate(x_weights[layer_index]):
                qml.RX(x_weight, wires=wire)
            for wire, z_weight in enumerate(z_weights[layer_index]):
                qml.RZ(z_weight, wires=wire)
            for wire in range(num_qubits - 1):
                qml.CNOT(wires=[wire, (wire + 1) % num_qubits])
        return [
            qml.expval(qml.PauliZ(6) @ qml.PauliZ(7)),
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(2)),
            qml.expval(qml.PauliZ(1) @ qml.PauliZ(3)),
            qml.expval(qml.PauliZ(4) @ qml.PauliZ(5)),
        ]

    return qml.qnn.TorchLayer(circuit, trainable_shapes)
