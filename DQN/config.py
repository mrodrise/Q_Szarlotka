from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class EnvironmentSettings:
    """Environment-specific configuration for the quantum DQN."""

    env_id: str
    num_layers: int
    num_qubits: int
    num_actions: int
    feature_groups: Tuple[Tuple[int, ...], ...]
    feature_bounds: Tuple[Tuple[float, float], ...]
    directional_features: Tuple[bool, ...]
    directional_threshold: float
    reward_threshold: float


@dataclass(frozen=True)
class DQNConfig:
    """Centralised hyperparameters mirroring the legacy CartPole DQN setup."""

    # Optimisation: RMSProp learning rate for quantum layers and classical scalers.
    learning_rate: float = 0.01
    input_layer_learning_rate: float = 0.01
    output_layer_learning_rate: float = 0.01
    gamma: float = 0.99

    # Exploration schedule for epsilon-greedy policy.
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.99
    epsilon_min: float = 0.01

    # Replay buffer sizing and mini-batch size.
    replay_memory_capacity: int = 10_000
    replay_burn_in: int = 1_000
    batch_size: int = 64

    # Target network update cadence (steps between SGD updates and target syncs).
    policy_update_frequency: int = 30
    target_sync_frequency: int = 100

    # Training loop control: hard cap on episodes and window for moving average.
    max_episodes: int = 3
    reward_window: int = 100

    # Which PennyLane backend to use when instantiating circuits.
    quantum_device: str = "lightning.qubit"

    # Environment registry (CartPole only for the legacy setup)
    environment_settings: Dict[str, EnvironmentSettings] = field(default_factory=dict)
    default_environment: str = "cartpole"

    # Output artefacts
    reward_plot_name: str = "training_rewards.png"
    metrics_filename: str = "training_metrics.npz"
    checkpoint_name: str = "dqn_quantum_policy.pt"

    def get_environment(self, name: str) -> EnvironmentSettings:
        key = name.lower()
        if key not in self.environment_settings:
            available = ", ".join(sorted(self.environment_settings.keys()))
            raise ValueError(f"Unknown environment '{name}'. Available: {available}")
        return self.environment_settings[key]


DEFAULT_CONFIG = DQNConfig(
    environment_settings={
        "lunarlander": EnvironmentSettings(
            env_id="LunarLander-v3",
            num_layers=5,
            num_qubits=5,
            num_actions=4,
            # Group state indices into qubits to match the training notebook encoding.
            feature_groups=(
                (0, 1),
                (2, 3),
                (4,),
                (5,),
                (6, 7),
            ),
            # Ranges used to normalise each observation before mapping to rotations.
            feature_bounds=(
                (-1.5, 1.5),
                (-1.5, 1.5),
                (-5.0, 5.0),
                (-5.0, 5.0),
                (-3.141592653589793, 3.141592653589793),
                (-5.0, 5.0),
                (0.0, 1.0),
                (0.0, 1.0),
            ),
            # Leg-contact sensors are directional (0/1) while the rest are continuous.
            directional_features=(
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
            ),
            directional_threshold=0.5,
            reward_threshold=200.0,
        ),
        "cartpole": EnvironmentSettings(
            env_id="CartPole-v1",
            num_layers=5,
            num_qubits=4,
            num_actions=2,
            # Each qubit processes one of the four CartPole observation components.
            feature_groups=(
                (0,),
                (1,),
                (2,),
                (3,),
            ),
            # Bounds used during feature scaling for the quantum encoder.
            feature_bounds=(
                (-2.4, 2.4),      # cart position
                (-3.0, 3.0),      # cart velocity
                (-0.418, 0.418),  # pole angle
                (-3.5, 3.5),      # pole angular velocity
            ),
            directional_features=(
                False,
                False,
                False,
                False,
            ),
            directional_threshold=0.5,
            reward_threshold=475.0,
        ),
    },
)
