from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """Primary hyper-parameters and output paths for the actor-critic pipeline."""

    # Learning rates used by the actor and critic optimisers.
    actor_learning_rate: float = 0.005
    critic_learning_rate: float = 0.005

    # Toggle AMSGrad (Adam variant) to improve numerical stability.
    actor_amsgrad_enabled: bool = True
    critic_amsgrad_enabled: bool = True

    # Discount factor applied when computing returns.
    discount_factor: float = 0.99

    # Total optimisation steps executed during training.
    num_training_iterations: int = 2000

    # Default locations where the trained weights are persisted.
    actor_checkpoint_path: str = "Quantum_Actor_Model.pt"
    critic_checkpoint_path: str = "Quantum_Critic_Model.pt"

    # Evaluation loop configuration and render output path.
    num_evaluation_episodes: int = 5
    evaluation_gif_path: str = "./training_actor_critic.gif"

    # Quantum policy architecture (mirrors the original notebook setup).
    num_qubits: int = 8
    num_actions: int = 4
    num_quantum_layers: int = 5


# Default configuration consumed by the CLI entry point and the game integration.
DEFAULT_CONFIG = TrainingConfig()
