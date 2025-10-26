from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
import torch

from . import algorithm
from .agent import DeepQLearningAgent
from .config import DQNConfig, DEFAULT_CONFIG, EnvironmentSettings
from .quantum_network import build_quantum_model
from .replay_buffer import ExperienceReplayBuffer
from .value_network import DQNValueNetwork


def create_environment(env_id: str, seed: Optional[int] = None):
    env = gym.make(env_id)
    if seed is not None:
        env.reset(seed=seed)
    return env


def build_quantum_network(
    env,
    config: DQNConfig,
    settings: EnvironmentSettings,
    device: torch.device,
) -> DQNValueNetwork:
    quantum_model = build_quantum_model(settings, config.quantum_device)
    return DQNValueNetwork(
        env=env,
        model=quantum_model,
        learning_rate=config.learning_rate,
        device=device,
        classic=False,
        input_layer_learning_rate=config.input_layer_learning_rate,
        output_layer_learning_rate=config.output_layer_learning_rate,
    )


def train_dqn_agent(
    *,
    config: DQNConfig = DEFAULT_CONFIG,
    device: Optional[torch.device] = None,
    environment: Optional[str] = None,
) -> tuple[DeepQLearningAgent, dict[str, np.ndarray]]:
    """Assemble the agent for the selected environment, train it, and return metrics."""
    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algorithm.EPISODE_REWARDS.clear()
    env_name = (environment or config.default_environment).lower()
    env_settings = config.get_environment(env_name)
    env = create_environment(env_settings.env_id)
    replay_buffer = ExperienceReplayBuffer(
        capacity=config.replay_memory_capacity,
        burn_in=config.replay_burn_in,
    )
    policy_network = build_quantum_network(env, config, env_settings, resolved_device)
    target_network = build_quantum_network(env, config, env_settings, resolved_device)

    agent = DeepQLearningAgent(
        env=env,
        policy_network=policy_network,
        target_network=target_network,
        replay_buffer=replay_buffer,
        config=config,
        device=resolved_device,
        reward_threshold=env_settings.reward_threshold,
    )

    history = agent.train()

    metrics = {
        "rewards": np.asarray(history.rewards),
        "mean_rewards": np.asarray(history.mean_rewards),
        "losses": np.asarray(history.losses),
        "sync_episodes": np.asarray(history.sync_episodes),
    }

    return agent, metrics
