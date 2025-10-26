from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import imageio
import numpy as np
import torch
import gymnasium as gym

from .agent import DeepQLearningAgent
from .config import DEFAULT_CONFIG, DQNConfig
from .training import train_dqn_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a quantum-enhanced DQN agent.")
    parser.add_argument(
        "--environment",
        required=True,
        choices=sorted(DEFAULT_CONFIG.environment_settings.keys()),
        help="Select which environment to train: 'cartpole' or 'lunarlander'.",
    )
    return parser.parse_args()


def save_training_artifacts(
    agent: DeepQLearningAgent,
    metrics: dict[str, np.ndarray],
    config: DQNConfig,
    output_directory: Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / config.checkpoint_name
    torch.save(agent.policy_network.state_dict(), checkpoint_path)

    metrics_path = output_directory / config.metrics_filename
    np.savez(metrics_path, **metrics)

    reward_plot_path = output_directory / config.reward_plot_name
    agent.save_reward_plot(path=str(reward_plot_path))

    print(f"Artifacts saved under {output_directory}")
    print(f"- Model: {checkpoint_path}")
    print(f"- Rewards plot: {reward_plot_path}")
    print(f"- Metrics: {metrics_path}")


def generate_evaluation_gif(
    agent: DeepQLearningAgent,
    env_settings,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    env = gym.make(env_settings.env_id, render_mode="rgb_array")
    state, _ = env.reset()
    state = np.asarray(state, dtype=np.float32)
    frames = []
    done = False
    while not done:
        frames.append(env.render())
        action = agent.policy_network.select_action(state, epsilon=0.0)
        state, _, terminated, truncated, _ = env.step(action)
        state = np.asarray(state, dtype=np.float32)
        done = terminated or truncated
        if done:
            frames.append(env.render())
    env.close()
    imageio.mimsave(output_path, frames, fps=30)


def main() -> None:
    args = parse_args()
    env_name = args.environment.lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("executions") / f"dqn_{env_name}_{timestamp}"
    agent, metrics = train_dqn_agent(config=DEFAULT_CONFIG, environment=env_name)
    save_training_artifacts(agent, metrics, DEFAULT_CONFIG, output_dir)
    env_settings = DEFAULT_CONFIG.get_environment(env_name)
    generate_evaluation_gif(agent, env_settings, output_dir / "evaluation.gif")


if __name__ == "__main__":
    main()
