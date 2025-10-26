from __future__ import annotations

from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from .actor import Actor
from .critic import Critic
from .algorithm import EPISODE_REWARDS
from .config import DEFAULT_CONFIG
from .training import train_actor_critic
from .testing import evaluate_policy


def main() -> None:
    """Entry point that mirrors the original notebook workflow."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_env = gym.make("LunarLander-v3")
    state_size = training_env.observation_space.shape[0]
    action_size = training_env.action_space.n

    actor_policy = Actor(DEFAULT_CONFIG).to(device)
    critic_value_function = Critic(state_size, action_size).to(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("executions") / f"actor_critic_lunar_lander_{timestamp}"

    rewards, returns = train_actor_critic(
        actor_policy,
        critic_value_function,
        training_env,
        device=device,
        config=DEFAULT_CONFIG,
        episode_rewards=EPISODE_REWARDS,
        output_dir=output_dir,
    )

    np.savez(
        output_dir / "training_metrics.npz",
        rewards=np.asarray(rewards),
        returns=np.asarray([t.detach().cpu().numpy() if hasattr(t, 'detach') else t for t in returns]),
    )

    evaluation_env = gym.make("LunarLander-v3", render_mode="rgb_array")
    evaluate_policy(
        actor_policy,
        critic_value_function,
        evaluation_env,
        device=device,
        num_episodes=DEFAULT_CONFIG.num_evaluation_episodes,
        gif_path=str(output_dir / "evaluation.gif"),
    )


if __name__ == "__main__":
    main()
