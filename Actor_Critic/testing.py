from __future__ import annotations

from typing import List

import imageio
import torch


def save_episode_animation(frames: List, destination_path: str = "./lra2c.gif") -> None:
    """Persist the recorded frames as an animated GIF."""
    imageio.mimwrite(destination_path, frames)


def evaluate_policy(
    actor_network,
    critic_network,
    evaluation_env,
    device: torch.device,
    num_episodes: int,
    gif_path: str,
) -> List[float]:
    """Run evaluation episodes and mirror the notebook's reporting."""
    episode_rewards: List[float] = []
    frame_buffer: List = []

    for episode in range(num_episodes):
        done = False
        episode_reward = 0.0
        state, _ = evaluation_env.reset()

        while not done:
            frame_buffer.append(evaluation_env.render())
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
            action_distribution = actor_network(state_tensor)
            critic_network(state_tensor)
            action = action_distribution.sample()
            obs, reward, term, trunc, _ = evaluation_env.step(action.cpu().numpy())
            done = term or trunc
            episode_reward += reward
            if not done:
                state = obs

        episode_rewards.append(episode_reward)
        print(f"Episode {episode}: Reward {episode_reward}")

    save_episode_animation(frame_buffer, destination_path=gif_path)
    evaluation_env.close()
    return episode_rewards
