from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.optim as optim

import matplotlib.pyplot as plt

from . import algorithm
from .config import TrainingConfig, DEFAULT_CONFIG


def train_actor_critic(
    actor_network,
    critic_network,
    training_env,
    device: torch.device,
    config: TrainingConfig = DEFAULT_CONFIG,
    episode_rewards: Optional[List[float]] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[Sequence[float], Sequence[float]]:
    """Run the original training loop for the requested number of iterations."""
    reward_history = episode_rewards if episode_rewards is not None else algorithm.EPISODE_REWARDS
    actor_optimizer = optim.Adam(
        actor_network.parameters(),
        lr=config.actor_learning_rate,
        amsgrad=config.actor_amsgrad_enabled,
    )
    critic_optimizer = optim.Adam(
        critic_network.parameters(),
        lr=config.critic_learning_rate,
        amsgrad=config.critic_amsgrad_enabled,
    )
    episode_reward_history: list[float] = []
    episode_return_history: list[float] = []

    for iteration in range(config.num_training_iterations):
        state, _ = training_env.reset()
        training_env.reset()  # Preserve redundant reset from the notebook for parity.
        log_probability_buffer: list[torch.Tensor] = []
        value_buffer: list[torch.Tensor] = []
        reward_buffer: list[torch.Tensor] = []
        mask_buffer: list[torch.Tensor] = []
        episode_reward = 0.0
        episode_finished = False

        while not episode_finished:
            # Convert the environment observation into a Torch tensor on the chosen device.
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
            action_distribution = actor_network(state_tensor)
            state_value = critic_network(state_tensor)
            action = action_distribution.sample()
            next_state, reward, terminated, truncated, _ = training_env.step(action.cpu().numpy())
            episode_finished = terminated or truncated
            episode_reward += reward
            log_probability_buffer.append(action_distribution.log_prob(action).unsqueeze(0))
            value_buffer.append(state_value)
            reward_buffer.append(torch.tensor([reward], dtype=torch.float32, device=device))
            # Masks keep track of whether the episode terminated to stop bootstrapping.
            mask_buffer.append(torch.tensor([1 - episode_finished], dtype=torch.float32, device=device))
            state = next_state

        print(f"Iteration: {iteration}, Score: {episode_reward}")
        reward_history.append(episode_reward)
        algorithm.plot_episode_rewards()
        episode_reward_history.append(episode_reward)

        next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32, device=device)
        next_state_value = critic_network(next_state_tensor)
        discounted_returns = algorithm.compute_discounted_returns(
            next_state_value,
            reward_buffer,
            mask_buffer,
            discount_factor=config.discount_factor,
        )
        episode_return_history.append(sum(discounted_returns))

        log_probabilities = torch.cat(log_probability_buffer)
        returns_tensor = torch.cat(discounted_returns).detach()
        values_tensor = torch.cat(value_buffer)
        # Advantage estimation: how much better the sampled actions performed than the baseline.
        advantages = returns_tensor - values_tensor

        actor_loss = -(log_probabilities * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()

        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        actor_optimizer.step()
        critic_optimizer.step()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        actor_path = output_dir / Path(config.actor_checkpoint_path).name
        critic_path = output_dir / Path(config.critic_checkpoint_path).name
    else:
        actor_path = Path(config.actor_checkpoint_path)
        critic_path = Path(config.critic_checkpoint_path)

    torch.save(actor_network.state_dict(), actor_path)
    torch.save(critic_network.state_dict(), critic_path)
    training_env.close()
    print("Complete")
    algorithm.plot_episode_rewards(show_result=True)
    if output_dir is not None:
        figure = plt.figure(1)
        figure.savefig(output_dir / "training_rewards.png", bbox_inches="tight")
        plt.close(figure)
    else:
        plt.ioff()
        plt.show()
    return episode_reward_history, episode_return_history
