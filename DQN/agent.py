from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from . import algorithm
from .config import DQNConfig
from .replay_buffer import ExperienceReplayBuffer
from .value_network import DQNValueNetwork


def _ensure_state_array(state) -> np.ndarray:
    """Convert Gymnasium reset/step returns into a 1D NumPy array."""
    if isinstance(state, tuple):
        state = state[0]
    return np.asarray(state, dtype=np.float32)


@dataclass
class TrainingHistory:
    rewards: List[float]
    mean_rewards: List[float]
    losses: List[float]
    sync_episodes: List[int]


class DeepQLearningAgent:
    """Faithful port of the legacy CartPole DQN training loop."""

    def __init__(
        self,
        env,
        policy_network: DQNValueNetwork,
        target_network: DQNValueNetwork,
        replay_buffer: ExperienceReplayBuffer,
        config: DQNConfig,
        *,
        device: torch.device,
        reward_threshold: float,
    ) -> None:
        self.env = env
        self.policy_network = policy_network
        self.target_network = target_network
        self.replay_buffer = replay_buffer
        self.config = config
        self.device = device

        self.epsilon = config.epsilon_start
        self.batch_size = config.batch_size
        self.reward_window = config.reward_window
        self.gamma = config.gamma
        self.reward_threshold = reward_threshold

        self.training_rewards: List[float] = []
        self.mean_training_rewards: List[float] = []
        self.training_losses: List[float] = []
        self.sync_episodes: List[int] = []
        self.loss_accumulator: List[float] = []
        self.step_counter = 0
        self.current_state = _ensure_state_array(self.env.reset())
        self.total_reward = 0.0

        self.target_network.load_state_dict(self.policy_network.state_dict())

    def _take_step(self, epsilon: float, mode: str = "train") -> bool:
        if mode == "explore":
            action = self.env.action_space.sample()
        else:
            action = self.policy_network.select_action(self.current_state, epsilon)
            self.step_counter += 1

        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        self.total_reward += reward
        self.replay_buffer.append(self.current_state, action, reward, done, _ensure_state_array(next_state))
        self.current_state = _ensure_state_array(next_state)
        if done:
            self.current_state = _ensure_state_array(self.env.reset())
        return done

    def _calculate_loss(self, batch) -> torch.Tensor:
        states, actions, rewards, dones, next_states = [np.array(values) for values in batch]

        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(-1)
        dones_tensor = torch.as_tensor(dones, dtype=torch.bool, device=self.device)

        state_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        q_values = self.policy_network.q_values(state_tensor)
        current_q = torch.gather(q_values, dim=1, index=actions_tensor)

        next_state_tensor = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        next_q = torch.max(self.target_network.q_values(next_state_tensor), dim=-1)[0].detach()
        next_q = next_q.masked_fill(dones_tensor, 0.0)
        expected_q = rewards_tensor + self.gamma * next_q

        return torch.nn.MSELoss()(current_q, expected_q.unsqueeze(-1))

    def _update_network(self) -> None:
        self.policy_network.optimizer.zero_grad()
        batch = self.replay_buffer.sample_batch(batch_size=self.batch_size)
        loss = self._calculate_loss(batch)
        loss.backward()
        self.policy_network.optimizer.step()
        if self.policy_network.device.type == "cuda":
            self.loss_accumulator.append(loss.detach().cpu().numpy())
        else:
            self.loss_accumulator.append(loss.detach().numpy())

    def train(self) -> TrainingHistory:
        print("Filling replay buffer...")
        while self.replay_buffer.burn_in_ratio() < 1:
            self._take_step(self.epsilon, mode="explore")

        episode = 0
        print("Training...")
        while True:
            self.current_state = _ensure_state_array(self.env.reset())
            self.total_reward = 0.0
            done = False

            while not done:
                done = self._take_step(self.epsilon, mode="train")

                if self.step_counter % self.config.policy_update_frequency == 0:
                    self._update_network()

                if self.step_counter % self.config.target_sync_frequency == 0:
                    self.target_network.load_state_dict(self.policy_network.state_dict())
                    self.sync_episodes.append(episode)

            episode += 1
            self.training_rewards.append(self.total_reward)
            self.training_losses.append(sum(self.loss_accumulator))
            self.loss_accumulator.clear()
            mean_reward = float(np.mean(self.training_rewards[-self.reward_window :]))
            self.mean_training_rewards.append(mean_reward)
            algorithm.EPISODE_REWARDS.append(self.total_reward)
            algorithm.plot_training_rewards()

            print(
                f"Episode {episode:d} | Mean reward(last {self.reward_window}) {mean_reward:.2f} | "
                f"Epsilon {self.epsilon:.3f}",
                end="\r",
            )

            if episode >= self.config.max_episodes:
                print("\nEpisode limit reached.")
                break

            if mean_reward >= self.reward_threshold:
                print(f"\nEnvironment solved in {episode} episodes!")
                break

            self.epsilon = max(self.epsilon * self.config.epsilon_decay, self.config.epsilon_min)

        self.env.close()
        algorithm.plot_training_rewards(show_result=True)
        return TrainingHistory(
            rewards=self.training_rewards,
            mean_rewards=self.mean_training_rewards,
            losses=self.training_losses,
            sync_episodes=self.sync_episodes,
        )

    def save_reward_plot(self, path: str | None = None) -> None:
        import matplotlib.pyplot as plt

        if not self.training_rewards:
            return
        figure = plt.figure(figsize=(12, 8))
        plt.plot(self.training_rewards, label="Rewards")
        plt.plot(self.mean_training_rewards, label=f"Mean reward (last {self.reward_window})")
        plt.axhline(self.reward_threshold, color="r", label="Reward threshold")
        coefficients = np.polyfit(range(len(self.mean_training_rewards)), self.mean_training_rewards, 1)
        trend = [coefficients[0] * x + coefficients[1] for x in range(len(self.mean_training_rewards))]
        plt.plot(trend, label="Mean reward trend")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.legend(loc="upper left")
        if path:
            figure.savefig(path, bbox_inches="tight")
            plt.close(figure)
        else:
            plt.show()
