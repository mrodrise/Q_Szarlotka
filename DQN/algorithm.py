from __future__ import annotations

from typing import Iterable, List

import torch

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Matplotlib is required for plotting training progress. "
        "Install it with 'pip install matplotlib'."
    ) from exc


RUNNING_IN_IPYTHON = False
if "inline" in matplotlib.get_backend():
    try:
        from IPython import display  # type: ignore
    except ModuleNotFoundError:
        display = None  # type: ignore
    else:
        RUNNING_IN_IPYTHON = True

plt.ion()

# Global buffers shared with the training loop.
EPISODE_REWARDS: List[float] = []


def compute_discounted_returns(
    next_state_value: torch.Tensor,
    rewards: Iterable[torch.Tensor],
    masks: Iterable[torch.Tensor],
    discount_factor: float,
) -> list[torch.Tensor]:
    """Compute discounted returns mirroring the canonical DQN backup."""
    running_return = next_state_value
    discounted_returns: list[torch.Tensor] = []
    for step in reversed(range(len(rewards))):
        running_return = rewards[step] + discount_factor * running_return * masks[step]
        discounted_returns.insert(0, running_return)
    return discounted_returns


def plot_training_rewards(show_result: bool = False) -> None:
    """Update the live reward plot, displaying the most recent history."""
    plt.figure(1)
    rewards_tensor = torch.tensor(EPISODE_REWARDS, dtype=torch.float)
    if show_result:
        plt.title("Training complete")
    else:
        plt.clf()
        plt.title("Training progress")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    if len(rewards_tensor) > 0:
        plt.plot(rewards_tensor.numpy(), label="Episode reward")
    if len(rewards_tensor) >= 10:
        rolling_mean = rewards_tensor.unfold(0, 10, 1).mean(1).view(-1)
        rolling_mean = torch.cat((torch.zeros(9), rolling_mean))
        plt.plot(rolling_mean.numpy(), label="Rolling mean (10 episodes)")
    plt.legend(loc="upper left")
    plt.pause(0.001)
    if RUNNING_IN_IPYTHON and display is not None:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

