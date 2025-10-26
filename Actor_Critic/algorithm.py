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

# Global buffer shared between the training loop and the plotting utility.
EPISODE_REWARDS: List[float] = []


def compute_discounted_returns(
    next_state_value: torch.Tensor,
    reward_tensors: Iterable[torch.Tensor],
    mask_tensors: Iterable[torch.Tensor],
    discount_factor: float = 0.99,
) -> list[torch.Tensor]:
    """Compute discounted returns in reverse order, matching the notebook behaviour."""
    running_return = next_state_value
    discounted_returns: list[torch.Tensor] = []
    for step in reversed(range(len(reward_tensors))):
        running_return = reward_tensors[step] + discount_factor * running_return * mask_tensors[step]
        discounted_returns.insert(0, running_return)
    return discounted_returns


def plot_episode_rewards(show_result: bool = False) -> None:
    """Live plotting of episodic rewards, helpful during long training runs."""
    plt.figure(1)
    rewards_tensor = torch.tensor(EPISODE_REWARDS, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("")
    if len(rewards_tensor) > 0:
        plt.plot(rewards_tensor.numpy())
    if len(rewards_tensor) >= 10:
        rolling_mean = rewards_tensor.unfold(0, 10, 1).mean(1).view(-1)
        rolling_mean = torch.cat((torch.zeros(9), rolling_mean))
        plt.plot(rolling_mean.numpy())
    plt.pause(0.001)
    if RUNNING_IN_IPYTHON:
        if display is not None:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
