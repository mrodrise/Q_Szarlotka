from __future__ import annotations

from collections import deque, namedtuple
from typing import Iterable, Iterator, Tuple

import numpy as np


Transition = namedtuple(
    "Transition",
    ["state", "action", "reward", "done", "next_state"],
)


class ExperienceReplayBuffer:
    """Fixed-size buffer that stores environment interactions for off-policy training."""

    def __init__(self, capacity: int = 50_000, burn_in: int = 10_000) -> None:
        self.capacity = capacity
        self.burn_in = burn_in
        self.memory: deque[Transition] = deque(maxlen=capacity)

    def append(self, state, action, reward, done, next_state) -> None:
        """Persist a new transition in the buffer."""
        transition = Transition(state, action, reward, done, next_state)
        self.memory.append(transition)

    def sample_batch(self, batch_size: int = 32) -> Iterator[Tuple]:
        """Return a uniformly sampled batch of transitions."""
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[idx] for idx in indices]
        # Unpack the namedtuples into grouped tuples per field.
        return zip(*batch)

    def burn_in_ratio(self) -> float:
        """Return the filled ratio relative to the requested burn-in size."""
        if self.burn_in == 0:
            return 1.0
        return len(self.memory) / self.burn_in

    def __len__(self) -> int:
        return len(self.memory)
