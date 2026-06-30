from __future__ import annotations
from .environment import ACTION_DIRECTIONS
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .config import Config
from .discretization import StateDiscretizer


class SarsaAgent:
    """On-policy TD controller matching the SARSA algorithm in lecture slide 29."""

    def __init__(self, config: Config, seed: int | None = None) -> None:
        self.config = config
        self.discretizer = StateDiscretizer(config)
        self.rng = np.random.default_rng(config.seed if seed is None else seed)
        self.q: NDArray[np.float32] = np.zeros(
            (self.discretizer.n_states, ACTION_DIRECTIONS.shape[0]), dtype=np.float32
        )

    def choose_action(self, state_index: int, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.q.shape[1]))
        values = self.q[state_index]
        best = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best))

    def update(
        self,
        state_index: int,
        action: int,
        reward: float,
        next_state_index: int,
        next_action: int,
        terminal: bool,
    ) -> float:
        """Q(s,a) <- Q(s,a)+alpha[r+gamma Q(s',a')-Q(s,a)]."""
        old_value = float(self.q[state_index, action])
        bootstrap = 0.0 if terminal else self.config.gamma * float(
            self.q[next_state_index, next_action]
        )
        td_error = reward + bootstrap - old_value
        self.q[state_index, action] = old_value + self.config.alpha * td_error
        return float(td_error)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, q=self.q)

    def load(self, path: str | Path) -> None:
        q = np.load(path)["q"]
        if q.shape != self.q.shape:
            raise ValueError(f"Q-table has shape {q.shape}; expected {self.q.shape}.")
        self.q = q.astype(np.float32, copy=True)

