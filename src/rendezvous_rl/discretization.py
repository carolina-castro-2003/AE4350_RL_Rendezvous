from __future__ import annotations

import numpy as np

from .config import Config
from .environment import State


class StateDiscretizer:
    """Map continuous [x,y,vx,vy] to a finite lookup-table index."""

    def __init__(self, config: Config) -> None:
        self.position_edges = np.asarray(config.position_edges, dtype=float)
        self.velocity_edges = np.asarray(config.velocity_edges, dtype=float)
        if not (np.all(np.diff(self.position_edges) > 0) and
                np.all(np.diff(self.velocity_edges) > 0)):
            raise ValueError("Discretisation edges must be strictly increasing.")
        self.n_position_bins = self.position_edges.size - 1
        self.n_velocity_bins = self.velocity_edges.size - 1
        self.shape = (
            self.n_position_bins, self.n_position_bins,
            self.n_velocity_bins, self.n_velocity_bins,
        )
        self.n_states = int(np.prod(self.shape))

    @staticmethod
    def _bin(value: float, edges: np.ndarray) -> int:
        # Clip out-of-range values to the end bins; escape termination still
        # comes from the physical environment, not from this representation.
        return int(np.clip(np.digitize(value, edges[1:-1]), 0, edges.size - 2))

    def encode(self, state: State) -> int:
        bins = (
            self._bin(float(state[0]), self.position_edges),
            self._bin(float(state[1]), self.position_edges),
            self._bin(float(state[2]), self.velocity_edges),
            self._bin(float(state[3]), self.velocity_edges),
        )
        return int(np.ravel_multi_index(bins, self.shape))

