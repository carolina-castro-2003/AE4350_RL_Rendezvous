from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Config:
    """All physical, reward, and learning parameters in one reproducible object."""

    # Reproducibility and physical model
    seed: int = 7
    dt: float = 10.0
    mean_motion: float = 0.0011
    thrust_acceleration: float = 0.002
    max_steps: int = 450
    episodes: int = 8_000

    # State discretisation
    position_edges: tuple[float, ...] = (
        -70, -35, -25, -18, -12, -8, -5, -3, -1.5, 0,
        1.5, 3, 5, 8, 12, 18, 25, 35, 70,
    )
    velocity_edges: tuple[float, ...] = (
        -0.30, -0.18, -0.12, -0.08, -0.05, -0.03, -0.015, 0,
        0.015, 0.03, 0.05, 0.08, 0.12, 0.18, 0.30,
    )

    # SARSA learning parameters
    alpha: float = 0.35
    gamma: float = 0.995
    epsilon_start: float = 1.0
    epsilon_end: float = 0.03
    epsilon_decay: float = 0.999

    # Initial conditions and terminal regions
    initial_radius_min: float = 18.0
    initial_radius_max: float = 35.0
    initial_speed: float = 0.025
    workspace_radius: float = 70.0
    docking_radius: float = 1.5
    docking_speed: float = 0.035
    keep_out_radius: float = 8.0
    safe_entry_speed: float = 0.060
    collision_radius: float = 0.50

    # Reward parameters
    progress_weight: float = 4.0
    distance_weight: float = 0.0
    speed_weight: float = 1.5
    step_penalty: float = 0.03
    fuel_penalty: float = 0.12
    safety_penalty: float = 10.0
    success_reward: float = 250.0
    collision_penalty: float = 250.0
    escape_penalty: float = 150.0
    timeout_penalty: float = 0.0

    # Evaluation and output
    evaluation_episodes: int = 250
    evaluation_seed: int = 10_000
    smoothing_window: int = 100
    results_dir: str = "results"

    def changed(self, **kwargs: Any) -> "Config":
        return replace(self, **kwargs)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["position_edges"] = list(self.position_edges)
        data["velocity_edges"] = list(self.velocity_edges)
        return data

    @property
    def output_path(self) -> Path:
        return Path(self.results_dir)