from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from .config import Config

State = NDArray[np.float64]


class Action(IntEnum):
    COAST = 0
    PLUS_X = 1
    MINUS_X = 2
    PLUS_Y = 3
    MINUS_Y = 4


ACTION_DIRECTIONS = np.array(
    [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
)


@dataclass(frozen=True)
class StepInfo:
    success: bool
    collision: bool
    escaped: bool
    unsafe: bool
    unsafe_entry: bool
    distance: float
    speed: float
    fuel: int


class RendezvousEnv:
    """Planar target-centred Clohessy-Wiltshire/Hill dynamics.

    State = [x, y, vx, vy]. The model is deliberately simplified for an
    educational RL experiment; it is not a flight-certified simulator.
    """

    n_actions = len(Action)

    def __init__(self, config: Config, seed: int | None = None) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed if seed is None else seed)
        self.state = np.zeros(4, dtype=np.float64)
        self.steps = 0

    def reset(self, state: State | None = None) -> State:
        if state is None:
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            radius = self.rng.uniform(
                self.config.initial_radius_min, self.config.initial_radius_max
            )
            position = radius * np.array([np.cos(angle), np.sin(angle)])
            velocity = self.rng.uniform(
                -self.config.initial_speed, self.config.initial_speed, size=2
            )
            self.state = np.concatenate((position, velocity))
        else:
            self.state = np.asarray(state, dtype=np.float64).copy()
            if self.state.shape != (4,):
                raise ValueError("Initial state must have shape (4,).")
        self.steps = 0
        return self.state.copy()

    def step(self, action: int) -> tuple[State, float, bool, StepInfo]:
        if action not in range(self.n_actions):
            raise ValueError(f"Action must be in [0, {self.n_actions - 1}].")

        cfg = self.config
        x, y, vx, vy = self.state
        thrust = cfg.thrust_acceleration * ACTION_DIRECTIONS[action]
        n = cfg.mean_motion

        # Planar Hill/CW equations: xddot=3n²x+2n*ydot+ux,
        # yddot=-2n*xdot+uy. Semi-implicit Euler advances velocity first.
        acceleration = np.array(
            [3.0 * n * n * x + 2.0 * n * vy + thrust[0],
             -2.0 * n * vx + thrust[1]]
        )
        velocity = np.array([vx, vy]) + cfg.dt * acceleration
        position = np.array([x, y]) + cfg.dt * velocity
        next_state = np.concatenate((position, velocity))

        old_distance = float(np.hypot(x, y))
        distance = float(np.linalg.norm(position))
        speed = float(np.linalg.norm(velocity))
        fuel = int(action != Action.COAST)
        entered_keep_out = old_distance >= cfg.keep_out_radius > distance
        unsafe = distance < cfg.keep_out_radius and speed > cfg.safe_entry_speed
        success = distance <= cfg.docking_radius and speed <= cfg.docking_speed
        collision = distance <= cfg.collision_radius and not success
        escaped = distance >= cfg.workspace_radius

        reward = (
            cfg.progress_weight * (old_distance - distance)
            - cfg.speed_weight * speed * (1.0 + 3.0 * (distance < cfg.keep_out_radius))
            - cfg.step_penalty
            - cfg.fuel_penalty * fuel
        )
        if unsafe:
            reward -= cfg.safety_penalty
        if success:
            reward += cfg.success_reward
        elif collision:
            reward -= cfg.collision_penalty
        elif escaped:
            reward -= cfg.escape_penalty

        self.state = next_state
        self.steps += 1
        time_limit = self.steps >= cfg.max_steps
        done = success or collision or escaped or time_limit
        info = StepInfo(
            success=success,
            collision=collision,
            escaped=escaped,
            unsafe=unsafe,
            unsafe_entry=entered_keep_out and unsafe,
            distance=distance,
            speed=speed,
            fuel=fuel,
        )
        return next_state.copy(), float(reward), done, info

