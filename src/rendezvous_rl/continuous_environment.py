from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import Config
from .environment import State


ContinuousAction = NDArray[np.float64]


@dataclass(frozen=True)
class ContinuousStepInfo:
    """Diagnostics returned by the continuous-thrust rendezvous environment."""

    success: bool
    collision: bool
    escaped: bool
    unsafe: bool
    unsafe_entry: bool
    distance: float
    speed: float
    fuel: float
    thrust_norm: float


class ContinuousRendezvousEnv:
    """Planar Hill-frame rendezvous environment with continuous thrust.

    This environment uses the same state, terminal conditions, and reward
    interpretation as the tabular SARSA version, but the control input is now a
    two-dimensional continuous acceleration command:

        action = [u_x, u_y],     ||action||_inf <= thrust_acceleration.

    That makes it suitable for an actor-critic controller: the actor represents
    a parameterised stochastic policy over continuous thrust vectors, and the
    critic estimates V(s) for TD learning.
    """

    action_dim = 2

    def __init__(self, config: Config, seed: int | None = None) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed if seed is None else seed)
        self.state = np.zeros(4, dtype=np.float64)
        self.steps = 0

    def reset(self, state: State | None = None) -> State:
        if state is None:
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            radius = self.rng.uniform(
                self.config.initial_radius_min,
                self.config.initial_radius_max,
            )
            position = radius * np.array([np.cos(angle), np.sin(angle)])
            velocity = self.rng.uniform(
                -self.config.initial_speed,
                self.config.initial_speed,
                size=2,
            )
            self.state = np.concatenate((position, velocity))
        else:
            self.state = np.asarray(state, dtype=np.float64).copy()
            if self.state.shape != (4,):
                raise ValueError("Initial state must have shape (4,).")
        self.steps = 0
        return self.state.copy()

    def step(
        self,
        action: ContinuousAction,
    ) -> tuple[State, float, bool, ContinuousStepInfo]:
        cfg = self.config
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != (2,):
            raise ValueError("Continuous action must have shape (2,).")

        thrust = np.clip(
            action_array,
            -cfg.thrust_acceleration,
            cfg.thrust_acceleration,
        )
        thrust_norm = float(np.linalg.norm(thrust))
        fuel = thrust_norm / (np.sqrt(2.0) * cfg.thrust_acceleration)

        x, y, vx, vy = self.state
        n = cfg.mean_motion

        acceleration = np.array(
            [
                3.0 * n * n * x + 2.0 * n * vy + thrust[0],
                -2.0 * n * vx + thrust[1],
            ],
            dtype=np.float64,
        )

        velocity = np.array([vx, vy], dtype=np.float64) + cfg.dt * acceleration
        position = np.array([x, y], dtype=np.float64) + cfg.dt * velocity
        next_state = np.concatenate((position, velocity))

        old_distance = float(np.hypot(x, y))
        distance = float(np.linalg.norm(position))
        speed = float(np.linalg.norm(velocity))

        entered_keep_out = old_distance >= cfg.keep_out_radius > distance
        unsafe = distance < cfg.keep_out_radius and speed > cfg.safe_entry_speed

        success = distance <= cfg.docking_radius and speed <= cfg.docking_speed
        collision = distance <= cfg.collision_radius and not success
        escaped = distance >= cfg.workspace_radius

        self.state = next_state
        self.steps += 1
        time_limit = self.steps >= cfg.max_steps

        reward = (
            cfg.progress_weight * (old_distance - distance)
            - cfg.distance_weight * distance
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
        elif time_limit:
            reward -= cfg.timeout_penalty

        done = success or collision or escaped or time_limit
        info = ContinuousStepInfo(
            success=success,
            collision=collision,
            escaped=escaped,
            unsafe=unsafe,
            unsafe_entry=entered_keep_out and unsafe,
            distance=distance,
            speed=speed,
            fuel=float(fuel),
            thrust_norm=thrust_norm,
        )
        return next_state.copy(), float(reward), done, info
