from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from .actor_critic import LinearGaussianActorCritic
from .config import Config
from .continuous_environment import ContinuousRendezvousEnv
from .environment import State


@dataclass
class ContinuousEpisode:
    states: NDArray[np.float64]
    actions: NDArray[np.float64]
    rewards: NDArray[np.float64]
    success: bool
    collision: bool
    escaped: bool
    fuel: float
    safety_violations: int
    final_distance: float
    final_speed: float


@dataclass(frozen=True)
class ContinuousEvaluationMetrics:
    success_rate: float
    collision_rate: float
    escape_rate: float
    mean_final_distance: float
    mean_final_speed: float
    mean_fuel: float
    mean_safety_violations: float
    mean_return: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def run_actor_critic_episode(
    agent: LinearGaussianActorCritic,
    config: Config,
    seed: int,
    initial_state: State | None = None,
) -> ContinuousEpisode:
    env = ContinuousRendezvousEnv(config, seed=seed)
    state = env.reset(initial_state)
    states = [state]
    actions: list[NDArray[np.float64]] = []
    rewards: list[float] = []
    fuel = 0.0
    violations = 0

    while True:
        action = agent.deterministic_action(state)
        state, reward, done, info = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        fuel += info.fuel
        violations += int(info.unsafe_entry)
        if done:
            return ContinuousEpisode(
                states=np.asarray(states, dtype=np.float64),
                actions=np.asarray(actions, dtype=np.float64),
                rewards=np.asarray(rewards, dtype=np.float64),
                success=info.success,
                collision=info.collision,
                escaped=info.escaped,
                fuel=float(fuel),
                safety_violations=violations,
                final_distance=info.distance,
                final_speed=info.speed,
            )


def evaluate_actor_critic(
    agent: LinearGaussianActorCritic,
    config: Config,
    episodes: int | None = None,
) -> tuple[ContinuousEvaluationMetrics, list[ContinuousEpisode]]:
    n = config.evaluation_episodes if episodes is None else episodes
    trials = [
        run_actor_critic_episode(agent, config, config.evaluation_seed + i)
        for i in range(n)
    ]
    metrics = ContinuousEvaluationMetrics(
        success_rate=float(np.mean([e.success for e in trials])),
        collision_rate=float(np.mean([e.collision for e in trials])),
        escape_rate=float(np.mean([e.escaped for e in trials])),
        mean_final_distance=float(np.mean([e.final_distance for e in trials])),
        mean_final_speed=float(np.mean([e.final_speed for e in trials])),
        mean_fuel=float(np.mean([e.fuel for e in trials])),
        mean_safety_violations=float(np.mean([e.safety_violations for e in trials])),
        mean_return=float(np.mean([e.rewards.sum() for e in trials])),
    )
    return metrics, trials
