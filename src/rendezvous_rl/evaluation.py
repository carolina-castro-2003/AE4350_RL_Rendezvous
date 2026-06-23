from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from .config import Config
from .environment import RendezvousEnv, State
from .sarsa import SarsaAgent


@dataclass
class Episode:
    states: NDArray[np.float64]
    actions: NDArray[np.int32]
    rewards: NDArray[np.float64]
    success: bool
    collision: bool
    escaped: bool
    fuel: int
    safety_violations: int
    final_distance: float
    final_speed: float


@dataclass(frozen=True)
class EvaluationMetrics:
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


def run_episode(
    agent: SarsaAgent,
    config: Config,
    seed: int,
    initial_state: State | None = None,
) -> Episode:
    env = RendezvousEnv(config, seed=seed)
    state = env.reset(initial_state)
    states = [state]
    actions: list[int] = []
    rewards: list[float] = []
    fuel = violations = 0

    while True:
        index = agent.discretizer.encode(state)
        action = agent.choose_action(index, epsilon=0.0)
        state, reward, done, info = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        fuel += info.fuel
        violations += int(info.unsafe_entry)
        if done:
            return Episode(
                states=np.asarray(states),
                actions=np.asarray(actions, dtype=np.int32),
                rewards=np.asarray(rewards),
                success=info.success,
                collision=info.collision,
                escaped=info.escaped,
                fuel=fuel,
                safety_violations=violations,
                final_distance=info.distance,
                final_speed=info.speed,
            )


def evaluate(
    agent: SarsaAgent, config: Config, episodes: int | None = None
) -> tuple[EvaluationMetrics, list[Episode]]:
    n = config.evaluation_episodes if episodes is None else episodes
    # All independently trained agents see the same held-out test cases.
    trials = [run_episode(agent, config, config.evaluation_seed + i) for i in range(n)]
    metrics = EvaluationMetrics(
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
