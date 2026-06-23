from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import Config
from .environment import RendezvousEnv
from .sarsa import SarsaAgent


@dataclass
class TrainingHistory:
    returns: NDArray[np.float64]
    steps: NDArray[np.int32]
    successes: NDArray[np.bool_]
    fuel: NDArray[np.int32]
    safety_violations: NDArray[np.int32]
    mean_abs_td_error: NDArray[np.float64]


def train(config: Config, verbose: bool = True) -> tuple[SarsaAgent, TrainingHistory]:
    """Train using online, on-policy SARSA with decaying epsilon-greedy exploration."""
    env = RendezvousEnv(config, seed=config.seed)
    agent = SarsaAgent(config, seed=config.seed + 1)
    n = config.episodes
    history = TrainingHistory(
        returns=np.zeros(n),
        steps=np.zeros(n, dtype=np.int32),
        successes=np.zeros(n, dtype=bool),
        fuel=np.zeros(n, dtype=np.int32),
        safety_violations=np.zeros(n, dtype=np.int32),
        mean_abs_td_error=np.zeros(n),
    )
    epsilon = config.epsilon_start

    for episode in range(n):
        state = env.reset()
        state_index = agent.discretizer.encode(state)
        action = agent.choose_action(state_index, epsilon)
        td_errors: list[float] = []

        while True:
            next_state, reward, done, info = env.step(action)
            next_index = agent.discretizer.encode(next_state)
            next_action = agent.choose_action(next_index, epsilon)
            td_errors.append(abs(agent.update(
                state_index, action, reward, next_index, next_action, done
            )))

            history.returns[episode] += reward
            history.fuel[episode] += info.fuel
            history.safety_violations[episode] += int(info.unsafe_entry)
            state_index, action = next_index, next_action
            if done:
                history.steps[episode] = env.steps
                history.successes[episode] = info.success
                history.mean_abs_td_error[episode] = float(np.mean(td_errors))
                break

        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        if verbose and (episode + 1) % 500 == 0:
            start = max(0, episode - 499)
            print(
                f"episode {episode + 1:5d} | "
                f"success {100 * history.successes[start:episode + 1].mean():5.1f}% | "
                f"return {history.returns[start:episode + 1].mean():8.1f} | "
                f"epsilon {epsilon:.3f}"
            )
    return agent, history

