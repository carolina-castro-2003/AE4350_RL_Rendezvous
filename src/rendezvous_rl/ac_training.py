from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .actor_critic import ActorCriticConfig, LinearGaussianActorCritic
from .config import Config
from .continuous_environment import ContinuousRendezvousEnv


@dataclass
class ActorCriticHistory:
    returns: NDArray[np.float64]
    steps: NDArray[np.int32]
    successes: NDArray[np.bool_]
    fuel: NDArray[np.float64]
    safety_violations: NDArray[np.int32]
    mean_abs_td_error: NDArray[np.float64]
    sigma: NDArray[np.float64]


def train_actor_critic(
    config: Config,
    ac_config: ActorCriticConfig | None = None,
    verbose: bool = True,
) -> tuple[LinearGaussianActorCritic, ActorCriticHistory]:
    """Train a continuous actor-critic agent with on-policy TD errors."""

    ac = ActorCriticConfig() if ac_config is None else ac_config
    env = ContinuousRendezvousEnv(config, seed=config.seed)
    agent = LinearGaussianActorCritic(config, ac, seed=config.seed + 11)
    n = config.episodes
    history = ActorCriticHistory(
        returns=np.zeros(n, dtype=np.float64),
        steps=np.zeros(n, dtype=np.int32),
        successes=np.zeros(n, dtype=bool),
        fuel=np.zeros(n, dtype=np.float64),
        safety_violations=np.zeros(n, dtype=np.int32),
        mean_abs_td_error=np.zeros(n, dtype=np.float64),
        sigma=np.zeros(n, dtype=np.float64),
    )
    sigma = ac.sigma_start
    best_actor_weights = agent.actor_weights.copy()
    best_score = _validation_score(agent, config, ac)

    for episode in range(n):
        state = env.reset()
        agent.critic_trace[:] = 0.0
        td_errors: list[float] = []

        while True:
            sample = agent.sample_action(state, sigma)
            next_state, reward, done, info = env.step(sample.action)
            td_error = agent.update(sample, reward, next_state, done)

            history.returns[episode] += reward
            history.fuel[episode] += info.fuel
            history.safety_violations[episode] += int(info.unsafe_entry)
            td_errors.append(abs(td_error))
            state = next_state
            if done:
                history.steps[episode] = env.steps
                history.successes[episode] = info.success
                history.mean_abs_td_error[episode] = float(np.mean(td_errors))
                history.sigma[episode] = sigma
                break

        sigma = max(ac.sigma_end, sigma * ac.sigma_decay)
        if ac.validation_interval > 0 and (episode + 1) % ac.validation_interval == 0:
            score = _validation_score(agent, config, ac)
            if score > best_score:
                best_score = score
                best_actor_weights = agent.actor_weights.copy()
        if verbose and (episode + 1) % 500 == 0:
            start = max(0, episode - 499)
            print(
                f"episode {episode + 1:5d} | "
                f"success {100 * history.successes[start:episode + 1].mean():5.1f}% | "
                f"return {history.returns[start:episode + 1].mean():8.1f} | "
                f"sigma {sigma:.3f}"
            )
    agent.actor_weights = best_actor_weights
    return agent, history


def _validation_score(
    agent: LinearGaussianActorCritic,
    config: Config,
    ac_config: ActorCriticConfig,
) -> float:
    """Score the deterministic actor on fixed validation initial conditions."""

    successes = 0
    collisions = 0
    escapes = 0
    returns = []
    final_distances = []
    for i in range(ac_config.validation_episodes):
        env = ContinuousRendezvousEnv(config, seed=config.evaluation_seed + 50_000 + i)
        state = env.reset()
        episode_return = 0.0
        while True:
            action = agent.deterministic_action(state)
            state, reward, done, info = env.step(action)
            episode_return += reward
            if done:
                successes += int(info.success)
                collisions += int(info.collision)
                escapes += int(info.escaped)
                returns.append(episode_return)
                final_distances.append(info.distance)
                break

    success_rate = successes / ac_config.validation_episodes
    collision_rate = collisions / ac_config.validation_episodes
    escape_rate = escapes / ac_config.validation_episodes
    mean_return = float(np.mean(returns))
    mean_final_distance = float(np.mean(final_distances))
    return (
        1_000.0 * success_rate
        - 400.0 * collision_rate
        - 250.0 * escape_rate
        + 0.05 * mean_return
        - 0.2 * mean_final_distance
    )
