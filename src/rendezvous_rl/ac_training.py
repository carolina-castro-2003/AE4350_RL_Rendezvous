from __future__ import annotations

from dataclasses import dataclass, fields

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
    actor_weight_change: NDArray[np.float64]
    critic_weight_norm: NDArray[np.float64]
    validation_score: NDArray[np.float64]


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
        actor_weight_change=np.zeros(n, dtype=np.float64),
        critic_weight_norm=np.zeros(n, dtype=np.float64),
        validation_score=np.full(n, np.nan, dtype=np.float64),
    )
    sigma = ac.sigma_start
    initial_actor_weights = agent.actor_weights.copy()
    best_actor_weights = agent.actor_weights.copy()
    best_critic_weights = agent.critic_weights.copy()
    best_score = -np.inf
    have_trained_checkpoint = False
    validations_without_improvement = 0
    completed_episodes = n

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
                history.actor_weight_change[episode] = float(
                    np.linalg.norm(agent.actor_weights - initial_actor_weights)
                )
                history.critic_weight_norm[episode] = float(np.linalg.norm(agent.critic_weights))
                break

        sigma = max(ac.sigma_end, sigma * ac.sigma_decay)
        should_validate = (
            ac.validation_interval > 0
            and (episode + 1) >= ac.validation_start_episode
            and (episode + 1) % ac.validation_interval == 0
        )
        if should_validate:
            score = _validation_score(agent, config, ac)
            history.validation_score[episode] = score
            if score > best_score + ac.validation_min_delta:
                best_score = score
                best_actor_weights = agent.actor_weights.copy()
                best_critic_weights = agent.critic_weights.copy()
                have_trained_checkpoint = True
                validations_without_improvement = 0
            else:
                validations_without_improvement += 1
                if ac.rollback_on_validation_failure and have_trained_checkpoint:
                    agent.actor_weights = best_actor_weights.copy()
                    agent.critic_weights = best_critic_weights.copy()
                    agent.critic_trace[:] = 0.0

            if (
                have_trained_checkpoint
                and ac.early_stop_patience > 0
                and validations_without_improvement >= ac.early_stop_patience
            ):
                completed_episodes = episode + 1
                if verbose:
                    print(
                        f"early stop at episode {completed_episodes:5d} | "
                        f"best validation score {best_score:8.1f}"
                    )
                break
        if verbose and (episode + 1) % 500 == 0:
            start = max(0, episode - 499)
            print(
                f"episode {episode + 1:5d} | "
                f"success {100 * history.successes[start:episode + 1].mean():5.1f}% | "
                f"return {history.returns[start:episode + 1].mean():8.1f} | "
                f"sigma {sigma:.3f}"
            )
    if have_trained_checkpoint:
        agent.actor_weights = best_actor_weights
        agent.critic_weights = best_critic_weights
    history = _truncate_history(history, completed_episodes)
    return agent, history


def _truncate_history(history: ActorCriticHistory, episodes: int) -> ActorCriticHistory:
    """Remove unused zero-filled tail after early stopping."""

    values = {
        field.name: getattr(history, field.name)[:episodes]
        for field in fields(ActorCriticHistory)
    }
    return ActorCriticHistory(**values)


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
    final_speeds = []
    fuels = []
    safety_violations = []
    for i in range(ac_config.validation_episodes):
        env = ContinuousRendezvousEnv(config, seed=config.evaluation_seed + 50_000 + i)
        state = env.reset()
        episode_return = 0.0
        episode_fuel = 0.0
        episode_unsafe = 0
        while True:
            action = agent.deterministic_action(state)
            state, reward, done, info = env.step(action)
            episode_return += reward
            episode_fuel += info.fuel
            episode_unsafe += int(info.unsafe_entry)
            if done:
                successes += int(info.success)
                collisions += int(info.collision)
                escapes += int(info.escaped)
                returns.append(episode_return)
                final_distances.append(info.distance)
                final_speeds.append(info.speed)
                fuels.append(episode_fuel)
                safety_violations.append(episode_unsafe)
                break

    success_rate = successes / ac_config.validation_episodes
    collision_rate = collisions / ac_config.validation_episodes
    escape_rate = escapes / ac_config.validation_episodes
    mean_return = float(np.mean(returns))
    mean_final_distance = float(np.mean(final_distances))
    mean_final_speed = float(np.mean(final_speeds))
    mean_fuel = float(np.mean(fuels))
    mean_safety_violations = float(np.mean(safety_violations))
    return (
        1_000.0 * success_rate
        - 1_000.0 * collision_rate
        - 250.0 * escape_rate
        - 350.0 * mean_safety_violations
        + 0.05 * mean_return
        - 0.2 * mean_final_distance
        - 200.0 * mean_final_speed
        - 2.0 * mean_fuel
    )
