from __future__ import annotations

import numpy as np

from rendezvous_rl.actor_critic import ActorCriticConfig, LinearGaussianActorCritic
from rendezvous_rl.config import Config
from rendezvous_rl.continuous_environment import ContinuousRendezvousEnv


def test_continuous_environment_accepts_bounded_action() -> None:
    config = Config(max_steps=3)
    env = ContinuousRendezvousEnv(config, seed=1)
    env.reset(np.array([10.0, 0.0, 0.0, 0.0]))
    state, reward, done, info = env.step(np.array([0.0, 0.0]))
    assert state.shape == (4,)
    assert isinstance(reward, float)
    assert not done
    assert info.fuel == 0.0


def test_actor_critic_outputs_physical_thrust_bounds() -> None:
    config = Config()
    agent = LinearGaussianActorCritic(config, ActorCriticConfig(), seed=2)
    action = agent.deterministic_action(np.array([20.0, -10.0, 0.01, 0.0]))
    assert action.shape == (2,)
    assert np.all(np.abs(action) <= config.thrust_acceleration)


def test_actor_critic_update_changes_parameters() -> None:
    config = Config()
    agent = LinearGaussianActorCritic(config, ActorCriticConfig(), seed=3)
    state = np.array([20.0, 0.0, 0.0, 0.0])
    sample = agent.sample_action(state, sigma=0.5)
    old_actor = agent.actor_weights.copy()
    old_critic = agent.critic_weights.copy()
    agent.update(sample, reward=1.0, next_state=np.array([19.0, 0.0, 0.0, 0.0]), done=False)
    assert not np.allclose(agent.actor_weights, old_actor)
    assert not np.allclose(agent.critic_weights, old_critic)
