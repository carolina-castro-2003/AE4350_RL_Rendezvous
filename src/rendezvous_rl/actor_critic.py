from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .config import Config
from .environment import State


FeatureVector = NDArray[np.float64]
ContinuousAction = NDArray[np.float64]


@dataclass(frozen=True)
class ActorCriticConfig:
    """Learning parameters for the continuous actor-critic controller."""

    actor_alpha: float = 0.010
    critic_alpha: float = 0.060
    gamma: float = 0.995
    entropy_beta: float = 0.0005
    prior_beta: float = 0.0015
    validation_interval: int = 250
    validation_episodes: int = 40
    sigma_start: float = 0.85
    sigma_end: float = 0.08
    sigma_decay: float = 0.9994
    eligibility_lambda: float = 0.0
    max_grad_norm: float = 8.0
    warm_start_scale: float = 1.0
    position_scale: float = 35.0
    velocity_scale: float = 0.08

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PolicySample:
    """One sampled action and the quantities needed for the policy-gradient step."""

    action: ContinuousAction
    raw_action: ContinuousAction
    mean_raw: ContinuousAction
    sigma: float
    features: FeatureVector


def state_features(state: State, config: Config, ac_config: ActorCriticConfig) -> FeatureVector:
    """Return bounded handcrafted features for linear function approximation.

    The lecture introduces function approximation as the natural next step when
    a table becomes unsuitable. For a compact assignment implementation, this
    keeps the approximator linear and readable instead of using a neural-network
    library. Features are normalised so actor and critic learning rates have a
    predictable scale.
    """

    x, y, vx, vy = np.asarray(state, dtype=np.float64)
    pos = np.array([x, y], dtype=np.float64)
    vel = np.array([vx, vy], dtype=np.float64)
    distance = float(np.linalg.norm(pos))
    speed = float(np.linalg.norm(vel))
    bearing = pos / max(distance, 1e-9)
    closing_speed = float(np.dot(pos, vel) / max(distance, 1e-9))

    p = ac_config.position_scale
    v = ac_config.velocity_scale
    return np.array(
        [
            1.0,
            np.clip(x / p, -2.0, 2.0),
            np.clip(y / p, -2.0, 2.0),
            np.clip(vx / v, -2.0, 2.0),
            np.clip(vy / v, -2.0, 2.0),
            np.clip(distance / p, 0.0, 2.0),
            np.clip(speed / v, 0.0, 2.0),
            np.clip(closing_speed / v, -2.0, 2.0),
            float(distance < config.keep_out_radius),
            bearing[0],
            bearing[1],
        ],
        dtype=np.float64,
    )


class LinearGaussianActorCritic:
    """Continuous on-policy actor-critic with a Gaussian thrust policy.

    Critic:
        V_w(s) = w^T phi(s)
        delta = r + gamma V_w(s') - V_w(s)
        w <- w + alpha_v delta phi(s)

    Actor:
        z ~ Normal(mu_theta(s), sigma^2 I)
        action = thrust_acceleration * tanh(z)
        theta <- theta + alpha_pi delta grad_theta log pi_theta(z | s)

    The actor samples in unconstrained space and then squashes through tanh, so
    commands are always inside the physical thrust bounds.
    """

    def __init__(
        self,
        config: Config,
        ac_config: ActorCriticConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.ac_config = ActorCriticConfig() if ac_config is None else ac_config
        self.rng = np.random.default_rng(config.seed + 2 if seed is None else seed)

        n_features = state_features(np.zeros(4), config, self.ac_config).size
        self.actor_weights = np.zeros((2, n_features), dtype=np.float64)
        self.critic_weights = np.zeros(n_features, dtype=np.float64)
        self.critic_trace = np.zeros(n_features, dtype=np.float64)
        self._initialise_guidance_prior()
        self.prior_actor_weights = self.actor_weights.copy()

    def _initialise_guidance_prior(self) -> None:
        """Initialise the actor with a weak proportional-derivative prior.

        Pure random continuous control is very inefficient for this problem.
        This prior gives the agent a sensible first guess: thrust roughly toward
        the target and oppose relative velocity. Learning can still overwrite it.
        In the report, describe this as an informed initial policy, not as a
        hand-coded final controller.
        """

        scale = self.ac_config.warm_start_scale
        position_gain = 0.50 * scale
        velocity_gain = 1.20 * scale
        bearing_gain = 0.50 * scale
        # Feature order: bias, x, y, vx, vy, distance, speed, closing, inside, bx, by
        self.actor_weights[0, 1] = -1.15 * position_gain
        self.actor_weights[0, 3] = -0.80 * velocity_gain
        self.actor_weights[0, 9] = -0.45 * bearing_gain
        self.actor_weights[1, 2] = -1.15 * position_gain
        self.actor_weights[1, 4] = -0.80 * velocity_gain
        self.actor_weights[1, 10] = -0.45 * bearing_gain

    def features(self, state: State) -> FeatureVector:
        return state_features(state, self.config, self.ac_config)

    def value(self, state: State) -> float:
        return float(self.critic_weights @ self.features(state))

    def mean_raw(self, features: FeatureVector) -> ContinuousAction:
        return np.clip(self.actor_weights @ features, -3.0, 3.0)

    def deterministic_action(self, state: State) -> ContinuousAction:
        phi = self.features(state)
        raw = self.mean_raw(phi)
        return self.config.thrust_acceleration * np.tanh(raw)

    def sample_action(self, state: State, sigma: float) -> PolicySample:
        phi = self.features(state)
        mean = self.mean_raw(phi)
        raw = mean + sigma * self.rng.standard_normal(2)
        action = self.config.thrust_acceleration * np.tanh(raw)
        return PolicySample(
            action=action,
            raw_action=raw,
            mean_raw=mean,
            sigma=float(sigma),
            features=phi,
        )

    def update(
        self,
        sample: PolicySample,
        reward: float,
        next_state: State,
        done: bool,
    ) -> float:
        phi = sample.features
        current_value = float(self.critic_weights @ phi)
        next_value = 0.0 if done else self.value(next_state)
        delta = reward + self.ac_config.gamma * next_value - current_value

        if self.ac_config.eligibility_lambda > 0.0:
            self.critic_trace = (
                self.ac_config.gamma * self.ac_config.eligibility_lambda * self.critic_trace
                + phi
            )
            critic_direction = self.critic_trace
        else:
            critic_direction = phi
        self.critic_weights += self.ac_config.critic_alpha * delta * critic_direction

        sigma2 = max(sample.sigma * sample.sigma, 1e-8)
        score = ((sample.raw_action - sample.mean_raw) / sigma2)[:, None] * phi[None, :]
        entropy_push = -sample.mean_raw[:, None] * phi[None, :]
        prior_pull = self.prior_actor_weights - self.actor_weights
        actor_grad = (
            delta * score
            + self.ac_config.entropy_beta * entropy_push
            + self.ac_config.prior_beta * prior_pull
        )
        grad_norm = float(np.linalg.norm(actor_grad))
        if grad_norm > self.ac_config.max_grad_norm:
            actor_grad *= self.ac_config.max_grad_norm / grad_norm
        self.actor_weights += self.ac_config.actor_alpha * actor_grad
        self.actor_weights = np.clip(self.actor_weights, -8.0, 8.0)

        return float(delta)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            actor_weights=self.actor_weights,
            critic_weights=self.critic_weights,
            config=self.config.as_dict(),
            actor_critic_config=self.ac_config.as_dict(),
        )

    @classmethod
    def load(
        cls,
        path: Path,
        config: Config,
        ac_config: ActorCriticConfig | None = None,
    ) -> "LinearGaussianActorCritic":
        data = np.load(path, allow_pickle=True)
        agent = cls(config=config, ac_config=ac_config)
        agent.actor_weights = np.asarray(data["actor_weights"], dtype=np.float64)
        agent.critic_weights = np.asarray(data["critic_weights"], dtype=np.float64)
        agent.critic_trace = np.zeros_like(agent.critic_weights)
        return agent
