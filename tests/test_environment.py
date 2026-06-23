import numpy as np

from rendezvous_rl.config import Config
from rendezvous_rl.environment import Action, RendezvousEnv


def test_coast_step_matches_shape_and_advances_time() -> None:
    cfg = Config(max_steps=10)
    env = RendezvousEnv(cfg)
    env.reset(np.array([20.0, 0.0, 0.0, 0.0]))
    state, reward, done, info = env.step(Action.COAST)
    assert state.shape == (4,)
    assert np.isfinite(reward)
    assert env.steps == 1
    assert not done
    assert info.fuel == 0


def test_low_speed_inside_docking_region_is_success() -> None:
    cfg = Config(dt=0.01, max_steps=10)
    env = RendezvousEnv(cfg)
    env.reset(np.array([1.0, 0.0, 0.0, 0.0]))
    _, _, done, info = env.step(Action.COAST)
    assert done
    assert info.success
    assert not info.collision


def test_thrust_consumes_one_fuel_unit() -> None:
    env = RendezvousEnv(Config())
    env.reset(np.array([20.0, 0.0, 0.0, 0.0]))
    _, _, _, info = env.step(Action.PLUS_X)
    assert info.fuel == 1

