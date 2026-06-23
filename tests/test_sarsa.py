import numpy as np

from rendezvous_rl.config import Config
from rendezvous_rl.discretization import StateDiscretizer
from rendezvous_rl.sarsa import SarsaAgent


def test_state_index_is_valid_at_and_outside_edges() -> None:
    d = StateDiscretizer(Config())
    for state in (
        np.zeros(4),
        np.array([-1e6, 1e6, -1e3, 1e3]),
    ):
        index = d.encode(state)
        assert 0 <= index < d.n_states


def test_terminal_sarsa_update_does_not_bootstrap() -> None:
    cfg = Config(alpha=0.5, gamma=0.9)
    agent = SarsaAgent(cfg)
    agent.q[1, 2] = 4.0
    agent.q[3, 1] = 100.0
    error = agent.update(1, 2, reward=2.0, next_state_index=3, next_action=1, terminal=True)
    assert error == -2.0
    assert agent.q[1, 2] == 3.0

