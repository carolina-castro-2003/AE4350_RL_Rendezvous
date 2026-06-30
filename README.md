# Reinforcement Learning for Fuel-Efficient and Collision-Aware Spacecraft Rendezvous

A complete AE4350-style Python repository for two reinforcement-learning controllers in a simplified planar spacecraft-rendezvous problem:

- a tabular, on-policy SARSA controller with discrete thrust commands;
- a continuous-thrust actor-critic controller with a stochastic Gaussian policy and a learned value critic.

## Why this matches the lecture

The lecture distinguishes discrete RL from continuous RL with function approximators. This project now includes both routes, which gives you a stronger comparison section.

The SARSA part follows the discrete route:

- Agent state: discretised relative position and velocity `[x, y, vx, vy]`.
- Environment: simplified planar Clohessy-Wiltshire/Hill relative dynamics.
- Actions: coast, `+x`, `-x`, `+y`, `-y` thrust.
- Policy: epsilon-greedy during learning; greedy during evaluation.
- Stored information: an action-value table `Q(s,a)`.
- Learning: online TD bootstrapping with the slide-29 SARSA update

  `Q(s,a) <- Q(s,a) + alpha [r + gamma Q(s',a') - Q(s,a)]`.

The assignment slide asks for a system description, RL design, learning effect, results, and sensitivity to learning parameters. The scripts generate each of those experimental ingredients. No black-box RL framework is used, so every step is visible and explainable.

The actor-critic part follows the continuous-control route:

- Agent state: continuous relative position and velocity `[x, y, vx, vy]`.
- Environment: the same simplified planar Clohessy-Wiltshire/Hill dynamics.
- Actions: continuous two-axis thrust acceleration `[u_x, u_y]`, clipped to the maximum thruster acceleration.
- Actor: a stochastic Gaussian policy over continuous thrust commands, squashed through `tanh` so physical thrust limits are always respected.
- Critic: a linear state-value approximation `V(s)` trained by the TD error.
- Learning signal: the same TD error updates both critic and actor, which is the core actor-critic idea.

This is still intentionally transparent rather than a black-box deep-RL implementation. It is more advanced than SARSA because it removes the discrete action table and uses function approximation, while remaining explainable enough for an AE4350 report.

## Setup in VS Code

Open this folder in VS Code, then run in its terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
pytest
```

## Run

First run the fast check:

```bash
python -m rendezvous_rl.experiment --quick
python -m rendezvous_rl.sensitivity --quick
```

Then generate report-quality results:

```bash
python -m rendezvous_rl.experiment
python -m rendezvous_rl.sensitivity
python -m rendezvous_rl.actor_critic_experiment
python -m rendezvous_rl.actor_critic_sensitivity
```

The main experiment trains three independent seeds, evaluates each policy on the same 250 unseen initial conditions, and reports mean and standard deviation. The sensitivity experiment changes one parameter at a time and repeats every setting across independent seeds. It is intentionally slower.

Useful overrides:

```bash
python -m rendezvous_rl.experiment --episodes 6000 --seeds 7 17 27 37 47
python -m rendezvous_rl.sensitivity --episodes 3000 --seeds 101 202 303 404 505
python -m rendezvous_rl.actor_critic_experiment --episodes 8000 --seeds 7 17 27
python -m rendezvous_rl.actor_critic_experiment --episodes 10000 --actor-alpha 0.012 --critic-alpha 0.050
```

## Generated evidence

The `results/` directory will contain:

- `learning_curves.png`: return, success rate, thrust use, and unsafe entries with uncertainty bands.
- `example_trajectory.png`: an interpretable greedy rendezvous trajectory.
- `evaluation_histograms.png`: final distance, speed, and fuel distributions.
- `evaluation_by_seed.csv` and `evaluation_aggregate.json`: repeated-run statistics.
- `q_table_seed_*.npz`: learned action-value tables.
- `sensitivity_raw.csv` and `sensitivity_success.png`: one-factor sensitivity results.

The `results_actor_critic/` directory will contain the continuous-method equivalents:

- `learning_curves.png`: return, success rate, normalised impulse, and exploration standard deviation.
- `example_trajectory.png`: deterministic actor trajectory with thrust-direction arrows.
- `evaluation_histograms.png`: final distance, speed, and normalised impulse distributions.
- `evaluation_by_seed.csv` and `evaluation_aggregate.json`: repeated-run statistics.
- `actor_critic_seed_*.npz`: learned actor and critic weights.
- `actor_critic_sensitivity_raw.csv` and `actor_critic_sensitivity_success.png`: actor-critic sensitivity results.

## Safety and reward definitions

Docking succeeds only when distance and relative speed are both low. The keep-out zone is not simply forbidden, because the chaser must enter it to dock. Instead, entering or occupying it above `safe_entry_speed` is unsafe and penalised. High-speed contact inside `collision_radius` is a collision. Each non-coast command consumes one abstract fuel unit.

The reward combines distance progress, relative-speed cost, time cost, thrust/fuel cost, unsafe-operation cost, and terminal success/collision/escape terms. Reward design strongly affects the learned policy, exactly as stressed in lecture slide 10.

## Suggested report structure

1. Research question and hypothesis.
2. Simplified Hill-frame dynamics, assumptions, state, actions, and terminal conditions.
3. State discretisation, Q-table, epsilon-greedy policy, SARSA pseudocode/equation.
4. Continuous actor-critic extension: continuous action space, policy, value function, TD error, exploration standard deviation.
5. Reward-function terms and physical interpretation.
6. Experimental protocol: seeds, training episodes, unseen evaluation cases, metrics.
7. Learning curves and uncertainty across seeds.
8. Trajectory interpretation: braking, coasting, keep-out entry, docking speed.
9. SARSA versus actor-critic comparison: success rate, final speed, fuel/impulse, safety violations, and smoothness of the trajectory.
10. Sensitivity to `alpha`, actor/critic learning rates, exploration standard deviation, fuel penalty, safety penalty, and keep-out radius.
11. Limitations: planar linearised dynamics, handcrafted features for the actor-critic method, no sensor noise, and no flight-readiness claim.

Before settling on conclusions, inspect the actual plots and explain what the agent learned. Do not claim success merely because the script finished.
