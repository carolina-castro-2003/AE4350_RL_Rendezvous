# Reinforcement Learning for Fuel-Efficient and Collision-Aware Spacecraft Rendezvous

A complete AE4350-style Python repository for a tabular, on-policy SARSA controller in a simplified planar spacecraft-rendezvous problem.

## Why this matches the lecture

The lecture distinguishes discrete RL from continuous RL with function approximators. This project deliberately follows the discrete route:

- Agent state: discretised relative position and velocity `[x, y, vx, vy]`.
- Environment: simplified planar Clohessy-Wiltshire/Hill relative dynamics.
- Actions: coast, `+x`, `-x`, `+y`, `-y` thrust.
- Policy: epsilon-greedy during learning; greedy during evaluation.
- Stored information: an action-value table `Q(s,a)`.
- Learning: online TD bootstrapping with the slide-29 SARSA update

  `Q(s,a) <- Q(s,a) + alpha [r + gamma Q(s',a') - Q(s,a)]`.

The assignment slide asks for a system description, RL design, learning effect, results, and sensitivity to learning parameters. The scripts generate each of those experimental ingredients. No black-box RL framework is used, so every step is visible and explainable.

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
```

The main experiment trains three independent seeds, evaluates each policy on the same 250 unseen initial conditions, and reports mean and standard deviation. The sensitivity experiment changes one parameter at a time and repeats every setting across independent seeds. It is intentionally slower.

Useful overrides:

```bash
python -m rendezvous_rl.experiment --episodes 6000 --seeds 7 17 27 37 47
python -m rendezvous_rl.sensitivity --episodes 3000 --seeds 101 202 303 404 505
```

## Generated evidence

The `results/` directory will contain:

- `learning_curves.png`: return, success rate, thrust use, and unsafe entries with uncertainty bands.
- `example_trajectory.png`: an interpretable greedy rendezvous trajectory.
- `evaluation_histograms.png`: final distance, speed, and fuel distributions.
- `evaluation_by_seed.csv` and `evaluation_aggregate.json`: repeated-run statistics.
- `q_table_seed_*.npz`: learned action-value tables.
- `sensitivity_raw.csv` and `sensitivity_success.png`: one-factor sensitivity results.

## Safety and reward definitions

Docking succeeds only when distance and relative speed are both low. The keep-out zone is not simply forbidden, because the chaser must enter it to dock. Instead, entering or occupying it above `safe_entry_speed` is unsafe and penalised. High-speed contact inside `collision_radius` is a collision. Each non-coast command consumes one abstract fuel unit.

The reward combines distance progress, relative-speed cost, time cost, thrust/fuel cost, unsafe-operation cost, and terminal success/collision/escape terms. Reward design strongly affects the learned policy, exactly as stressed in lecture slide 10.

## Suggested report structure

1. Research question and hypothesis.
2. Simplified Hill-frame dynamics, assumptions, state, actions, and terminal conditions.
3. State discretisation, Q-table, epsilon-greedy policy, SARSA pseudocode/equation.
4. Reward-function terms and physical interpretation.
5. Experimental protocol: seeds, training episodes, unseen evaluation cases, metrics.
6. Learning curves and uncertainty across seeds.
7. Trajectory interpretation: braking, coasting, keep-out entry, docking speed.
8. Sensitivity to `alpha`, fuel penalty, safety penalty, and keep-out radius.
9. Limitations: planar linearised dynamics, coarse bins, no sensor noise, and no flight-readiness claim.

Before settling on conclusions, inspect the actual plots and explain what the agent learned. Do not claim success merely because the script finished.
