# Reinforcement Learning for Fuel-Efficient and Collision-Aware Spacecraft Rendezvous

A complete Python repository for two reinforcement-learning controllers in a simplified planar spacecraft-rendezvous problem:
- a tabular, on-policy SARSA controller with discrete thrust commands;
- a continuous-thrust actor-critic controller with a stochastic Gaussian policy and a learned value critic.


The lecture distinguishes discrete RL from continuous RL with function approximators. This project now includes both routes, which gives you a stronger comparison section.

The SARSA part follows the discrete route:
- Agent state: discretised relative position and velocity `[x, y, vx, vy]`.
- Environment: simplified planar Clohessy-Wiltshire/Hill relative dynamics.
- Actions: coast, `+x`, `-x`, `+y`, `-y` thrust.
- Policy: epsilon-greedy during learning; greedy during evaluation.
- Stored information: an action-value table `Q(s,a)`.
- Learning: online TD bootstrapping with the slide-29 SARSA update
  `Q(s,a) <- Q(s,a) + alpha [r + gamma Q(s',a') - Q(s,a)]`.


The actor-critic part follows the continuous-control route:
- Agent state: continuous relative position and velocity `[x, y, vx, vy]`.
- Environment: the same simplified planar Clohessy-Wiltshire/Hill dynamics.
- Actions: continuous two-axis thrust acceleration `[u_x, u_y]`, clipped to the maximum thruster acceleration.
- Actor: a stochastic Gaussian policy over continuous thrust commands, squashed through `tanh` so physical thrust limits are always respected.
- Critic: a linear state-value approximation `V(s)` trained by the TD error.
- Learning signal: the same TD error updates both critic and actor, which is the core actor-critic idea.
- Exploration: Gaussian action noise during training, decayed over time.
- Model selection: validation-based early stopping is used to avoid actor-critic policy collapse. The best validation actor/critic checkpoint is restored for final evaluation.
- Initial policy: by default, the actor starts from a near-random policy (`warm_start_scale = 0.0`) so the reported improvement is due to learning rather than a hand-designed guidance law.

It is more advanced than SARSA because it removes the discrete action table and uses function approximation, while remaining explainable enough for the report. 

## Setup in VS Code
```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
pytest
```

## Run
```bash
python -m rendezvous_rl.experiment
python -m rendezvous_rl.sensitivity
python -m rendezvous_rl.actor_critic_experiment
python -m rendezvous_rl.actor_critic_sensitivity
```
The main experiment trains three independent seeds, evaluates each policy on the same 250 unseen initial conditions, and reports mean and standard deviation. The sensitivity experiment changes one parameter at a time and repeats every setting across independent seeds. It is intentionally slower.

For actor-critic, `--episodes` is the maximum training budget. The run may stop earlier because validation-based early stopping is used. This prevents the actor from continuing to update after a good policy has been found, which can otherwise cause policy collapse in actor-critic methods. Actor-critic uses early stopping and checkpoint rollback, so 8000 episodes is a maximum budget, not a guaranteed run length.

Useful overrides:

```bash
python -m rendezvous_rl.experiment --episodes 6000 --seeds 7 17 27 37 47
python -m rendezvous_rl.sensitivity --episodes 3000 --seeds 101 202 303 404 505
python -m rendezvous_rl.actor_critic_experiment --episodes 8000 --evaluation-episodes 250 --seeds 7 17 27
python -m rendezvous_rl.actor_critic_sensitivity --episodes 1500 --evaluation-episodes 80 --seeds 101 202
python -m rendezvous_rl.actor_critic_experiment --episodes 8000 --actor-alpha 0.001 --critic-alpha 0.02
```


## Generated evidence

The `results/` directory will contain:
- `learning_curves.png`: return, success rate, thrust use, and unsafe entries with uncertainty bands.
- `example_trajectory.png`: an interpretable greedy rendezvous trajectory.
- `evaluation_histograms.png`: final distance, speed, and fuel distributions.
- `evaluation_by_seed.csv` and `evaluation_aggregate.json`: repeated-run statistics.
- `q_table_seed_*.npz`: learned action-value tables.
- `sensitivity_raw.csv` and `sensitivity_success.png`: one-factor sensitivity results.


The `results_actor_critic/` directory will contain:
- `learning_curves.png`: return, success rate, normalised impulse, and actor weight change.
- `example_trajectory.png`: deterministic actor trajectory with thrust-direction arrows.
- `evaluation_histograms.png`: final distance, speed, and normalised impulse distributions.
- `evaluation_by_seed.csv` and `evaluation_aggregate.json`: repeated-run actor-critic statistics.
- `prior_evaluation_by_seed.csv`: performance of the initial actor before learning.
- `actor_critic_seed_*.npz`: learned actor weights, critic weights, and initial actor weights.
- `actor_critic_sensitivity_baseline.csv`: baseline sensitivity run.
- `actor_critic_sensitivity_raw.csv`: all sensitivity rows.
- `actor_critic_sensitivity_unique_runs.csv`: only retrained non-duplicate sensitivity configurations.
- `actor_critic_sensitivity_success.png`: actor-critic sensitivity plot.

## Safety and reward definitions
Docking succeeds only when distance and relative speed are both low. The keep-out zone is not simply forbidden, because the chaser must enter it to dock. Instead, entering or occupying it above `safe_entry_speed` is unsafe and penalised. High-speed contact inside `collision_radius` is a collision. Each non-coast command consumes one abstract fuel unit.

The reward combines distance progress, relative-speed cost, time cost, thrust/fuel cost, unsafe-operation cost, and terminal success/collision/escape terms. Reward design strongly affects the learned policy, exactly as stressed in lecture slide 10.
