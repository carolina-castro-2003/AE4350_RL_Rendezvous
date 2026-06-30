from __future__ import annotations

import argparse
from dataclasses import fields

import numpy as np

from .ac_evaluation import evaluate_actor_critic
from .ac_plotting import (
    plot_actor_critic_evaluation,
    plot_actor_critic_learning,
    plot_actor_critic_trajectory,
)
from .ac_training import train_actor_critic
from .actor_critic import ActorCriticConfig
from .config import Config
from .io_utils import write_json, write_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate continuous actor-critic rendezvous agents."
    )
    parser.add_argument("--episodes", type=int, default=8_000)
    parser.add_argument("--evaluation-episodes", type=int, default=250)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--actor-alpha", type=float, default=0.010)
    parser.add_argument("--critic-alpha", type=float, default=0.060)
    parser.add_argument("--quick", action="store_true", help="Fast code check; not report-quality data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = 200 if args.quick else args.episodes
    evaluation_episodes = 20 if args.quick else args.evaluation_episodes
    seeds = args.seeds[:1] if args.quick else args.seeds

    base = Config(
        episodes=episodes,
        evaluation_episodes=evaluation_episodes,
        results_dir="results_actor_critic",
    )
    ac_config = ActorCriticConfig(
        actor_alpha=args.actor_alpha,
        critic_alpha=args.critic_alpha,
    )

    output = base.output_path
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", base.as_dict())
    write_json(output / "actor_critic_config.json", ac_config.as_dict())

    histories = []
    agents = []
    all_trials = []
    summary_rows: list[dict[str, float | int]] = []

    for seed in seeds:
        print(f"\n=== actor-critic independent training seed {seed} ===")
        config = base.changed(seed=seed)
        agent, history = train_actor_critic(config, ac_config)
        metrics, trials = evaluate_actor_critic(agent, config)
        row: dict[str, float | int] = {"seed": seed, **metrics.as_dict()}
        summary_rows.append(row)
        histories.append(history)
        agents.append(agent)
        all_trials.append(trials)
        agent.save(output / f"actor_critic_seed_{seed}.npz")
        print(row)

    write_rows(output / "evaluation_by_seed.csv", summary_rows)
    numeric_keys = [f.name for f in fields(type(metrics))]
    aggregate = {
        key: {
            "mean": float(np.mean([float(row[key]) for row in summary_rows])),
            "std": float(np.std([float(row[key]) for row in summary_rows], ddof=1))
            if len(summary_rows) > 1
            else 0.0,
        }
        for key in numeric_keys
    }
    write_json(output / "evaluation_aggregate.json", aggregate)

    plot_actor_critic_learning(histories, base, output / "learning_curves.png")
    best_index = int(np.argmax([row["success_rate"] for row in summary_rows]))
    best_trials = all_trials[best_index]
    successful = [trial for trial in best_trials if trial.success]
    example = successful[0] if successful else best_trials[0]
    plot_actor_critic_trajectory(example, base, output / "example_trajectory.png")
    plot_actor_critic_evaluation(best_trials, output / "evaluation_histograms.png")
    print(f"\nActor-critic results written to {output.resolve()}")


if __name__ == "__main__":
    main()
