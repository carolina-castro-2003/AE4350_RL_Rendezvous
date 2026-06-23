from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import numpy as np

from .config import Config
from .evaluation import evaluate
from .io_utils import write_json, write_rows
from .plotting import plot_evaluation, plot_learning, plot_trajectory
from .training import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate tabular SARSA rendezvous agents.")
    parser.add_argument("--episodes", type=int, default=4_000)
    parser.add_argument("--evaluation-episodes", type=int, default=250)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--quick", action="store_true", help="Fast code check; not report-quality data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = 150 if args.quick else args.episodes
    evaluation_episodes = 20 if args.quick else args.evaluation_episodes
    seeds = args.seeds[:1] if args.quick else args.seeds
    base = Config(episodes=episodes, evaluation_episodes=evaluation_episodes)
    output = base.output_path
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", base.as_dict())

    histories = []
    summary_rows: list[dict[str, float | int]] = []
    agents = []
    all_trials = []
    for seed in seeds:
        print(f"\n=== independent training seed {seed} ===")
        config = base.changed(seed=seed)
        agent, history = train(config)
        metrics, trials = evaluate(agent, config)
        row: dict[str, float | int] = {"seed": seed, **metrics.as_dict()}
        summary_rows.append(row)
        histories.append(history)
        agents.append(agent)
        all_trials.append(trials)
        agent.save(output / f"q_table_seed_{seed}.npz")
        print(row)

    write_rows(output / "evaluation_by_seed.csv", summary_rows)
    numeric_keys = [f.name for f in fields(type(metrics))]
    aggregate = {
        key: {
            "mean": float(np.mean([float(row[key]) for row in summary_rows])),
            "std": float(np.std([float(row[key]) for row in summary_rows], ddof=1))
            if len(summary_rows) > 1 else 0.0,
        }
        for key in numeric_keys
    }
    write_json(output / "evaluation_aggregate.json", aggregate)
    plot_learning(histories, base, output / "learning_curves.png")

    best_index = int(np.argmax([row["success_rate"] for row in summary_rows]))
    best_trials = all_trials[best_index]
    successful = [trial for trial in best_trials if trial.success]
    example = successful[0] if successful else best_trials[0]
    plot_trajectory(example, base, output / "example_trajectory.png")
    plot_evaluation(best_trials, output / "evaluation_histograms.png")
    print(f"\nResults written to {output.resolve()}")


if __name__ == "__main__":
    main()

