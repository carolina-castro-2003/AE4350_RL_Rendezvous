from __future__ import annotations

import argparse
from dataclasses import replace

from .ac_evaluation import evaluate_actor_critic
from .ac_training import train_actor_critic
from .actor_critic import ActorCriticConfig
from .config import Config
from .io_utils import write_rows
from .plotting import plot_sensitivity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-factor actor-critic sensitivity study.")
    parser.add_argument("--episodes", type=int, default=3_000)
    parser.add_argument("--evaluation-episodes", type=int, default=120)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = 150 if args.quick else args.episodes
    evaluation_episodes = 15 if args.quick else args.evaluation_episodes
    seeds = args.seeds[:1] if args.quick else args.seeds

    base = Config(
        episodes=episodes,
        evaluation_episodes=evaluation_episodes,
        results_dir="results_actor_critic",
    )
    base_ac = ActorCriticConfig()
    output = base.output_path
    output.mkdir(parents=True, exist_ok=True)

    sweeps: dict[str, list[float]] = {
        "actor_alpha": [0.006, 0.010, 0.018],
        "critic_alpha": [0.035, 0.060, 0.090],
        "sigma_start": [0.50, 0.85, 1.15],
        "fuel_penalty": [0.06, 0.12, 0.24],
    }

    rows: list[dict[str, float | int | str]] = []
    for parameter, values in sweeps.items():
        for value in values:
            for seed in seeds:
                print(f"\n=== {parameter}={value} seed={seed} ===")
                config = base.changed(seed=seed)
                ac_config = base_ac
                if hasattr(base, parameter):
                    config = config.changed(**{parameter: value})
                else:
                    ac_config = replace(base_ac, **{parameter: value})

                agent, _history = train_actor_critic(config, ac_config, verbose=False)
                metrics, _trials = evaluate_actor_critic(agent, config)
                row: dict[str, float | int | str] = {
                    "parameter": parameter,
                    "value": value,
                    "seed": seed,
                    **metrics.as_dict(),
                }
                rows.append(row)
                print(row)

    write_rows(output / "actor_critic_sensitivity_raw.csv", rows)
    plot_sensitivity(rows, output / "actor_critic_sensitivity_success.png")
    print(f"\nActor-critic sensitivity results written to {output.resolve()}")


if __name__ == "__main__":
    main()
