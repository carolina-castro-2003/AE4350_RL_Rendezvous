from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Any

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
        safety_penalty=40.0,
        collision_penalty=800.0,
        speed_weight=4.0,
    )
    base_ac = ActorCriticConfig()
    output = base.output_path
    output.mkdir(parents=True, exist_ok=True)

    sweeps: dict[str, list[float]] = {
        "actor_alpha": [0.0010, 0.0025, 0.0050],
        "critic_alpha": [0.010, 0.020, 0.040],
        "sigma_start": [0.25, 0.45, 0.70],
        "warm_start_scale": [0.0, 0.5, 1.0],
        "fuel_penalty": [0.06, 0.12, 0.24],
    }

    baseline_rows: list[dict[str, float | int | str]] = []
    baseline_by_seed: dict[int, dict[str, float | int | str]] = {}
    print("\n=== baseline actor-critic configuration ===")
    for seed in seeds:
        print(f"\n=== baseline seed={seed} ===")
        config = base.changed(seed=seed)
        agent, _history = train_actor_critic(config, base_ac, verbose=False)
        metrics, _trials = evaluate_actor_critic(agent, config)
        row: dict[str, float | int | str] = {
            "parameter": "baseline",
            "value": "baseline",
            "seed": seed,
            "source": "trained",
            **metrics.as_dict(),
            "delta_success_rate": 0.0,
            "delta_mean_return": 0.0,
        }
        baseline_rows.append(row)
        baseline_by_seed[seed] = row
        print(row)

    rows: list[dict[str, float | int | str]] = []
    unique_run_rows: list[dict[str, float | int | str]] = baseline_rows.copy()
    for parameter, values in sweeps.items():
        default_value = _default_value(parameter, base, base_ac)
        for value in values:
            for seed in seeds:
                print(f"\n=== {parameter}={value} seed={seed} ===")
                config = base.changed(seed=seed)
                ac_config = base_ac
                if hasattr(base, parameter):
                    config = config.changed(**{parameter: value})
                else:
                    ac_config = replace(base_ac, **{parameter: value})

                baseline = baseline_by_seed[seed]
                is_default_case = default_value is not None and float(value) == float(default_value)
                if is_default_case:
                    metrics_dict = {
                        key: baseline[key]
                        for key in (
                            "success_rate",
                            "collision_rate",
                            "escape_rate",
                            "mean_final_distance",
                            "mean_final_speed",
                            "mean_fuel",
                            "mean_safety_violations",
                            "mean_return",
                        )
                    }
                    source = "baseline_reused"
                else:
                    agent, _history = train_actor_critic(config, ac_config, verbose=False)
                    metrics, _trials = evaluate_actor_critic(agent, config)
                    metrics_dict = metrics.as_dict()
                    source = "trained"

                row: dict[str, float | int | str] = {
                    "parameter": parameter,
                    "value": value,
                    "seed": seed,
                    "source": source,
                    **metrics_dict,
                    "delta_success_rate": float(metrics_dict["success_rate"]) - float(baseline["success_rate"]),
                    "delta_mean_return": float(metrics_dict["mean_return"]) - float(baseline["mean_return"]),
                }
                rows.append(row)
                if source == "trained":
                    unique_run_rows.append(row)
                print(row)

    write_rows(output / "actor_critic_sensitivity_baseline.csv", baseline_rows)
    write_rows(output / "actor_critic_sensitivity_raw.csv", rows)
    write_rows(output / "actor_critic_sensitivity_unique_runs.csv", unique_run_rows)
    plot_sensitivity(rows, output / "actor_critic_sensitivity_success.png")
    print(f"\nActor-critic sensitivity results written to {output.resolve()}")


def _default_value(parameter: str, base: Config, base_ac: ActorCriticConfig) -> Any:
    if hasattr(base, parameter):
        return getattr(base, parameter)
    if hasattr(base_ac, parameter):
        return getattr(base_ac, parameter)
    return None


if __name__ == "__main__":
    main()
