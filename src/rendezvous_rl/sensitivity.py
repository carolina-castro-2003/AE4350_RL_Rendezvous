from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config
from .evaluation import evaluate
from .io_utils import write_rows
from .plotting import plot_sensitivity
from .training import train


PARAMETERS: dict[str, tuple[float, ...]] = {
    "safety_penalty": (2.0, 10.0, 30.0),
    "fuel_penalty": (0.02, 0.12, 0.35),
    "keep_out_radius": (3.0, 5.0, 8.0),
    "alpha": (0.05, 0.15, 0.35),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeated one-factor sensitivity experiment.")
    parser.add_argument("--episodes", type=int, default=2_500)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = 80 if args.quick else args.episodes
    evaluation_episodes = 10 if args.quick else args.evaluation_episodes
    seeds = args.seeds[:1] if args.quick else args.seeds
    base = Config(episodes=episodes, evaluation_episodes=evaluation_episodes)
    output = Path(base.results_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []

    for parameter, values in PARAMETERS.items():
        for value in values:
            for seed in seeds:
                print(f"{parameter}={value:g}, seed={seed}")
                config = base.changed(**{parameter: value, "seed": seed})
                agent, _ = train(config, verbose=False)
                metrics, _ = evaluate(agent, config)
                rows.append({
                    "parameter": parameter,
                    "value": value,
                    "seed": seed,
                    **metrics.as_dict(),
                })

    write_rows(output / "sensitivity_raw.csv", rows)
    plot_sensitivity(rows, output / "sensitivity_success.png")
    print(f"Sensitivity results written to {output.resolve()}")


if __name__ == "__main__":
    main()

