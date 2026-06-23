from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .config import Config
from .evaluation import Episode
from .training import TrainingHistory


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    window = max(1, min(window, data.size))
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def plot_learning(histories: Sequence[TrainingHistory], config: Config, path: Path) -> None:
    fields = [
        ("returns", "Episode return"),
        ("successes", "Success rate (%)"),
        ("fuel", "Thrust commands"),
        ("safety_violations", "Unsafe entries"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, (field, label) in zip(axes.flat, fields):
        curves = np.array([
            moving_average(np.asarray(getattr(h, field), dtype=float), config.smoothing_window)
            for h in histories
        ])
        if field == "successes":
            curves *= 100.0
        x = np.arange(curves.shape[1]) + config.smoothing_window
        mean, std = curves.mean(axis=0), curves.std(axis=0)
        axis.plot(x, mean, lw=1.8)
        axis.fill_between(x, mean - std, mean + std, alpha=0.25, label="±1 SD")
        axis.set(xlabel="Training episode", ylabel=label)
        axis.grid(alpha=0.3)
        if len(histories) > 1:
            axis.legend()
    fig.suptitle(f"SARSA learning curves ({len(histories)} independent seeds)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_trajectory(episode: Episode, config: Config, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    theta = np.linspace(0, 2 * np.pi, 300)
    axis.fill(
        config.keep_out_radius * np.cos(theta),
        config.keep_out_radius * np.sin(theta),
        color="mistyrose", alpha=0.8, label="Keep-out zone",
    )
    axis.fill(
        config.docking_radius * np.cos(theta),
        config.docking_radius * np.sin(theta),
        color="palegreen", alpha=0.9, label="Docking region",
    )
    axis.plot(episode.states[:, 0], episode.states[:, 1], lw=1.8, label="Chaser")
    axis.scatter(*episode.states[0, :2], s=60, c="black", label="Start")
    axis.scatter(*episode.states[-1, :2], s=80, marker="*", c="blue", label="End")
    axis.set(
        xlabel="Radial relative position x [m]",
        ylabel="Along-track relative position y [m]",
        title=(f"Greedy trajectory: success={episode.success}, fuel={episode.fuel}, "
               f"final speed={episode.final_speed:.3f} m/s"),
        aspect="equal",
    )
    axis.grid(alpha=0.3)
    axis.legend(loc="best")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_evaluation(episodes: Sequence[Episode], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    data = [
        ([e.final_distance for e in episodes], "Final distance [m]"),
        ([e.final_speed for e in episodes], "Final speed [m/s]"),
        ([e.fuel for e in episodes], "Thrust commands"),
    ]
    for axis, (values, label) in zip(axes, data):
        axis.hist(values, bins=20, edgecolor="white")
        axis.set(xlabel=label, ylabel="Count")
        axis.grid(alpha=0.25)
    fig.suptitle("Greedy-policy evaluation on unseen initial conditions")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_sensitivity(rows: list[dict[str, float | int | str]], path: Path) -> None:
    parameters = list(dict.fromkeys(str(row["parameter"]) for row in rows))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, parameter in zip(axes.flat, parameters):
        subset = [r for r in rows if r["parameter"] == parameter]
        values = sorted(set(float(r["value"]) for r in subset))
        samples = [
            [100 * float(r["success_rate"]) for r in subset if float(r["value"]) == value]
            for value in values
        ]
        means = [np.mean(s) for s in samples]
        stds = [np.std(s, ddof=1) if len(s) > 1 else 0.0 for s in samples]
        axis.errorbar(values, means, yerr=stds, marker="o", capsize=4)
        axis.set(xlabel=parameter.replace("_", " "), ylabel="Success rate [%]", ylim=(0, 100))
        axis.grid(alpha=0.3)
    fig.suptitle("One-factor-at-a-time sensitivity: mean ± SD across seeds")
    fig.savefig(path, dpi=180)
    plt.close(fig)

