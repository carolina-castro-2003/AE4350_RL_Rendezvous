from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .ac_evaluation import ContinuousEpisode
from .ac_training import ActorCriticHistory
from .config import Config
from .plotting import moving_average


def plot_actor_critic_learning(
    histories: Sequence[ActorCriticHistory],
    config: Config,
    path: Path,
) -> None:
    fields = [
        ("returns", "Episode return"),
        ("successes", "Success rate (%)"),
        ("fuel", "Normalised impulse"),
        ("sigma", "Exploration std"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, (field, label) in zip(axes.flat, fields):
        curves = np.array(
            [
                moving_average(
                    np.asarray(getattr(history, field), dtype=float),
                    config.smoothing_window,
                )
                for history in histories
            ],
            dtype=np.float64,
        )
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
    fig.suptitle(f"Continuous actor-critic learning curves ({len(histories)} seeds)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_actor_critic_trajectory(
    episode: ContinuousEpisode,
    config: Config,
    path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    theta = np.linspace(0, 2 * np.pi, 300)
    axis.fill(
        config.keep_out_radius * np.cos(theta),
        config.keep_out_radius * np.sin(theta),
        color="mistyrose",
        alpha=0.8,
        label="Keep-out zone",
    )
    axis.fill(
        config.docking_radius * np.cos(theta),
        config.docking_radius * np.sin(theta),
        color="palegreen",
        alpha=0.9,
        label="Docking region",
    )
    states = episode.states
    axis.plot(states[:, 0], states[:, 1], lw=1.8, label="Chaser")
    skip = max(1, len(episode.actions) // 25)
    if len(episode.actions):
        quiver_states = states[:-1:skip]
        quiver_actions = episode.actions[::skip]
        scale = max(config.thrust_acceleration, 1e-9)
        axis.quiver(
            quiver_states[:, 0],
            quiver_states[:, 1],
            quiver_actions[:, 0] / scale,
            quiver_actions[:, 1] / scale,
            angles="xy",
            scale_units="xy",
            scale=3.5,
            width=0.003,
            alpha=0.55,
            color="tab:orange",
            label="Thrust direction",
        )
    axis.scatter(*states[0, :2], s=60, c="black", label="Start")
    axis.scatter(*states[-1, :2], s=80, marker="*", c="blue", label="End")
    axis.set(
        xlabel="Radial relative position x [m]",
        ylabel="Along-track relative position y [m]",
        title=(
            f"Actor-critic trajectory: success={episode.success}, "
            f"impulse={episode.fuel:.1f}, final speed={episode.final_speed:.3f} m/s"
        ),
        aspect="equal",
    )
    axis.grid(alpha=0.3)
    axis.legend(loc="best")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_actor_critic_evaluation(
    episodes: Sequence[ContinuousEpisode],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    data = [
        ([e.final_distance for e in episodes], "Final distance [m]"),
        ([e.final_speed for e in episodes], "Final speed [m/s]"),
        ([e.fuel for e in episodes], "Normalised impulse"),
    ]
    for axis, (values, label) in zip(axes, data):
        axis.hist(values, bins=20, edgecolor="white")
        axis.set(xlabel=label, ylabel="Count")
        axis.grid(alpha=0.25)
    fig.suptitle("Actor-critic deterministic-policy evaluation")
    fig.savefig(path, dpi=180)
    plt.close(fig)
