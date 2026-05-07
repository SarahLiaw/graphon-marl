import argparse
import csv
import json
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm


RESULT_DIRS = {
    "graphon": "results_robotics_n1000_localized_graphon",
    "uniform": "results_robotics_n1000_localized_uniform",
    "gaussian": "results_robotics_n1000_localized_gaussian",
    "subgaussian": "results_robotics_n1000_localized_subgaussian",
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_results(results_dir):
    results = {}
    for path in glob(os.path.join(results_dir, "kappa_*", "data_full.json")):
        with open(path, "r") as f:
            payload = json.load(f)
        for k_str, value in payload.get("results", {}).items():
            results[int(k_str)] = value
    return results


def save_fig(out_dir, name):
    base = os.path.join(out_dir, name)
    plt.tight_layout()
    plt.savefig(base + ".pdf", bbox_inches="tight")
    plt.savefig(base + ".png", dpi=220, bbox_inches="tight")


def format_kappa_axis(ks):
    plt.xscale("log")
    plt.xticks(ks, [str(k) for k in ks], fontsize=12)
    plt.xlim(min(ks) * 0.85, max(ks) * 1.15)


def write_rows(out_dir, name, rows):
    if not rows:
        return
    with open(os.path.join(out_dir, name), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def series(results, ks):
    means = np.array([results[k]["mean"] for k in ks], dtype=float)
    errs = np.array([results[k].get("stderr", 0.0) for k in ks], dtype=float)
    return means, errs


def plot_return_vs_kappa(all_results, out_dir):
    graphon = all_results["graphon"]
    uniform = all_results["uniform"]
    ks = sorted(set(graphon) & set(uniform))
    g_mean, g_err = series(graphon, ks)
    u_mean, u_err = series(uniform, ks)

    plt.figure(figsize=(8.8, 5.6))
    plt.errorbar(
        ks,
        g_mean,
        yerr=g_err,
        fmt="o-",
        linewidth=2.4,
        markersize=6,
        capsize=5,
        elinewidth=1.4,
        capthick=1.4,
        label="Graphon-weighted",
    )
    plt.fill_between(ks, g_mean - g_err, g_mean + g_err, alpha=0.18)
    plt.errorbar(
        ks,
        u_mean,
        yerr=u_err,
        fmt="s--",
        linewidth=2.4,
        markersize=6,
        capsize=5,
        elinewidth=1.4,
        capthick=1.4,
        label="Uniform",
    )
    plt.fill_between(ks, u_mean - u_err, u_mean + u_err, alpha=0.18)
    plt.xlabel(r"$\kappa$ (subsample size)", fontsize=17)
    plt.ylabel("Discounted return", fontsize=17)
    format_kappa_axis(ks)
    plt.yticks(fontsize=13)
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(fontsize=13)
    save_fig(out_dir, "localized_return_vs_kappa")

    rows = []
    for k in ks:
        rows.append(
            {
                "kappa": k,
                "graphon_mean": graphon[k]["mean"],
                "graphon_stderr": graphon[k].get("stderr", 0.0),
                "uniform_mean": uniform[k]["mean"],
                "uniform_stderr": uniform[k].get("stderr", 0.0),
                "graphon_minus_uniform": graphon[k]["mean"] - uniform[k]["mean"],
            }
        )
    write_rows(out_dir, "localized_return_vs_kappa.csv", rows)


def plot_noise(all_results, out_dir):
    labels = {
        "graphon": "Reference graphon",
        "gaussian": r"Gaussian noise ($\sigma=0.05$)",
        "subgaussian": r"Subgaussian noise ($\sigma=0.05$)",
    }
    styles = {
        "graphon": ("o-", "tab:blue"),
        "gaussian": ("s--", "tab:orange"),
        "subgaussian": ("^--", "tab:green"),
    }
    common = set(all_results["graphon"])
    common &= set(all_results["gaussian"])
    common &= set(all_results["subgaussian"])
    ks = sorted(common)

    plt.figure(figsize=(8.8, 5.6))
    for key in ("graphon", "gaussian", "subgaussian"):
        mean, err = series(all_results[key], ks)
        fmt, color = styles[key]
        plt.errorbar(
            ks,
            mean,
            yerr=err,
            fmt=fmt,
            color=color,
            linewidth=2.2,
            markersize=6,
            capsize=5,
            elinewidth=1.4,
            capthick=1.4,
            label=labels[key],
        )
        plt.fill_between(ks, mean - err, mean + err, color=color, alpha=0.14)
    plt.xlabel(r"$\kappa$ (subsample size)", fontsize=17)
    plt.ylabel("Discounted return", fontsize=17)
    format_kappa_axis(ks)
    plt.yticks(fontsize=13)
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(fontsize=12)
    save_fig(out_dir, "localized_noise_robustness")

    rows = []
    for key in ("graphon", "gaussian", "subgaussian"):
        for k in ks:
            rows.append(
                {
                    "condition": key,
                    "kappa": k,
                    "mean": all_results[key][k]["mean"],
                    "stderr": all_results[key][k].get("stderr", 0.0),
                    "std": all_results[key][k].get("std", 0.0),
                    "n_returns": len(all_results[key][k].get("returns", [])),
                }
            )
    write_rows(out_dir, "localized_noise_robustness.csv", rows)

    clean = all_results["graphon"]
    clean_mean, clean_err = series(clean, ks)
    plt.figure(figsize=(8.8, 4.8))
    for key, label, color in (
        ("gaussian", "Reference - Gaussian", "tab:orange"),
        ("subgaussian", "Reference - Subgaussian", "tab:green"),
    ):
        noisy_mean, noisy_err = series(all_results[key], ks)
        gaps = clean_mean - noisy_mean
        gap_err = np.sqrt(clean_err**2 + noisy_err**2)
        plt.errorbar(
            ks,
            gaps,
            yerr=gap_err,
            fmt="o-",
            color=color,
            linewidth=2.2,
            markersize=5,
            capsize=5,
            elinewidth=1.4,
            capthick=1.4,
            label=label,
        )
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel(r"$\kappa$ (subsample size)", fontsize=17)
    plt.ylabel("Return gap", fontsize=17)
    format_kappa_axis(ks)
    plt.yticks(fontsize=13)
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(fontsize=12)
    save_fig(out_dir, "localized_noise_gap")


def build_heat(results, positions, k, steps=10, bins=42, normalize=True):
    diag = results[k].get("diagnostics", [{}])[0]
    neighbors_over_time = diag.get("focal_neighbors_over_time")
    focal_idx = diag.get("focal_idx")
    if neighbors_over_time is None or focal_idx is None:
        return None, None
    sampled = []
    for neighbors in neighbors_over_time[:steps]:
        sampled.extend(neighbors)
    sampled = np.array(sampled, dtype=int)
    heat, _, _ = np.histogram2d(
        positions[sampled, 0],
        positions[sampled, 1],
        bins=bins,
        range=[[0, 1], [0, 1]],
    )
    if normalize and heat.sum() > 0:
        heat = heat / heat.sum()
    return heat, focal_idx


def plot_perception(all_results, out_dir):
    data = np.load(os.path.join(RESULT_DIRS["graphon"], "graphon_data.npz"))
    positions = data["positions"]
    radius = float(data["radius"])
    results = all_results["graphon"]
    ks = [k for k in (1, 10, 30, 100) if k in results]
    heatmaps = {}
    focal_idx = None
    for k in ks:
        heat, focal = build_heat(results, positions, k, steps=10)
        if heat is not None:
            heatmaps[k] = heat
            focal_idx = focal if focal_idx is None else focal_idx
    if not heatmaps:
        return

    vmax = max(float(np.max(h)) for h in heatmaps.values())
    focal = positions[focal_idx]

    def draw(name, zoom):
        fig, axes = plt.subplots(
            1,
            len(heatmaps),
            figsize=(4.6 * len(heatmaps), 4.4),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        if len(heatmaps) == 1:
            axes = [axes]
        mappable = None
        for ax, k in zip(axes, heatmaps):
            heat = heatmaps[k]
            im = ax.imshow(
                heat.T,
                origin="lower",
                extent=[0, 1, 0, 1],
                cmap="viridis",
                norm=PowerNorm(gamma=0.55, vmin=0.0, vmax=vmax),
                alpha=0.95,
            )
            mappable = im
            ax.scatter(
                focal[0],
                focal[1],
                c="gold",
                s=120,
                marker="*",
                edgecolor="black",
                linewidth=0.7,
                zorder=3,
            )
            circle = plt.Circle(
                (focal[0], focal[1]),
                radius,
                color="white",
                fill=False,
                linestyle="--",
                linewidth=1.4,
            )
            ax.add_artist(circle)
            ax.set_title(rf"$\kappa={k}$", fontsize=17)
            if zoom:
                margin = radius * 1.65
                ax.set_xlim(max(0.0, focal[0] - margin), min(1.0, focal[0] + margin))
                ax.set_ylim(max(0.0, focal[1] - margin), min(1.0, focal[1] + margin))
            else:
                ax.set_xlim(-0.02, 1.02)
                ax.set_ylim(-0.02, 1.02)
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(axis="both", labelsize=11)
        if mappable is not None:
            cbar = fig.colorbar(mappable, ax=axes, fraction=0.025, pad=0.02)
            cbar.set_label("Empirical sample mass", fontsize=13)
            cbar.ax.tick_params(labelsize=11)
        base = os.path.join(out_dir, name)
        fig.savefig(base + ".pdf", bbox_inches="tight")
        fig.savefig(base + ".png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    draw("localized_perception_heatmap", zoom=True)
    draw("localized_perception_heatmap_full", zoom=False)


def plot_layout(out_dir):
    data = np.load(os.path.join(RESULT_DIRS["graphon"], "graphon_data.npz"))
    positions = data["positions"]
    cfg_path = os.path.join(RESULT_DIRS["graphon"], "config_robotics_n1000_localized_graphon.json")
    if not os.path.exists(cfg_path):
        cfg_path = "configs/config_robotics_n1000_localized_graphon.json"
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    split = float(np.quantile(positions[:, 0], cfg["initial_state_params"]["split_quantile"]))

    plt.figure(figsize=(6.2, 5.2))
    left = positions[:, 0] <= split
    plt.scatter(positions[~left, 0], positions[~left, 1], s=11, alpha=0.55, label="Low initial work density")
    plt.scatter(positions[left, 0], positions[left, 1], s=11, alpha=0.65, label="High initial work density")
    plt.axvline(split, color="black", linestyle="--", linewidth=1.2)
    plt.xlabel(r"latent position $\alpha_x$", fontsize=16)
    plt.ylabel(r"latent position $\alpha_y$", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.22)
    plt.legend(fontsize=11, loc="upper right")
    save_fig(out_dir, "localized_powerlaw_layout")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="exp_folder")
    args = parser.parse_args()
    ensure_dir(args.out_dir)
    all_results = {key: load_results(path) for key, path in RESULT_DIRS.items()}
    plot_return_vs_kappa(all_results, args.out_dir)
    plot_noise(all_results, args.out_dir)
    plot_perception(all_results, args.out_dir)
    plot_layout(args.out_dir)


if __name__ == "__main__":
    main()
