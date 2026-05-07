"""Generate figures for the regular-grid noisy-graphon experiment."""

import argparse
import csv
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from gmfs_core import build_grid_positions, radial_graphon_weights


EXPERIMENTS = {
    "clean": ("results_robotics_n100_clean", "Reference graphon", "tab:blue", "o", "-"),
    "gaussian": (
        "results_robotics_n100_gaussian",
        r"Gaussian noise ($\sigma=0.05$)",
        "tab:orange",
        "s",
        "--",
    ),
    "subgaussian": (
        "results_robotics_n100_subgaussian",
        r"Subgaussian noise ($\sigma=0.05$)",
        "tab:green",
        "^",
        "--",
    ),
}


def load_results(results_dir):
    data = {}
    for path in glob.glob(os.path.join(results_dir, "kappa_*", "data_full.json")):
        with open(path) as f:
            payload = json.load(f)
        for k_str, kd in payload.get("results", {}).items():
            data[int(k_str)] = kd
    return data


def load_summary_csv(path):
    data = {}
    if not os.path.exists(path):
        return data
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = row["condition"]
            data.setdefault(key, {})[int(row["kappa"])] = {
                "mean": float(row["mean"]),
                "stderr": float(row["stderr"]),
                "std": float(row.get("std", "nan")),
                "n_training_seeds": float(row.get("n_training_seeds", 1)),
                "returns": [0.0] * int(float(row.get("n_total_returns", 0))),
            }
    return data


def format_axis():
    plt.xlabel(r"$\kappa$ (subsample size)", fontsize=18)
    plt.tick_params(axis="both", labelsize=15)
    plt.grid(True, alpha=0.28)


def save_fig(out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, name + ".png"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, name + ".pdf"), bbox_inches="tight")
    plt.close()


def write_summary(all_results, out_dir):
    rows = []
    for key, results in all_results.items():
        for k in sorted(results):
            kd = results[k]
            rows.append(
                {
                    "condition": key,
                    "kappa": k,
                    "mean": kd["mean"],
                    "stderr": kd["stderr"],
                    "std": kd.get("std", float("nan")),
                    "n_training_seeds": kd.get("n_training_seeds", 1),
                    "n_total_returns": len(kd.get("returns", [])),
                }
            )
    if not rows:
        return
    with open(os.path.join(out_dir, "graphon_learning_noise_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_return_curves(all_results, out_dir):
    plt.figure(figsize=(10.2, 6.0))
    for key, results in all_results.items():
        _, label, color, marker, linestyle = EXPERIMENTS[key]
        ks = np.asarray(sorted(results), dtype=float)
        means = np.asarray([results[int(k)]["mean"] for k in ks])
        errs = np.asarray([results[int(k)]["stderr"] for k in ks])
        plt.errorbar(
            ks,
            means,
            yerr=errs,
            fmt=marker,
            linestyle=linestyle,
            color=color,
            capsize=5,
            capthick=1.6,
            elinewidth=1.6,
            linewidth=2.8,
            markersize=7.5,
            label=label,
        )
        plt.fill_between(ks, means - errs, means + errs, color=color, alpha=0.14)
    plt.ylabel("Discounted return", fontsize=18)
    format_axis()
    plt.legend(fontsize=14)
    save_fig(out_dir, "graphon_learning_noise_comparison")


def plot_gap(all_results, out_dir):
    if "clean" not in all_results:
        return
    clean = all_results["clean"]
    plt.figure(figsize=(10.2, 5.2))
    for key in ("gaussian", "subgaussian"):
        if key not in all_results:
            continue
        _, label, color, marker, _ = EXPERIMENTS[key]
        noisy = all_results[key]
        ks = np.asarray([k for k in sorted(clean) if k in noisy], dtype=float)
        gaps = np.asarray([clean[int(k)]["mean"] - noisy[int(k)]["mean"] for k in ks])
        errs = np.asarray(
            [
                np.hypot(clean[int(k)]["stderr"], noisy[int(k)]["stderr"])
                for k in ks
            ]
        )
        plt.errorbar(
            ks,
            gaps,
            yerr=errs,
            fmt=marker,
            linestyle="-",
            color=color,
            capsize=5,
            capthick=1.5,
            elinewidth=1.5,
            linewidth=2.6,
            markersize=7.0,
            label="Reference - " + label,
        )
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    plt.ylabel("Return gap", fontsize=18)
    format_axis()
    plt.legend(fontsize=13)
    save_fig(out_dir, "graphon_learning_noise_gap")


def perturb_weights(W, noise_type, sigma, seed):
    rng = np.random.default_rng(seed)
    noisy = W.copy()
    if noise_type == "gaussian":
        noisy = noisy + rng.normal(0.0, sigma, noisy.shape)
    elif noise_type == "subgaussian":
        noisy = noisy + rng.uniform(-sigma, sigma, noisy.shape)
    else:
        raise ValueError(noise_type)
    noisy = np.clip(noisy, 0.0, None)
    np.fill_diagonal(noisy, 0.0)
    row_sums = noisy.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return noisy / row_sums


def plot_weight_maps(out_dir):
    positions = build_grid_positions(n_agents=100, grid_rows=10, grid_cols=10)
    W = radial_graphon_weights(positions, radius=0.3)
    W_gauss = perturb_weights(W, "gaussian", 0.05, 12345)
    W_sub = perturb_weights(W, "subgaussian", 0.05, 12345)
    mats = [
        ("Reference graphon weights", W),
        (r"Gaussian perturbation", W_gauss),
        (r"Subgaussian perturbation", W_sub),
    ]
    vmax = max(np.quantile(M, 0.995) for _, M in mats)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), constrained_layout=True)
    for ax, (title, M) in zip(axes, mats):
        im = ax.imshow(M, cmap="magma", vmin=0.0, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("sampled agent index", fontsize=12)
        ax.set_ylabel("focal agent index", fontsize=12)
        ax.tick_params(axis="both", labelsize=9)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82)
    cbar.set_label("row-normalized weight", fontsize=12)
    save_fig(out_dir, "graphon_learning_weight_maps")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="exp_folder")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    all_results = {}
    for key, (results_dir, *_rest) in EXPERIMENTS.items():
        results = load_results(results_dir)
        if results:
            all_results[key] = results
    if not all_results:
        all_results = load_summary_csv("plots_n100_noise/n100_noise_summary.csv")
    plot_return_curves(all_results, args.out_dir)
    plot_gap(all_results, args.out_dir)
    plot_weight_maps(args.out_dir)
    write_summary(all_results, args.out_dir)


if __name__ == "__main__":
    main()
