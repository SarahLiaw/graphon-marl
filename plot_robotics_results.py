import argparse
import csv
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt


def load_kappa_results(results_dir):
    data = {}
    summary_path = os.path.join(results_dir, "data_full.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            payload = json.load(f)
        for k_str, k_data in payload.get("results", {}).items():
            data[int(k_str)] = k_data
    for path in glob.glob(os.path.join(results_dir, "kappa_*", "data_full.json")):
        with open(path, "r") as f:
            payload = json.load(f)
        results = payload.get("results", {})
        for k_str, k_data in results.items():
            k = int(k_str)
            data[k] = k_data
    return data

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def evenly_spaced_ks(available, count):
    available = sorted(set(available))
    if len(available) <= count:
        return available
    idx = np.linspace(0, len(available) - 1, num=count, dtype=int)
    return [available[i] for i in sorted(set(idx.tolist()))]


def select_ks(available, preferred=None, fallback_count=3):
    available = sorted(set(available))
    if not available:
        return []
    selected = []
    if preferred is not None:
        for k in preferred:
            if k in available:
                selected.append(k)
    if not selected:
        return evenly_spaced_ks(available, fallback_count)
    for endpoint in (available[0], available[-1]):
        if endpoint not in selected:
            selected.append(endpoint)
    selected = sorted(set(selected))
    if len(selected) < fallback_count:
        for k in evenly_spaced_ks(available, fallback_count):
            if k not in selected:
                selected.append(k)
        selected = sorted(set(selected))
    if len(selected) > fallback_count:
        selected = evenly_spaced_ks(selected, fallback_count)
    return selected


def save_fig(base_path):
    plt.savefig(base_path + ".png")
    plt.savefig(base_path + ".pdf")


def export_summary(results, out_dir):
    rows = []
    for k in sorted(results.keys()):
        rows.append({
            "kappa": k,
            "mean": results[k].get("mean"),
            "std": results[k].get("std"),
            "stderr": results[k].get("stderr"),
            "training_time": results[k].get("training_time"),
            "q_table_size": results[k].get("q_table_size"),
            "n_z": results[k].get("n_z"),
        })
    if not rows:
        return
    out_csv = os.path.join(out_dir, "kappa_summary.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    out_json = os.path.join(out_dir, "kappa_summary.json")
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)


def plot_reward_vs_k(results, out_dir):
    ks = sorted(results.keys())
    means = [results[k].get("mean") for k in ks]
    errs = [results[k].get("stderr", 0.0) for k in ks]
    if any(m is None for m in means):
        return
    plt.figure(figsize=(10, 6))
    plt.errorbar(ks, means, yerr=errs, fmt="o-", capsize=5)
    plt.xlabel("k (Subsample size)")
    plt.ylabel("Discounted Return")
    plt.title("Robotics GMFS: Discounted Return vs k")
    plt.grid(True, alpha=0.3)
    save_fig(os.path.join(out_dir, "discounted_reward_vs_k"))


def plot_bellman_delta(results, out_dir):
    ks = select_ks(results.keys(), fallback_count=3)
    plt.figure(figsize=(10, 6))
    plotted = False
    for k in ks:
        deltas = results[k].get("training_deltas")
        if not deltas:
            continue
        deltas = np.array(deltas, dtype=float)
        deltas = np.clip(deltas, 1e-12, None)
        plt.plot(np.log10(deltas), label=f"k={k}")
        plotted = True
    if plotted:
        plt.xlabel("Training Iteration")
        plt.ylabel("log10(Max Delta)")
        plt.title("Bellman Error Convergence")
        plt.grid(True, alpha=0.3)
        plt.legend()
        save_fig(os.path.join(out_dir, "bellman_delta_log"))


def plot_complexity(results, out_dir):
    ks = sorted(results.keys())
    times = [results[k].get("training_time") for k in ks]
    sizes = [results[k].get("q_table_size") for k in ks]
    if all(t is not None for t in times):
        plt.figure(figsize=(10, 6))
        plt.plot(ks, times, "o-")
        plt.xlabel("k (Subsample size)")
        plt.ylabel("Training Time (s)")
        plt.title("Training Time vs k")
        plt.grid(True, alpha=0.3)
        save_fig(os.path.join(out_dir, "complexity_time_vs_k"))
    if all(s is not None for s in sizes):
        plt.figure(figsize=(10, 6))
        plt.plot(ks, sizes, "o-")
        plt.xlabel("k (Subsample size)")
        plt.ylabel("Q-Table Size")
        plt.title("Q-Table Size vs k")
        plt.grid(True, alpha=0.3)
        save_fig(os.path.join(out_dir, "complexity_q_table_size_vs_k"))


def plot_policy_diagnostics(results, out_dir):
    ks = select_ks(results.keys(), fallback_count=2)
    plt.figure(figsize=(10, 6))
    plotted = False
    for k in ks:
        diagnostics = results[k].get("diagnostics")
        if not diagnostics:
            continue
        diag = diagnostics[0]
        if "avg_true_g2" in diag:
            plt.plot(diag["avg_true_g2"], label=f"true g2 (k={k})")
            plotted = True
        if "avg_hat_g2" in diag:
            plt.plot(diag["avg_hat_g2"], linestyle="--", label=f"hat g2 (k={k})")
            plotted = True
    if plotted:
        plt.xlabel("Time step")
        plt.ylabel("g(s=2)")
        plt.title("True vs Sampled g(s=2)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        save_fig(os.path.join(out_dir, "g2_true_vs_hat"))


def plot_tv_decay(results, out_dir):
    ks = sorted(results.keys())
    tv_means = []
    valid_ks = []
    for k in ks:
        diagnostics = results[k].get("diagnostics")
        if not diagnostics:
            continue
        diag = diagnostics[0]
        if "avg_tv" not in diag or not diag["avg_tv"]:
            continue
        valid_ks.append(k)
        tv_means.append(float(np.mean(diag["avg_tv"])))
    if not valid_ks:
        return
    tv_means = np.array(tv_means)
    plt.figure(figsize=(10, 6))
    plt.plot(valid_ks, tv_means, "o-", label="avg TV distance")
    # overlay scaled 1/sqrt(k)
    scale = tv_means[0] * np.sqrt(valid_ks[0])
    theory = scale / np.sqrt(np.array(valid_ks, dtype=float))
    plt.plot(valid_ks, theory, "--", label="scaled 1/sqrt(k)")
    plt.xlabel("k (Subsample size)")
    plt.ylabel("TV distance")
    plt.title("TV Distance Decay")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_fig(os.path.join(out_dir, "tv_distance_decay"))


def build_perception_heat(results, positions, k, max_steps):
    diagnostics = results[k].get("diagnostics")
    if not diagnostics:
        return None, None
    diag = diagnostics[0]
    neighbors_over_time = diag.get("focal_neighbors_over_time")
    focal_idx = diag.get("focal_idx")
    if neighbors_over_time is None or focal_idx is None:
        return None, None
    if max_steps is not None:
        neighbors_over_time = neighbors_over_time[:max_steps]
    sampled = []
    for neigh in neighbors_over_time:
        sampled.extend(neigh)
    sampled = np.array(sampled, dtype=int)
    sampled_positions = positions[sampled]
    heat, _, _ = np.histogram2d(sampled_positions[:, 0], sampled_positions[:, 1], bins=30, range=[[0, 1], [0, 1]])
    return heat, focal_idx


def plot_perception_heatmap(results, results_dir, out_dir, max_steps=10):
    graphon_path = os.path.join(results_dir, "graphon_data.npz")
    if not os.path.exists(graphon_path):
        return
    data = np.load(graphon_path)
    positions = data["positions"]
    radius = float(data["radius"])
    ks = select_ks(results.keys(), fallback_count=3)
    heatmaps = {}
    focal_indices = {}
    for k in ks:
        heat, focal_idx = build_perception_heat(results, positions, k, max_steps)
        if heat is None:
            continue
        heatmaps[k] = heat
        focal_indices[k] = focal_idx
    if not heatmaps:
        return
    vmax = max(np.max(h) for h in heatmaps.values())
    tick_fontsize = 15
    panel_fontsize = 20
    legend_fontsize = 14
    for k in ks:
        if k not in heatmaps:
            continue
        heat = heatmaps[k]
        focal_idx = focal_indices[k]
        plt.figure(figsize=(6, 6))
        plt.imshow(heat.T, origin="lower", extent=[0, 1, 0, 1], cmap="viridis", alpha=0.9, vmin=0, vmax=vmax)
        plt.scatter(positions[focal_idx, 0], positions[focal_idx, 1], c="gold", s=140, marker="*", edgecolor="black", label="focal agent")
        circle = plt.Circle((positions[focal_idx, 0], positions[focal_idx, 1]), radius, color="black", fill=False, linestyle="--", linewidth=1)
        plt.gca().add_artist(circle)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid(True, alpha=0.2)
        plt.tick_params(axis="both", labelsize=tick_fontsize)
        plt.legend(loc="best", fontsize=legend_fontsize)
        save_fig(os.path.join(out_dir, f"perception_heatmap_k{k}_first_{max_steps}"))

    # Combined comparison panel
    fig, axes = plt.subplots(1, len(heatmaps), figsize=(6.5 * len(heatmaps), 6.5), sharex=True, sharey=True)
    if len(heatmaps) == 1:
        axes = [axes]
    for ax, k in zip(axes, ks):
        if k not in heatmaps:
            continue
        heat = heatmaps[k]
        focal_idx = focal_indices[k]
        ax.imshow(heat.T, origin="lower", extent=[0, 1, 0, 1], cmap="viridis", alpha=0.9, vmin=0, vmax=vmax)
        ax.scatter(positions[focal_idx, 0], positions[focal_idx, 1], c="gold", s=140, marker="*", edgecolor="black")
        circle = plt.Circle((positions[focal_idx, 0], positions[focal_idx, 1]), radius, color="black", fill=False, linestyle="--", linewidth=1)
        ax.add_artist(circle)
        ax.set_title(rf"$\kappa={k}$", fontsize=panel_fontsize)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
    save_fig(os.path.join(out_dir, f"perception_heatmap_compare_first_{max_steps}"))


def plot_perception_time_evolution(results, results_dir, out_dir, k=24, steps_list=None):
    graphon_path = os.path.join(results_dir, "graphon_data.npz")
    if not os.path.exists(graphon_path):
        return
    data = np.load(graphon_path)
    positions = data["positions"]
    radius = float(data["radius"])
    if steps_list is None:
        steps_list = [1, 30, 60, 100]
    if k not in results:
        return
    diagnostics = results[k].get("diagnostics")
    if not diagnostics:
        return
    neighbors_over_time = diagnostics[0].get("focal_neighbors_over_time")
    focal_idx = diagnostics[0].get("focal_idx")
    if neighbors_over_time is None or focal_idx is None:
        return
    max_available = len(neighbors_over_time)
    steps_list = [s for s in steps_list if s <= max_available]
    if not steps_list:
        return
    heatmaps = {}
    for s in steps_list:
        heat, _ = build_perception_heat(results, positions, k, s)
        if heat is not None:
            heatmaps[s] = heat
    if not heatmaps:
        return
    vmax = max(np.max(h) for h in heatmaps.values())
    fig, axes = plt.subplots(1, len(steps_list), figsize=(5 * len(steps_list), 5), sharex=True, sharey=True)
    if len(steps_list) == 1:
        axes = [axes]
    for ax, s in zip(axes, steps_list):
        heat = heatmaps.get(s)
        if heat is None:
            continue
        ax.imshow(heat.T, origin="lower", extent=[0, 1, 0, 1], cmap="viridis", alpha=0.9, vmin=0, vmax=vmax)
        ax.scatter(positions[focal_idx, 0], positions[focal_idx, 1], c="gold", s=120, marker="*", edgecolor="black")
        circle = plt.Circle((positions[focal_idx, 0], positions[focal_idx, 1]), radius, color="black", fill=False, linestyle="--", linewidth=1)
        ax.add_artist(circle)
        ax.set_title(f"t = {s}", fontsize=16)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=12)
    save_fig(os.path.join(out_dir, f"perception_time_evolution_k{k}"))


def plot_perception_time_evolution_compare(results, results_dir, out_dir, ks=(2, 8, 24), steps_list=None):
    graphon_path = os.path.join(results_dir, "graphon_data.npz")
    if not os.path.exists(graphon_path):
        return
    data = np.load(graphon_path)
    positions = data["positions"]
    radius = float(data["radius"])
    if steps_list is None:
        steps_list = [1, 30, 60, 100]
    k_list = [k for k in ks if k in results]
    if len(k_list) < 2:
        return
    heatmaps = {}
    focal_idx = None
    max_available = None
    for k in k_list:
        diagnostics = results[k].get("diagnostics")
        if not diagnostics:
            return
        neighbors_over_time = diagnostics[0].get("focal_neighbors_over_time")
        focal_idx_k = diagnostics[0].get("focal_idx")
        if neighbors_over_time is None or focal_idx_k is None:
            return
        if focal_idx is None:
            focal_idx = focal_idx_k
        max_available = len(neighbors_over_time) if max_available is None else min(max_available, len(neighbors_over_time))
    steps_list = [s for s in steps_list if s <= max_available]
    if not steps_list:
        return
    vmax = 0.0
    for k in k_list:
        for s in steps_list:
            heat, _ = build_perception_heat(results, positions, k, s)
            if heat is not None:
                heatmaps[(k, s)] = heat
                vmax = max(vmax, float(np.max(heat)))
    if not heatmaps:
        return
    fig, axes = plt.subplots(len(k_list), len(steps_list), figsize=(5.5 * len(steps_list), 5.5 * len(k_list)), sharex=True, sharey=True)
    if len(k_list) == 1:
        axes = [axes]
    mappable = None
    for row, k in enumerate(k_list):
        for col, s in enumerate(steps_list):
            ax = axes[row][col] if len(steps_list) > 1 else axes[row]
            heat = heatmaps.get((k, s))
            if heat is None:
                ax.axis("off")
                continue
            im = ax.imshow(heat.T, origin="lower", extent=[0, 1, 0, 1], cmap="viridis", alpha=0.9, vmin=0, vmax=vmax)
            mappable = im
            ax.scatter(positions[focal_idx, 0], positions[focal_idx, 1], c="gold", s=80, marker="*", edgecolor="black")
            circle = plt.Circle((positions[focal_idx, 0], positions[focal_idx, 1]), radius, color="black", fill=False, linestyle="--", linewidth=1)
            ax.add_artist(circle)
            if row == 0:
                ax.set_title(f"t = {s}", fontsize=18)
            if col == 0:
                ax.set_ylabel(r"$\kappa$ = " + f"{k}", fontsize=18)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=12)
    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes, fraction=0.02, pad=0.02)
        cbar.set_label("Sample density (counts)", fontsize=14)
        cbar.ax.tick_params(labelsize=12)
    suffix = "_".join(f"k{k}" for k in k_list)
    save_fig(os.path.join(out_dir, f"perception_time_evolution_compare_{suffix}"))


def plot_graphon_heatmap(results_dir, out_dir):
    graphon_path = os.path.join(results_dir, "graphon_data.npz")
    if not os.path.exists(graphon_path):
        return
    data = np.load(graphon_path)
    positions = data["positions"]
    radius = float(data["radius"])
    n_agents = positions.shape[0]
    W = np.zeros((n_agents, n_agents), dtype=float)
    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                continue
            if np.linalg.norm(positions[i] - positions[j]) <= radius:
                W[i, j] = 1.0
    plt.figure(figsize=(6, 5))
    plt.imshow(W, cmap="viridis")
    plt.colorbar(label="W(x_i, x_j)")
    plt.title("Graphon Weight Matrix")
    plt.xlabel("j")
    plt.ylabel("i")
    save_fig(os.path.join(out_dir, "graphon_weight_heatmap"))


def plot_warehouse_heatmap(results, results_dir, out_dir):
    graphon_path = os.path.join(results_dir, "graphon_data.npz")
    if not os.path.exists(graphon_path):
        return
    data = np.load(graphon_path)
    positions = data["positions"]
    grid_size = int(data["grid_size"])
    radius = float(data["radius"])
    ks = select_ks(results.keys(), fallback_count=2)
    for k in ks:
        diagnostics = results[k].get("diagnostics")
        if not diagnostics:
            continue
        diag = diagnostics[0]
        states = diag.get("agent_states_t_end")
        if states is None:
            continue
        states = np.array(states, dtype=int)
        plt.figure(figsize=(6, 6))
        plt.scatter(positions[:, 0], positions[:, 1], c="lightgray", s=40)
        working = np.where(states == 2)[0]
        for idx in working:
            circle = plt.Circle((positions[idx, 0], positions[idx, 1]), radius, color="red", alpha=0.15, fill=True)
            plt.gca().add_artist(circle)
        plt.scatter(positions[working, 0], positions[working, 1], c="red", s=70, label="Working")
        plt.title(f"Coordination Heatmap (k={k})")
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid(True, alpha=0.2)
        plt.legend(loc="best")
        save_fig(os.path.join(out_dir, f"coordination_heatmap_k{k}"))


def plot_coordination_compare(results, results_dir, out_dir):
    graphon_path = os.path.join(results_dir, "graphon_data.npz")
    if not os.path.exists(graphon_path):
        return
    data = np.load(graphon_path)
    positions = data["positions"]
    radius = float(data["radius"])
    ks = select_ks(results.keys(), fallback_count=2)
    if len(ks) < 2:
        return
    k_left, k_right = ks[0], ks[-1]
    diag_left = results[k_left].get("diagnostics", [])
    diag_right = results[k_right].get("diagnostics", [])
    if not diag_left or not diag_right:
        return
    states_left = np.array(diag_left[0].get("agent_states_t_end", []), dtype=int)
    states_right = np.array(diag_right[0].get("agent_states_t_end", []), dtype=int)
    if states_left.size == 0 or states_right.size == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    for ax, k, states in [(axes[0], k_left, states_left), (axes[1], k_right, states_right)]:
        ax.scatter(positions[:, 0], positions[:, 1], c="lightgray", s=40)
        working = np.where(states == 2)[0]
        for idx in working:
            circle = plt.Circle((positions[idx, 0], positions[idx, 1]), radius, color="red", alpha=0.15, fill=True)
            ax.add_artist(circle)
        ax.scatter(positions[working, 0], positions[working, 1], c="red", s=70, label="Working")
        ax.set_title(f"k={k} (same seed run)")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
    axes[0].legend(loc="best")
    fig.suptitle("Coordination Heatmap Comparison")
    save_fig(os.path.join(out_dir, f"coordination_compare_k{k_left}_k{k_right}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results_robotics_n1000")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir
    out_dir = args.out_dir or os.path.join(results_dir, "plots")
    ensure_dir(out_dir)

    results = load_kappa_results(results_dir)
    if not results:
        print("No results found to plot.")
        return

    export_summary(results, out_dir)
    plot_reward_vs_k(results, out_dir)
    plot_bellman_delta(results, out_dir)
    plot_complexity(results, out_dir)
    plot_policy_diagnostics(results, out_dir)
    plot_tv_decay(results, out_dir)
    plot_graphon_heatmap(results_dir, out_dir)
    plot_perception_heatmap(results, results_dir, out_dir)
    for k in select_ks(results.keys(), fallback_count=2):
        plot_perception_time_evolution(results, results_dir, out_dir, k=k)
    plot_perception_time_evolution_compare(
        results,
        results_dir,
        out_dir,
        ks=tuple(select_ks(results.keys(), fallback_count=3)),
    )
    plot_warehouse_heatmap(results, results_dir, out_dir)
    plot_coordination_compare(results, results_dir, out_dir)


if __name__ == "__main__":
    main()
