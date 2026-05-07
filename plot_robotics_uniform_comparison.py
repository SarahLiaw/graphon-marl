import argparse
import csv
import os

import matplotlib.pyplot as plt

from plot_robotics_results import load_kappa_results


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_comparison_csv(graphon_results, uniform_results, out_dir, prefix):
    rows = []
    for k in sorted(set(graphon_results) | set(uniform_results)):
        graphon = graphon_results.get(k, {})
        uniform = uniform_results.get(k, {})
        rows.append(
            {
                "kappa": k,
                "graphon_mean": graphon.get("mean"),
                "graphon_stderr": graphon.get("stderr"),
                "uniform_mean": uniform.get("mean"),
                "uniform_stderr": uniform.get("stderr"),
                "mean_gap_graphon_minus_uniform": (
                    graphon.get("mean") - uniform.get("mean")
                    if graphon.get("mean") is not None and uniform.get("mean") is not None
                    else None
                ),
            }
        )
    if not rows:
        return
    path = os.path.join(out_dir, f"{prefix}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(graphon_results, uniform_results, out_dir, prefix):
    common_ks = sorted(set(graphon_results) & set(uniform_results))
    if not common_ks:
        print("No overlapping kappa values between graphon and uniform results.")
        return

    graphon_means = [graphon_results[k].get("mean") for k in common_ks]
    uniform_means = [uniform_results[k].get("mean") for k in common_ks]
    graphon_errs = [graphon_results[k].get("stderr", 0.0) for k in common_ks]
    uniform_errs = [uniform_results[k].get("stderr", 0.0) for k in common_ks]
    if any(v is None for v in graphon_means + uniform_means):
        print("Skipping plot because some overlapping kappa values are missing means.")
        return

    plt.figure(figsize=(8.5, 5.5))
    plt.errorbar(
        common_ks,
        graphon_means,
        yerr=graphon_errs,
        fmt="o-",
        capsize=4,
        linewidth=2,
        label="Graphon-weighted sampling",
    )
    plt.errorbar(
        common_ks,
        uniform_means,
        yerr=uniform_errs,
        fmt="s--",
        capsize=4,
        linewidth=2,
        label="Uniform sampling",
    )
    plt.xlabel(r"$\kappa$ (subsample size)", fontsize=16)
    plt.ylabel("Discounted return", fontsize=16)
    plt.tick_params(axis="both", labelsize=13)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=13)
    plt.tight_layout()
    base = os.path.join(out_dir, prefix)
    plt.savefig(base + ".png", dpi=200)
    plt.savefig(base + ".pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphon_dir", default="results_robotics_n1000")
    parser.add_argument("--uniform_dir", default="results_robotics_n1000_uniform")
    parser.add_argument("--out_dir", default="Figures")
    parser.add_argument("--prefix", default="uniform_sampling_comparison")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    graphon_results = load_kappa_results(args.graphon_dir)
    uniform_results = load_kappa_results(args.uniform_dir)
    write_comparison_csv(graphon_results, uniform_results, args.out_dir, args.prefix)
    plot_comparison(graphon_results, uniform_results, args.out_dir, args.prefix)


if __name__ == "__main__":
    main()
