"""
Usage: uv run main.py
Output: plots.svg
Inputs are assumed valid. NumPy handles the calculations; Matplotlib draws the plot.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")


def exponential_naive(observed_times):
    """Estimate the rate as n / sum(y), ignoring censoring."""
    return 1 / np.mean(observed_times)


def exponential_mle(observed_times, event_observed):
    """Estimate the rate as r / sum(y), with NaN when no events are observed."""
    event_count = np.count_nonzero(event_observed)
    return event_count / np.sum(observed_times)


# 1. Choose the study settings.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sample-size", type=int, default=1000)
parser.add_argument("--true-rate", type=float, default=0.05)
parser.add_argument("--repetitions", type=int, default=2000)
parser.add_argument("--censoring", type=float, nargs="+", default=[0.0, 0.1, 0.5])
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

n = args.sample_size
true_rate = args.true_rate
repetitions = args.repetitions
rng = np.random.default_rng(args.seed)
results = []


# 2. Run the experiment separately for each target censoring proportion.
for target_censoring in args.censoring:
    # Calibrate C ~ Uniform(0, c_max) before generating any replicates.
    # P(T > C) = (1 - exp(-lambda*c_max)) / (lambda*c_max).
    # With x = lambda*c_max, the root lies between 0 and 1/p_c.
    if target_censoring == 0:
        c_max = np.inf  # No finite uniform bound gives exactly 0% censoring.
    else:
        left, right = 0.0, 1.0 / target_censoring
        for _ in range(80):
            midpoint = (left + right) / 2
            probability = -np.expm1(-midpoint) / midpoint
            if probability > target_censoring:
                left = midpoint
            else:
                right = midpoint
        c_max = ((left + right) / 2) / true_rate

    naive_estimates = np.empty(repetitions)
    mle_estimates = np.empty(repetitions)
    event_counts = np.empty(repetitions, dtype=np.int64)

    # 3. Generate each sample and compute both estimates.
    for replicate in range(repetitions):
        event_times = rng.exponential(scale=1 / true_rate, size=n)
        if np.isposinf(c_max):
            censoring_times = np.full(n, np.inf)
        else:
            censoring_times = rng.uniform(low=0.0, high=c_max, size=n)

        observed_times = np.minimum(event_times, censoring_times)
        event_observed = event_times <= censoring_times
        event_count = np.count_nonzero(event_observed)

        naive_estimates[replicate] = exponential_naive(observed_times)
        mle_estimates[replicate] = exponential_mle(observed_times, event_observed)
        event_counts[replicate] = event_count

    # 4. Summarize the valid replicates; variance uses the population divisor.
    summaries = {}
    for name, estimates in (("naive", naive_estimates), ("mle", mle_estimates)):
        valid = estimates[~np.isnan(estimates)]
        if valid.size == 0:
            summaries[name] = dict(bias=np.nan, variance=np.nan, mse=np.nan, valid_replicates=0)
        else:
            summaries[name] = dict(
                bias=float(np.mean(valid) - true_rate),
                variance=float(np.var(valid, ddof=0)),
                mse=float(np.mean((valid - true_rate) ** 2)),
                valid_replicates=int(valid.size),
            )

    results.append({
        "sample_size": n,
        "true_rate": true_rate,
        "target_censoring": target_censoring,
        "c_max": float(c_max),
        "seed": args.seed,
        "naive_estimates": naive_estimates,
        "mle_estimates": mle_estimates,
        "event_counts": event_counts,
        "observed_censoring": float(1 - np.mean(event_counts) / n),
        "zero_event_proportion": float(np.mean(event_counts == 0)),
        "naive_summary": summaries["naive"],
        "mle_summary": summaries["mle"],
    })


# 5. Plot bias, variance, and MSE against the target censoring proportion.
plot_results = sorted(results, key=lambda result: result["target_censoring"])
targets = [100 * result["target_censoring"] for result in plot_results]
series = (
    ("Naive", [result["naive_summary"] for result in plot_results], "#d55e00", "s-"),
    ("MLE", [result["mle_summary"] for result in plot_results], "#0072b2", "o--"),
)

plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
for ax, metric, units in zip(axes, ("bias", "variance", "mse"), ("time⁻¹", "time⁻²", "time⁻²")):
    title = "MSE (log scale)" if metric == "mse" else metric.capitalize()
    label = "MSE" if metric == "mse" else metric.capitalize()
    for name, summary, color, style in series:
        values = [item[metric] for item in summary]
        ax.plot(targets, values, style, color=color, label=name, linewidth=1.5)
    ax.set(title=f"Empirical {title}", xlabel="Target censoring (%)", ylabel=f"{label} ({units})")
    ax.set_xticks(targets)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False)
axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
axes[2].set_yscale("log")
fig.suptitle(f"Uniform censoring · n={n} · λ={true_rate:g} · "
             f"R={repetitions} per scenario · seed={args.seed}")
fig.savefig("plots.svg", format="svg")
plt.close(fig)
print(f"Results saved at plots.svg")
