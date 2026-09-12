# Exponential rate estimation under right censoring

`main.py` contains the full simulation in one file, with separate functions for
the naive estimator and the MLE. NumPy handles the calculations and Matplotlib
draws the plot.

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed.
From the project directory, set up Python 3.14 and the virtual environment:

```sh
uv python install 3.14
uv venv --python 3.14
uv sync --locked
```

This creates `.venv` and installs NumPy, Matplotlib, and their dependencies
using the versions recorded in `uv.lock`.

`uv run` uses the project's environment automatically. To activate it manually
in bash or zsh instead:

```sh
source .venv/bin/activate
python main.py
deactivate
```

## Running the simulation

Run with the default settings:

```sh
uv run main.py
```

All arguments are optional:

- `--sample-size N`: number of individuals in each replicate; default `1000`.
- `--true-rate RATE`: exponential event rate, lambda; default `0.05`, corresponding
  to a mean event time of `20`.
- `--repetitions R`: number of Monte Carlo replicates **per censoring target**;
  default `2000`.
- `--censoring PC [PC ...]`: one or more target censoring proportions, expressed
  as fractions; default `0 0.1 0.5` means 0%, 10%, and 50%.
- `--seed SEED`: nonnegative integer seed for the random number generator;
  default `42`. The same settings and seed reproduce the numerical results.
- `-h`, `--help`: display the available arguments and exit.

For example:

```sh
uv run main.py --sample-size 500 --true-rate 0.1 --repetitions 1000 --censoring 0 0.2 0.5 --seed 7
```

Inputs are assumed valid: positive integer sample size and replicate count,
a finite positive rate, and a nonempty list of censoring targets in `[0, 1)`.

Independent censoring times follow `Uniform(0, c_max)`. Bisection solves
`(1 - exp(-lambda*c_max)) / (lambda*c_max) = p_c` for `c_max`.
Each bound stays fixed across replicates. Actual censoring fluctuates around
the target; at 0%, censoring is disabled.

## Output

Each run saves **`plots.svg` in the current working directory**, replacing that
file if it already exists, and prints `Results saved at plots.svg`.
Open the SVG in a browser or vector graphics editor.

The figure compares the naive estimator (orange) and MLE (blue) across the
target censoring proportions, with three panels:

- **Bias:** mean estimated rate minus the true rate.
- **Variance:** mean squared deviation from the mean estimate, using `ddof=0`.
- **MSE:** mean squared error relative to the true rate, shown on a logarithmic
  vertical axis.

The horizontal axis shows target censoring in percent. The title records the
sample size, true rate, replicate count, and seed. Summary calculations exclude
`NaN` estimates. The current MLE function returns `0` when no events are observed,
so those replicates remain included in the summaries.
