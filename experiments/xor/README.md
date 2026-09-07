# XOR experiment

A four-mode synthetic classification task, and the smallest demonstration of
variable-width inference: it needs no external dataset. It uses the same
Bayesian neural network, node birth/death proposals, RJNUTS driver, and NUTS
kernel as the OptDigits experiments.

The runtime target is Julia 1.6.1. From the repository root:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. scripts/run_xor.jl \
  --config=experiments/xor/low-noise.toml \
  --iterations=1 \
  --samples-per-mode=2
```

Remove the two command-line overrides to use the full sample and iteration
counts. The runner executes a single chain; the `chains = 16` setting in the
configuration files records the original 16-thread HPC runs.

The runner saves its trace and acceptance information beneath `results/xor/`.
Generate the analysis used by the thesis-style workflow with:

```sh
julia --project=. scripts/analyze_xor.jl \
  --input=results/xor/xor-low-noise-seed-1.jls \
  --burn-in=0
```

The analysis produces a text summary, a per-iteration CSV file, and SVG figures
for the log posterior, test accuracy, hidden-width histogram, and
posterior-averaged classification surface. Generated results are ignored by
Git.

Two configurations:

- `low-noise.toml`: diagonal mode covariance `0.015`
- `noisy.toml`: diagonal mode covariance `0.1`

The implementation follows these sources closely:

- `OptDigits/dockerxor/BNN.jl`
- `OptDigits/dockerxor/NUTS.jl`
- `OptDigits/dockerxor/RJNUTS.jl`
- `OptDigits/dockerxor/proposals.jl`
