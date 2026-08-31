# XOR experiment

This is the first runnable slice of the consolidated thesis implementation. It
uses the variable-width Bayesian neural network, node birth/death proposals,
RJNUTS driver, and NUTS kernel preserved from the final `OptDigits` Docker
experiments.

The historical runtime target is Julia 1.6.1. From the repository root:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. scripts/run_xor.jl \
  --config=experiments/xor/low-noise.toml \
  --iterations=1 \
  --samples-per-mode=2
```

Remove the two command-line overrides to use the full historical sample count
and iteration count. The current runner intentionally executes one chain; the
`chains = 16` values in the configuration files record the original HPC runs
and are not yet consumed by the runner.

Two historical configurations are retained:

- `low-noise.toml`: diagonal mode covariance `0.015`
- `noisy.toml`: diagonal mode covariance `0.1`

The source files remain deliberately close to these historical inputs:

- `OptDigits/dockerxor/BNN.jl`
- `OptDigits/dockerxor/NUTS.jl`
- `OptDigits/dockerxor/RJNUTS.jl`
- `OptDigits/dockerxor/proposals.jl`
