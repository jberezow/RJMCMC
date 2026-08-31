# Trans-dimensional inference over Bayesian neural networks

This repository contains the Julia implementation developed for Jonathan
Berezowski's MSc thesis, *Trans-dimensional Inference over Bayesian Neural
Networks*, completed at UiT The Arctic University of Norway in 2021.

The project uses reversible-jump Markov chain Monte Carlo (RJMCMC) to perform
Bayesian inference jointly over neural-network parameters and architecture. A
custom No-U-Turn Sampler (NUTS) updates weights and biases within a fixed
architecture, while reversible-jump proposals move between architectures of
different dimensions. The resulting sampler is referred to in the thesis as
RJNUTS.

The research considers two complementary architecture variables:

- **Network width:** neuron birth and death moves vary the number of hidden
  nodes in a single-layer classifier. This is used for the XOR and OptDigits
  experiments.
- **Network depth:** layer birth and death moves vary the number of hidden
  layers in a regression network. This is used for the Boston Housing
  experiments.

Rather than selecting one network, predictions can be averaged over the
sampled posterior distribution of network parameters and architectures.

## Run the XOR experiment

The XOR classifier is the current self-contained example. It demonstrates
variable-width inference without requiring an external dataset.

The historical environment uses Linux and Julia 1.6.1. From the repository
root, instantiate the pinned dependencies and run the tests:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

Run a representative low-noise XOR chain:

```sh
julia --project=. scripts/run_xor.jl \
  --config=experiments/xor/low-noise.toml \
  --iterations=100 \
  --seed=1
```

The result is saved to `results/xor/xor-low-noise-seed-1.jls`. Generate the
summary, trace data, and thesis-style figures with:

```sh
julia --project=. scripts/analyze_xor.jl \
  --input=results/xor/xor-low-noise-seed-1.jls \
  --output-dir=results/xor/xor-low-noise-seed-1-analysis \
  --burn-in=20
```

The analysis includes the log-posterior trace, classification accuracy,
sampled-width histogram, and posterior-averaged decision surface. To use the
full historical setting of 1,000 iterations, omit the `--iterations` override.
The original thesis runs used 16 independent chains; the current command runs
one chain.

The noisy variant can be selected with
`experiments/xor/noisy.toml`. See
[`experiments/xor/README.md`](experiments/xor/README.md) for additional details.

## Repository layout

```text
src/          Models, inference algorithms, and reversible-jump proposals
scripts/      Experiment runners and analysis scripts
experiments/  Experiment configurations and usage notes
test/         Julia tests and short sampler checks
notebooks/    Thesis-era exploratory and analysis notebooks
archive/      Historical implementations and research artifacts
```

Generated experiment results are written beneath `results/` and are not
tracked by Git.

## Thesis

Jonathan Berezowski. *Trans-dimensional Inference over Bayesian Neural
Networks*. MSc thesis, UiT The Arctic University of Norway, 2021.

The implementation was built with [Gen](https://www.gen.dev/) and preserves a
custom NUTS implementation derived from earlier Julia/Turing code. The sampling
method builds on Green's reversible-jump MCMC and the Hoffman--Gelman NUTS
algorithm; source-level attribution is retained alongside the implementations.

Repository cleanup and consolidation of the original thesis materials is
ongoing.
