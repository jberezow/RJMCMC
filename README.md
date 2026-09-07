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

The XOR classifier is the self-contained example. It demonstrates
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

## Run the OptDigits experiment

The OptDigits experiments apply the same variable-width sampler to handwritten
digit classification. The `optdigits_x.jld` and `optdigits_y.jld` arrays ship in
`data/optdigits/`; set `OPTDIGITS_DIR` to read them from elsewhere.

```sh
julia --project=. scripts/run_optdigits.jl \
  --config=experiments/optdigits/5-class-a.toml \
  --chain=1
```

The runner advances one chain, starting at the hidden width the historical
chain index implies. The four thesis configurations and the data preparation are
described in
[`experiments/optdigits/README.md`](experiments/optdigits/README.md).

## Run the Boston Housing experiment

The Boston Housing experiments vary network *depth* rather than width, using
layer birth/death moves on a regression network. The dataset ships with the
repository, so no additional setup is needed.

```sh
julia --project=. scripts/run_boston.jl \
  --config=experiments/boston/2-node-a.toml \
  --chain=1
```

The four thesis configurations and the data preparation are described in
[`experiments/boston/README.md`](experiments/boston/README.md).

## Repository layout

```text
src/          Models, inference algorithms, and reversible-jump proposals
scripts/      Experiment runners and analysis scripts
experiments/  Experiment configurations and usage notes
data/         Input datasets
test/         Julia tests and short sampler checks
archive/      Earlier NUTS implementations
```

The Boston Housing and OptDigits datasets are tracked in `data/`. Generated
results are written beneath `results/`, which is not tracked by Git.

`Dockerfile` builds the Julia 1.6.1 reproduction environment and runs the test
suite by default:

```sh
docker build -t rjbnn . && docker run --rm rjbnn
```

## Thesis

Jonathan Berezowski. *Trans-dimensional Inference over Bayesian Neural
Networks*. MSc thesis, UiT The Arctic University of Norway, 2021. Included here
as `Thesis_Final.pdf`.

The implementation was built with [Gen](https://www.gen.dev/). The sampler
builds on Green's reversible-jump MCMC and the Hoffman--Gelman NUTS algorithm,
and the NUTS kernel derives from an earlier Julia implementation by Kai Xu,
attributed in its source file.

The archived `BostonHousing` and `OptDigits` repositories hold the per-chain
output of the original runs, which is what the tables in the thesis were
computed from.
