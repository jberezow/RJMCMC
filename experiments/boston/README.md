# Boston Housing: variable-depth regression

The Boston model varies the number of hidden layers while holding their width
fixed. Hidden layers use ReLU activations; a linear output layer predicts the
response under a Gaussian likelihood. This differs from the variable-width
classifiers used for XOR and OptDigits.

The implementation is available as `RJBNN.DepthBNN` in
[`src/models/depth_bnn.jl`](../../src/models/depth_bnn.jl), with layer birth/death
moves in `RJBNN.LayerProposals`
([source](../../src/proposals/layer_birth_death.jl)). The data loader is
available as `RJBNN.load_boston`, and the depth sampler as `RJBNN.DepthRJNUTS`
([source](../../src/inference/depth_rjnuts.jl)).

## Running

```bash
julia --project=. scripts/run_boston.jl --config=experiments/boston/4-node-b.toml --chain=3
```

The runner advances one chain. `--chain=i` uses the initialization schedule of
historical chain `i`, starting at depth `((i - 1) % maximum_depth) + 1`;
`--iterations`, `--candidates`, `--initial-depth`, `--seed`, and `--output`
override the configuration for shorter runs. The original runners ran 16 such
chains across 16 threads.

Initialization keeps the candidate with the lowest scaled MSE against the
training responses. The width experiments rank candidates by posterior score
instead.

## Data preparation

The canonical stored dataset contains 506 rows and 14 columns: 13 predictors
and the median home-value response. The historical workflow shuffles the rows
with seed 23, fits separate z-score transforms to the predictors and response
using all 506 rows, and then splits the standardized data into 253 training and
253 test observations. Inputs are transposed to the
`features × observations` orientation used by the model.

```julia
using RJBNN
data = load_boston()
size(data.x_train) # (13, 253)
size(data.x_test)  # (13, 253)
```

The fitted transforms are retained in `data.feature_standardizer` and
`data.response_standardizer`. Set `BOSTON_HOUSING_DIR` or pass `directory` to
load the canonical `boston.jld` from another location.

An earlier prototype used a 490-row variant, produced by removing the 16
observations whose response is capped at 50. The four experiments use the
complete 506-row file, and the loader rejects anything of another size.

## Model settings

| Configuration | Hidden width | Maximum depth | Likelihood variance | Post-jump update |
|---|---:|---:|---:|---|
| [`2-node-a.toml`](2-node-a.toml) | 2 | 8 | 1.0 | Random layer-wise/all-parameter |
| [`2-node-b.toml`](2-node-b.toml) | 2 | 8 | 0.8 | All-parameter |
| [`4-node-a.toml`](4-node-a.toml) | 4 | 4 | 1.0 | Random layer-wise/all-parameter |
| [`4-node-b.toml`](4-node-b.toml) | 4 | 4 | 0.8 | Random layer-wise/all-parameter |

All configurations record 16 chains, 1,000 iterations, 1,000 initialization
candidates per chain, data seed 23, and a NUTS target acceptance of 0.65. The
2-node A/B pair differs in its RJNUTS scheduling; the 4-node B run additionally
uses two NUTS samples and two adaptation steps where the others use one of
each.

The model takes `interpolator(x, width, maximum_depth, likelihood_variance)`
as its Gen arguments, with defaults `(x, 2, 8, 1.0)`. Inputs have 13 feature
rows and one observation per column. Depth has a uniform prior on
`1:maximum_depth`; weight and bias priors retain unit covariance. The
likelihood variance is used directly as the diagonal observation covariance.

The `:τᵧ` Gamma draw is retained as a trace choice even though the likelihood
uses a fixed variance. Responses occupy one vector choice at `:y`;
`:l` records depth, and `(:k, i)`, `(:W, i)`, and `(:b, i)` describe each layer.

## Inspecting a layer move

This small example uses synthetic inputs, without requiring Boston data:

```julia
using RJBNN, Gen, Random
Random.seed!(1)
x = zeros(13, 3)
observations = choicemap((:l, 2), (:y, [0.2, -0.3, 1.0]))
trace, _ = generate(DepthBNN.interpolator, (x, 2, 8, 1.0), observations)
born, q_birth = LayerProposals.layer_birth(trace)
restored, q_death = LayerProposals.layer_death(born)
```

Birth appends a hidden layer immediately before the output; death removes the
last hidden layer. Existing parameters and responses are carried through Gen's
`update`. The returned `q` values preserve the historical forward/reverse
proposal-density terms. Bounds are read from the model arguments stored in the
trace. `DepthRJNUTS.layer_parameter` composes these moves into the full
across-dimension kernel.

## Historical source

The source is `BostonHousing` commit
[`3a2c64d`](https://github.com/jberezow/BostonHousing/tree/3a2c64d0b8c0f1483f018ab6db0fcb2365a91021).
All four `docker-parallel*` snapshots share the same `proposals.jl`. The model
comes from `docker-parallel2a/BNN.jl`, parameterized for the width and depth
changes in 4A and the covariance change in 4B. Trace addresses, priors,
activations, and proposal-density calculations are preserved. The 2B likelihood
variance of 0.8 follows the thesis; the archived 2B model file carries the 2A
value.

Running this implementation over the archived chains reproduces the Chapter 7
marginal RMSE and layer-count modes for all four experiments. The tests
additionally cover model scores and predictions against each snapshot, seeded
layer transitions and proposal weights, depth bounds, likelihood covariance,
gradients, and birth/death restoration.
