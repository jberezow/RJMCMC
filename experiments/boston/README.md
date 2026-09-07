# Boston Housing: variable-depth regression

The Boston model varies the number of hidden layers while holding their width
fixed. Hidden layers use ReLU activations; a linear output layer predicts the
response under a Gaussian likelihood. This differs from the variable-width
classifiers used for XOR and OptDigits.

The implementation is available as `RJBNN.DepthBNN` in
[`src/models/depth_bnn.jl`](../../src/models/depth_bnn.jl), with layer birth/death
moves in `RJBNN.LayerProposals`
([source](../../src/proposals/layer_birth_death.jl)). The Boston data loader and
full RJNUTS experiment runner are not yet included.

## Model settings

| Thesis experiment | Hidden width | Maximum depth | Likelihood variance |
|---|---:|---:|---:|
| 2A | 2 | 8 | 1.0 |
| 2B | 2 | 8 | 0.8 (thesis reconstruction) |
| 4A | 4 | 4 | 1.0 |
| 4B | 4 | 4 | 0.8 |

The model takes `interpolator(x, width, maximum_depth, likelihood_variance)`
as its Gen arguments, with defaults `(x, 2, 8, 1.0)`. Inputs have 13 feature
rows and one observation per column. Depth has a uniform prior on
`1:maximum_depth`; weight and bias priors retain unit covariance. The
likelihood variance is used directly as the diagonal observation covariance.

The historical `:τᵧ` Gamma draw is retained as a trace choice even though the
likelihood uses a fixed variance. Responses occupy one vector choice at `:y`;
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
proposal-density terms; these moves alone are not a complete RJNUTS sampler.
Bounds are read from the model arguments stored in the trace.

## Historical source

The source is `BostonHousing` commit
[`3a2c64d`](https://github.com/jberezow/BostonHousing/tree/3a2c64d0b8c0f1483f018ab6db0fcb2365a91021).
All four `docker-parallel*` snapshots share the same `proposals.jl`. The model
comes from `docker-parallel2a/BNN.jl`, parameterized for the width/depth changes
in 4A and the covariance change in 4B. Trace addresses, priors, activations,
and proposal-density calculations are preserved.

The surviving 2B model is identical to 2A and specifies variance 1.0. The thesis
specifies 0.8 for 2B; that setting is a reconstruction, not an exact copy of the
surviving source. It does not resolve which source file was used for that run.

Validation covers model scores and predictions against each surviving snapshot,
plus seeded layer transitions and proposal weights. The tests also check depth
bounds, likelihood covariance, gradients, and birth/death restoration. This
validates the extracted components, not reproduction of a full Boston run.
