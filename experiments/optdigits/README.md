# OptDigits experiments

These configurations record the four variable-width classification experiments
from the thesis.

| Configuration | Historical directory | Classes | Maximum width | Softmax scale |
|---|---|---:|---:|---:|
| [5-class-a.toml](5-class-a.toml) | `dockeropt5b` | 5 | 64 | 0.1 |
| [5-class-b.toml](5-class-b.toml) | `dockeropt5c` | 5 | 64 | 0.5 |
| [10-class-a.toml](10-class-a.toml) | `dockeropt10b` | 10 | 128 | 0.1 |
| [10-class-b.toml](10-class-b.toml) | `dockeropt10c` | 10 | 128 | 0.5 |

The names follow the thesis A/B labels, which differ from the Docker b/c
suffixes. The softmax scale multiplies the logits before exponentiation. All
four models use one hidden layer, a uniform prior over widths from 1 to the
configured maximum, and fixed unit Gaussian weight and bias priors.

## Running

The `optdigits_x.jld` and `optdigits_y.jld` arrays ship in `data/optdigits/`.
Point `OPTDIGITS_DIR` or `--data=` at another directory to read them from
elsewhere.

```bash
julia --project=. scripts/run_optdigits.jl --config=experiments/optdigits/10-class-a.toml --chain=3
```

The runner advances one chain. `--chain=i` uses the initialization schedule of historical chain `i`,
starting at hidden width `initial_width_stride * i`; `--iterations`,
`--candidates`, `--initial-width`, `--seed`, and `--output` override the
configuration for shorter runs. The original runners ran 16 such chains across
16 threads.

Each experiment selects 50 training and 200 test samples per class and fits PCA
on the training sample with `maxoutdim = 20`. The five-class experiments use
digits 0–4; the historical arrays encode class labels starting at 1.
`nuts_samples`, `nuts_adaptation`, and `nuts_delta_max` record the original
globals `m`, `m2`, and `Δ_max`; the target acceptance is `acc_prob`.

## Sources

The settings come from `main.jl`, `BNN.jl`, `LoadData.jl`, and `program.sh` in
the four directories above, at OptDigits commit
[`ff0b6e3`](https://github.com/jberezow/OptDigits/tree/ff0b6e3f8ddb15326bd8ddd8f65aecf4b264954e).
The sampling helper is `balanced_set` in `utils.jl`. The corresponding thesis
sections are `sec:Opt_A` and `sec:Opt_B` in Chapter 6.

All four runs drew 1,000 candidate traces at initialization, and every
configuration here records that.

`load_optdigits` reproduces the original preparation exactly:

- The `.jld` inputs are read with Julia `deserialize`, despite their extension.
- `ZScoreTransform` is fitted with `dims=2` over the `5620 × 64`
  observations-by-pixels array, standardizing each image across its own pixels.
  PCA is then fitted to the transposed training sample and applied to both
  samples.
- Training and test samples are drawn by separate `balanced_set` calls over the
  same array, with seeds 1 and 300.

The input arrays are the UCI OptDigits training and test partitions
concatenated in that order. Running this pipeline over the archived chains
reproduces the Chapter 7 Top-1 accuracies to within 0.1 percentage points for
all four experiments.

`MultivariateStats` is pinned to 0.8.0 because `transform` was renamed to
`predict` in 0.9 and removed in 0.10.
