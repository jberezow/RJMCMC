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

The `optdigits_x.jld` and `optdigits_y.jld` arrays are not distributed with the
repository. Place them in `data/optdigits/`, or point `OPTDIGITS_DIR` or
`--data=` at the directory holding them.

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

## Sources and reproducibility

The settings were extracted from `main.jl`, `BNN.jl`, `LoadData.jl`, and
`program.sh` in the four directories above at OptDigits commit
[`ff0b6e3`](https://github.com/jberezow/OptDigits/tree/ff0b6e3f8ddb15326bd8ddd8f65aecf4b264954e).
The sampling helper is `balanced_set` in `utils.jl`. No datasets or output
traces are included here.

The thesis reference is `Thesis/Chapters/Chapter6.tex`, sections labelled
`sec:Opt_A` and `sec:Opt_B`, and the preceding classification/data description.
The local source is under `LaTeX_Backups/`; it is not required to run the code.
The five-class subsection calls its variants “10a” and “10b”, while the later
results paragraph identifies them as 5A and 5B. The ten-class subsection repeats
the five-class sample totals; its introductory description and the loader
agree on 500 training and 2,000 test samples for ten classes.

`dockeropt5c/main.jl` calls `find_best_trace(xt, y, 100, obs)` where the other
three pass 1,000, but the function ignores its count argument and always loops
1,000 times in all four copies. Every reported run therefore used 1,000
initialization candidates, matching the thesis description, and all four
configurations record 1,000.

`load_optdigits` reproduces the historical loader bit for bit at both class
counts. Three of its behaviors are preserved deliberately:

- The `.jld` inputs are read with Julia `deserialize`, despite their extension.
- `ZScoreTransform` is fitted with `dims=2` over the raw `5620 × 64`
  observations-by-pixels array, which standardizes each image across its own 64
  pixels rather than each pixel across the dataset. PCA is then fitted to the
  transposed training sample and applied to both samples.
- Training and test samples are drawn by separate `balanced_set` calls over the
  same array, using seeds 1 and 300. The test-seed comment says 2, but the call
  uses 300.

The source arrays were identified as the UCI OptDigits training and test
partitions concatenated in that order, using their per-class counts; dataset
redistribution licensing remains unresolved. The historical data pipeline and
archived-trace evaluation reproduce the Chapter 7 Top-1 accuracies within
0.1 percentage points for all four experiments; convergence of fresh full-length
MCMC runs to those figures has not been verified (see
[provenance](../../docs/provenance.md)).

The historical training and evaluation samples overlap, so these reproduced
accuracies should not be interpreted as results on a strictly held-out test set.

`MultivariateStats` is pinned to 0.8.0 because `transform` was renamed to
`predict` in 0.9 and removed in 0.10.
