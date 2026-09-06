# OptDigits experiments

These configurations record the four variable-width classification experiments
from the thesis. OptDigits loading and runner support have not yet been migrated;
these files cannot be used with `scripts/run_xor.jl`. The current width model
still fixes two output classes, a maximum width of 16, and softmax scaling of 0.5.

| Configuration | Historical directory | Classes | Maximum width | Softmax scale |
|---|---|---:|---:|---:|
| [5-class-a.toml](5-class-a.toml) | `dockeropt5b` | 5 | 64 | 0.1 |
| [5-class-b.toml](5-class-b.toml) | `dockeropt5c` | 5 | 64 | 0.5 |
| [10-class-a.toml](10-class-a.toml) | `dockeropt10b` | 10 | 128 | 0.1 |
| [10-class-b.toml](10-class-b.toml) | `dockeropt10c` | 10 | 128 | 0.5 |

The names follow the thesis A/B labels, which differ from the Docker b/c
suffixes. The softmax scale multiplies the logits before exponentiation.
All four models use one hidden layer, a uniform prior over widths from 1 to
the configured maximum, and fixed unit Gaussian weight and bias priors.

Each experiment selects 50 training and 200 test samples per class and fits
PCA on the training sample with `maxoutdim = 20`. The five-class experiments
use digits 0–4; the historical arrays encode class labels starting at 1.

The original runners use 16 threads for 16 chains, each with 1,000 RJNUTS
iterations. Chain `i` starts at width `initial_width_stride * i`, retaining
the best of `initial_candidates` traces at that width. The surviving 5-class B
runner uses 100 candidates; the other three use 1,000. `nuts_samples`,
`nuts_adaptation`, and `nuts_delta_max` record the original globals `m`, `m2`,
and `Δ_max`; the target acceptance is `acc_prob`.

## Sources and migration notes

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

The thesis describes 1,000 initialization candidates for the five-class runs,
but `dockeropt5c/main.jl` passes 100 to `find_best_trace`. The 5-class B
configuration preserves that source value. Which value was used for the
reported run remains unresolved.

The loader has several details to preserve when extracting it:

- The `.jld` inputs are read with Julia `deserialize`, despite their extension.
- `ZScoreTransform` is fitted to the full loaded array with `dims=2` before
  sampling. PCA is then fitted to the transposed training sample and applied
  to both samples.
- Training and test samples are drawn independently from the same arrays,
  using seeds 1 and 300. The test-seed comment says 2, but the call uses 300.
  `balanced_set` shuffles and selects each class without excluding training
  rows from the test selection. The source therefore does not guarantee
  disjoint samples, despite the thesis prose referring to train/test datasets.

The next implementation step is to parameterize the shared width model and
extract the loader with these behaviors intact. Dataset provenance and the
saved arrays still need inspection before choosing a distribution method or
claiming reproduction of the reported results.
