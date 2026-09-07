# Provenance

Where the consolidated implementation came from, what has been verified against
the historical sources, and what remains open. Claims here are deliberately
narrow: reproduction of a stored artifact is not the same as reproduction of a
sampling run.

## Source repositories

The thesis-era code was consolidated from three repositories, each of which
carries a `pre-cleanup` tag preserving its original committed state:

| Repository | Revision inspected | Contributes |
|---|---|---|
| `RJMCMC` | `bc0f872` | This repository; earlier prototypes under `archive/` |
| `BostonHousing` | [`3a2c64d`](https://github.com/jberezow/BostonHousing/tree/3a2c64d0b8c0f1483f018ab6db0fcb2365a91021) | Variable-depth model, layer proposals, depth sampler, data |
| `OptDigits` | [`ff0b6e3`](https://github.com/jberezow/OptDigits/tree/ff0b6e3f8ddb15326bd8ddd8f65aecf4b264954e) | Variable-width model, node proposals, width sampler, XOR and OptDigits data |

## Shared NUTS kernel

`src/inference/nuts.jl` is preserved from the historical snapshots, where the
identical blob appears in every final Boston and OptDigits directory. Its header
attributes the implementation to Kai Xu (2016) and cites Algorithm 6 of Hoffman
and Gelman. That attribution is retained verbatim and should not be edited
without establishing the full upstream lineage first.

The kernel reads its divergence threshold from module state rather than an
argument. That is the one piece of ambient state the depth experiments still
depend on, and it is preserved rather than refactored.

## Variable-width experiments (XOR, OptDigits)

### Data

The 5,620-row OptDigits input arrays were identified as UCI `optdigits.tra`
followed by `optdigits.tes`: the per-class counts match each partition
separately, with 3,823 rows in the first and 1,797 in the second. Redistribution
licensing remains unresolved, so the arrays are not tracked in this repository.

`load_optdigits` reproduces the historical loader bit for bit at both class
counts. The training and evaluation samples are drawn by separate `balanced_set`
calls over the same array and overlap, so the reproduced accuracies below should
not be read as results on a strictly held-out test set.

### Evaluation reproduction

The September 2026 evaluation of archived traces recovered the Chapter 7 table
`tab:acc_unc` Top-1 accuracies within 0.1 percentage points:

| Thesis experiment | Published | Recomputed |
|---|---:|---:|
| 5A | 97.50% | 97.60% |
| 5B | 97.80% | 97.70% |
| 10A | 90.65% | 90.75% |
| 10B | 92.60% | 92.65% |

Evaluation averages class probabilities across chains 1–16 and stored trace
indices 500–1001 inclusive (8,032 traces per experiment), then takes the argmax.
The thesis notebooks instead estimated the posterior average with 1,000 Monte
Carlo draws. Reconstructed PCA arrays agree with the saved training and
evaluation arrays to approximately `5.5e-14`. This verifies the data pipeline
and the evaluation of archived samples; no fresh full-length MCMC reproduction
is claimed.

### Reading historical traces

Use Julia 1.6.1 and the pinned project environment. Add the historical
`OptDigits/dockeropt5b` directory to `LOAD_PATH` and load `BNN` with `using BNN`
as a top-level package module; including it as `Main.BNN` is insufficient.
The `.jld` files use Julia `Serialization.deserialize`, not JLD storage.

Gen's generated classifier function names vary between sessions. If
`deserialize` raises `UndefVarError` for that generated function type, take
`String(e.var)[2:end]` (remove exactly one leading `#`), define the corresponding
function in `BNN`, and bind its varargs method to `BNN.classifier.julia_function`;
then retry deserialization. Use this binding only to load traces, never to
execute their serialized model functions. Read the saved choices and evaluate the
forward pass with the experiment's explicit class count and softmax scale; this
is an evaluation recipe, not a sampler-resumption procedure.

The four result directories `Data/Opt5b`, `Data/Opt5c`, `Data/Opt10b`, and
`Data/Opt10c` contain the 64 chain files and saved PCA arrays needed for this
comparison. Preserve them together with the raw input arrays and historical
source snapshots when archiving the auxiliary repository.

## Variable-depth experiments (Boston Housing)

### Data

`data/boston/boston.jld` is a 506×14 matrix stored under the JLD key `boston`.
It is byte-identical to the copy in every final Boston snapshot. The 490×14
`boston2.jld` and `docker_bh/boston.jld` belong to the earlier prototype and are
not the thesis data.

`load_boston` reproduces the historical preparation exactly: rows are shuffled
under seed 23, all 13 predictors and the response are z-scored over the full
dataset, and the result is split into halves. Replacing the historical global
`Random.seed!(23)` with an explicit `MersenneTwister(23)` was verified to yield
an identical permutation.

The executed split is therefore **253/253**. The thesis describes a 256/256
split of 512 observations, which matches neither surviving data file; the loader
computed the split from the stored row count. The `Random.seed!(23)` comment
mentioning seed 3 is stale — seed 23 is what executed.

### Canonical sampler

`docker-parallel2a/RJNUTS.jl` is the anchor. The 4B blob is byte-identical to it,
4A differs only in whitespace, and 2B is the single substantive variant: 2A, 4A
and 4B choose between layer-wise and all-parameter NUTS with equal probability
for the within-dimension step, while 2B always takes the all-parameter step.
That difference is configuration (`within_dimension_update`), not separate code.

Chain `i` starts at depth `((i - 1) % maximum_depth) + 1`, which spread the
historical 16 chains evenly over the available depths.

Initialization ranks candidates by **lowest scaled MSE against the training
responses**, not by posterior score as the width experiments do. The historical
routine also generated one candidate whose error it computed but never compared;
`best_initial_boston_trace` preserves that draw so the random stream matches.
Unlike the width `find_best_trace`, the depth version genuinely honors its
candidate-count argument.

### Known discrepancy: 2B likelihood variance

The thesis specifies a likelihood variance of `0.8` for experiment 2B and reports
coherent 2B results, but the surviving `docker-parallel2b/BNN.jl` is identical to
2A and contains `1.0`. The 4B snapshot independently shows the intended `0.8`
change. `experiments/boston/2-node-b.toml` encodes `0.8` as the intended and
executed value and records the discrepancy in a comment. The exact source used
for the 2B run has not been recovered.

## Open questions

- Under what license may the thesis code, the Kai Xu-derived NUTS
  implementation, and the redistributed datasets be published?
- Was the OptDigits `ZScoreTransform` axis intended? It is fitted with `dims=2`
  over the observations-by-pixels array, standardizing each image across its own
  pixels rather than each pixel across the dataset. The behavior is preserved for
  parity and changes what the PCA basis represents.
- Can the 2B source actually used for execution be recovered from a notebook,
  container image, or job snapshot?
- Do fresh full-length runs of either family converge to the published figures?
  Nothing here establishes that.
