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
| `RJMCMC` | `bc0f872` | This repository, and its pre-consolidation history |
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

### Evaluation reproduction

The September 2026 evaluation of the archived depth chains recovered the
Chapter 7 table `tab:results_bh` figures:

| Thesis experiment | Published RMSE | Recomputed | Published layer mode | Recomputed |
|---|---:|---:|---:|---:|
| 2A | 0.236 | 0.2357 | 2 | 2 |
| 2B | 0.245 | 0.2464 | 4 | 4 |
| 4A | 0.223 | 0.2233 | 3 | 3 |
| 4B | 0.215 | 0.2155 | 3 | 3 |

The metric is the historical `mse_scaled`, which the thesis reports as RMSE:
predictions and targets are returned to the original housing scale, and the
square root of the summed squared error is divided by the number of
observations. `scaled_mse` reproduces it.

Marginalization averages the predicted response across every stored iteration of
all 16 chains and then applies the metric once. The analysis notebooks used no
burn-in for these figures, and reproducing the published values requires the
same: discarding the first 500 iterations moves 2A to 0.2302 and 4A to 0.2174.

The training inputs stored in every chain's trace arguments match those produced
by `load_boston`, which verifies the migrated loader, `DepthBNN.G`, and
`scaled_mse` together against the historical runs. As with the width
experiments, this verifies the data pipeline and the evaluation of archived
samples; no fresh full-length MCMC reproduction is claimed.

Two details of the published tables:

- The `tab:rmse_arch` best-chain column is not the minimum over all 16 chains.
  For 2A the published 0.235 is reproduced exactly by chain 3, while chain 11
  scores 0.231, so at least one chain was excluded from that column. The
  notebooks carry a `good_traces` list that selects chains.
- The 2B directory holds 16,006 stored traces where the other three hold 16,016,
  so at least one 2B chain stopped slightly short of 1,000 iterations. Chains
  were written every five iterations.

The four result directories `Data/BostonTwo`, `Data/BostonTwob`,
`Data/BostonFour`, and `Data/BostonFourb` hold the 64 chain files and the
acceptance records. Preserve them with the source snapshots when archiving.
Historical traces are read the same way as the width traces described above,
binding the generated function to `BNN.interpolator.julia_function` rather than
`BNN.classifier.julia_function`.

### Known discrepancy: 2B likelihood variance

The thesis specifies a likelihood variance of `0.8` for experiment 2B and reports
coherent 2B results, but the surviving `docker-parallel2b/BNN.jl` is identical to
2A and contains `1.0`. The 4B snapshot independently shows the intended `0.8`
change. `experiments/boston/2-node-b.toml` encodes `0.8` as the intended and
executed value and records the discrepancy in a comment. The exact source used
for the 2B run has not been recovered.

## Removed material

The following were removed from the active branch once the migration they
informed was complete. All remain in Git history and under each repository's
`pre-cleanup` tag.

- `docker_bh/` — a February 2021 Boston prototype built on the earlier
  `NUTS_CS` sampler lineage. Its RJNUTS driver sampled the weight and bias
  hyperparameters directly (`propose_hyperparameters`, `nuts_hyperparameters`),
  an approach the final experiments abandoned; the corresponding hyperprior code
  is commented out rather than deleted in the surviving models. It predates the
  finalized strategy and is not the provenance anchor for anything.
- `Run1.jld` — early development output. It no longer deserializes: the stored
  Gen version cannot be resolved against the pinned environment.
- `boston2.jld` — the 490-row dataset variant belonging to that prototype,
  produced by removing the 16 observations whose response is capped at 50. The
  four final experiments used the complete 506-row file instead.
- `notebooks/` — 23 thesis-era working notebooks. None carried any narrative
  text, and those closest to being demonstrators depended on helper files in
  `archive/legacy-julia/`. The notebooks that produced the published tables are
  the ones in `BostonHousing` and `OptDigits` named above, not these.
- `archive/early-exploration/`, `archive/notebooks/autosaves/`,
  `archive/admin-notebooks/`, and the `experiments/`, `proposals/` and
  `utilities/` subdirectories of `archive/legacy-julia/` — internship-era
  exploration, editor checkpoints, and development code superseded by the
  migrated implementations.

`archive/legacy-julia/samplers/` is deliberately retained. `NUTS_CS.jl`,
`oldNUTS.jl` and `hmc_mod.jl` document the development lineage behind the
preserved NUTS kernel and are evidence for the attribution and licensing
question below. `archive/thesis-source-backups/` is retained pending the thesis
source work. Both should go once those are settled.

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
