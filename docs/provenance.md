# OptDigits data and archived results

The source snapshots are in `OptDigits` at commit
[`ff0b6e3`](https://github.com/jberezow/OptDigits/tree/ff0b6e3f8ddb15326bd8ddd8f65aecf4b264954e).
The 5,620-row input arrays were identified as UCI `optdigits.tra` followed by
`optdigits.tes`: the per-class counts match each partition separately, with
3,823 rows in the first partition and 1,797 in the second. Redistribution
licensing remains unresolved.

## Evaluation reproduction

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
Carlo draws. Reconstructed PCA arrays agree with saved training/evaluation
arrays to approximately `5.5e-14`. This verifies the data pipeline and evaluation
of archived samples; no fresh full-length MCMC reproduction is claimed.
The historical training and evaluation draws overlap, so this is not a strictly
held-out evaluation.

## Reading historical traces

Use Julia 1.6.1 and the pinned project environment. Add the historical
`OptDigits/dockeropt5b` directory to `LOAD_PATH` and load `BNN` with `using BNN`
as a top-level package module; including it as `Main.BNN` is insufficient.
The `.jld` files use Julia `Serialization.deserialize`, not JLD storage.

Gen's generated classifier function names vary between sessions. If
`deserialize` raises `UndefVarError` for that generated function type, take
`String(e.var)[2:end]` (remove exactly one leading `#`), define the corresponding
function in `BNN`, and bind its varargs method to `BNN.classifier.julia_function`;
then retry deserialization. The recovery assessment used this binding only to
load traces, never to execute their serialized model functions. Read the saved
choices and evaluate the forward pass with the experiment's explicit class
count and softmax scale; this is an evaluation recipe, not a sampler-resumption
procedure.

The four result directories `Data/Opt5b`, `Data/Opt5c`, `Data/Opt10b`, and
`Data/Opt10c` contain the 64 chain files and saved PCA arrays needed for this
comparison. Preserve them together with the raw input arrays and historical
source snapshots when archiving the auxiliary repository.
