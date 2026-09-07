"""Standardized Boston Housing data in the orientation used by the depth model."""
struct BostonData{TX, TY} <: ExperimentData
    x_train::Matrix{Float64}
    y_train::Vector{Float64}
    x_test::Matrix{Float64}
    y_test::Vector{Float64}
    feature_standardizer::TX
    response_standardizer::TY
    row_order::Vector{Int}
end

"""Directory holding the canonical 506-row `boston.jld` dataset."""
boston_directory() = get(
    ENV,
    "BOSTON_HOUSING_DIR",
    normpath(joinpath(@__DIR__, "..", "..", "data", "boston")),
)

"""
    load_boston(; seed=23, directory=boston_directory())

Reproduce the data preparation shared by the four final Boston experiment
snapshots. The stored rows are shuffled, then all 13 predictors and the response
are z-scored using the complete dataset before it is split into equal halves.
Inputs are returned as `features × observations`, as expected by the depth
model. The fitted transforms are retained for converting predictions back to
the original housing-value scale.
"""
function load_boston(;
    seed::Int=23,
    directory::AbstractString=boston_directory(),
)
    path = joinpath(directory, "boston.jld")
    isfile(path) || throw(ArgumentError(
        "missing $path; set BOSTON_HOUSING_DIR or pass `directory`",
    ))

    stored = JLD.load(path, "boston")
    stored isa AbstractMatrix{<:Real} ||
        throw(ArgumentError("$path must contain a numeric matrix named `boston`"))
    size(stored) == (506, 14) || throw(ArgumentError(
        "$path has size $(size(stored)); expected 506 rows and 14 columns",
    ))

    row_order = shuffle(MersenneTwister(seed), 1:size(stored, 1))
    shuffled = Matrix{Float64}(stored[row_order, :])
    inputs = shuffled[:, 1:13]
    responses = shuffled[:, 14]

    feature_standardizer = fit(ZScoreTransform, inputs, dims=1)
    StatsBase.transform!(feature_standardizer, inputs)
    response_standardizer = fit(ZScoreTransform, responses, dims=1)
    StatsBase.transform!(response_standardizer, responses)

    split = div(size(stored, 1), 2)
    return BostonData(
        Matrix(transpose(inputs[1:split, :])),
        responses[1:split],
        Matrix(transpose(inputs[(split + 1):end, :])),
        responses[(split + 1):end],
        feature_standardizer,
        response_standardizer,
        row_order,
    )
end
