"""Standardized, PCA-projected OptDigits samples for one classification task."""
struct OptDigitsData <: ExperimentData
    x_train::Matrix{Float64}
    y_train::Vector{Int}
    x_test::Matrix{Float64}
    y_test::Vector{Int}
    classes::Int
    projection::PCA{Float64}
end

"""Recorded output from one single-chain OptDigits run."""
struct OptDigitsResult <: ChainResult
    data::OptDigitsData
    traces::Vector{Any}
    scores::Vector{Float64}
    widths::Vector{Int}
    across_acceptance::Vector{Int}
    within_acceptance::Vector{Int}
    settings::NamedTuple
end

"""Directory holding `optdigits_x.jld` and `optdigits_y.jld`."""
optdigits_directory() = get(
    ENV,
    "OPTDIGITS_DIR",
    normpath(joinpath(@__DIR__, "..", "..", "data", "optdigits")),
)

"""
    balanced_set(x, y, n, c, seed=0)

Draw `n` observations of each class `1:c` after a seeded shuffle, ordered by
class. Ported verbatim from `utils.jl`.
"""
function balanced_set(x::AbstractMatrix, y::AbstractVector, n::Int, c::Int, seed::Int=0)
    if seed != 0
        Random.seed!(seed)
    end

    shuffled_indices = shuffle(1:length(y))
    x = x[shuffled_indices, :]
    y = y[shuffled_indices]

    x_ordered = zeros(Float64, n * c, size(x, 2))
    y_ordered = zeros(Int, n * c)
    for k = 1:c
        labels = [i for i in 1:length(y) if y[i] == k]
        length(labels) >= n ||
            throw(ArgumentError("class $k has $(length(labels)) observations, fewer than $n"))
        x_ordered[k*n-(n-1):k*n, :] = x[labels, :][1:n, :]
        y_ordered[k*n-(n-1):k*n] = y[labels][1:n]
    end
    return x_ordered, y_ordered
end

"""
    load_optdigits(; classes, kwargs...)

Load, standardize, sample, and project the OptDigits arrays as the historical
`LoadData.jl` did. The `.jld` files are read with `deserialize`, and the
`ZScoreTransform` is fitted over `dims=2` of the raw `observations × pixels`
array, which standardizes each image across its own pixels.
"""
function load_optdigits(;
    classes::Int,
    samples_per_class::Int=50,
    test_samples_per_class::Int=200,
    training_seed::Int=1,
    test_seed::Int=300,
    pca_dimensions::Int=20,
    directory::AbstractString=optdigits_directory(),
)
    inputs_path = joinpath(directory, "optdigits_x.jld")
    labels_path = joinpath(directory, "optdigits_y.jld")
    for path in (inputs_path, labels_path)
        isfile(path) || throw(ArgumentError(
            "missing $path; set OPTDIGITS_DIR or pass `directory`",
        ))
    end

    x_total = deserialize(inputs_path)
    y_total = deserialize(labels_path)

    standardizer = fit(ZScoreTransform, x_total, dims=2)
    StatsBase.transform!(standardizer, x_total)

    x_train, y_train = balanced_set(x_total, y_total, samples_per_class, classes, training_seed)
    x_test, y_test = balanced_set(x_total, y_total, test_samples_per_class, classes, test_seed)

    projection = fit(PCA, transpose(x_train), maxoutdim=pca_dimensions)
    projected_train = MultivariateStats.transform(projection, transpose(x_train))
    projected_test = MultivariateStats.transform(projection, transpose(x_test))

    return OptDigitsData(
        Matrix(transpose(projected_train)),
        y_train,
        Matrix(transpose(projected_test)),
        y_test,
        classes,
        projection,
    )
end

"""Install one OptDigits dataset and its sampler settings."""
function prepare_optdigits!(
    data::OptDigitsData;
    maximum_width::Int=64,
    softmax_scale::Float64=0.1,
    target_acceptance::Float64=0.65,
    nuts_samples::Int=3,
    nuts_adaptation::Int=3,
    divergence_threshold::Real=10,
)
    install_data!(
        data.x_train,
        data.y_train;
        classes=data.classes,
        maximum_width=maximum_width,
        scale=softmax_scale,
        target_acceptance=target_acceptance,
        nuts_samples=nuts_samples,
        nuts_adaptation=nuts_adaptation,
        divergence_threshold=divergence_threshold,
    )
    return data
end

"""Run one OptDigits chain, retaining the best of `candidates` initial traces."""
function run_optdigits(;
    classes::Int,
    iterations::Int=1,
    samples_per_class::Int=50,
    test_samples_per_class::Int=200,
    training_seed::Int=1,
    test_seed::Int=300,
    pca_dimensions::Int=20,
    seed::Int=1,
    initial_width::Int=4,
    candidates::Int=1000,
    maximum_width::Int=64,
    softmax_scale::Float64=0.1,
    target_acceptance::Float64=0.65,
    nuts_samples::Int=3,
    nuts_adaptation::Int=3,
    divergence_threshold::Real=10,
    chain::Int=1,
    directory::AbstractString=optdigits_directory(),
)
    data = load_optdigits(
        classes=classes,
        samples_per_class=samples_per_class,
        test_samples_per_class=test_samples_per_class,
        training_seed=training_seed,
        test_seed=test_seed,
        pca_dimensions=pca_dimensions,
        directory=directory,
    )
    prepare_optdigits!(
        data;
        maximum_width=maximum_width,
        softmax_scale=softmax_scale,
        target_acceptance=target_acceptance,
        nuts_samples=nuts_samples,
        nuts_adaptation=nuts_adaptation,
        divergence_threshold=divergence_threshold,
    )
    Random.seed!(seed)
    trace = best_initial_trace(initial_width, candidates=candidates)
    traces, scores, widths, across_acceptance, within_acceptance =
        run_chain(trace, iterations, chain=chain)

    settings = (;
        classes,
        iterations,
        samples_per_class,
        test_samples_per_class,
        training_seed,
        test_seed,
        pca_dimensions,
        seed,
        initial_width,
        candidates,
        maximum_width,
        softmax_scale,
        target_acceptance,
        nuts_samples,
        nuts_adaptation,
        divergence_threshold,
        chain,
    )
    return OptDigitsResult(
        data,
        traces,
        scores,
        widths,
        across_acceptance,
        within_acceptance,
        settings,
    )
end
