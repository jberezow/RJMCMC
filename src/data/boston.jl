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

"""Directory holding the 506-row `boston.jld` dataset."""
boston_directory() = get(
    ENV,
    "BOSTON_HOUSING_DIR",
    normpath(joinpath(@__DIR__, "..", "..", "data", "boston")),
)

"""
    load_boston(; seed=23, directory=boston_directory())

Load the Boston Housing data. The stored rows are shuffled, then all 13
predictors and the response are z-scored using the complete dataset before it is
split into equal halves.
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

"""Recorded output from one single-chain Boston run."""
struct BostonResult <: ChainResult
    data::BostonData
    traces::Vector{Any}
    scores::Vector{Float64}
    depths::Vector{Int}
    across_acceptance::Vector{Int}
    within_acceptance::Vector{Int}
    settings::NamedTuple
end

"""Predicted responses for observations stored one per column."""
boston_predictions(trace, inputs::AbstractMatrix) = vec(DepthBNN.G(inputs, trace))

"""
    scaled_mse(predictions, targets, standardizer)

Both vectors are returned to the original housing scale, then the square root
of the summed squared error is divided by the number of observations.
"""
function scaled_mse(predictions, targets, standardizer)
    p = StatsBase.reconstruct(standardizer, collect(float(predictions)))
    t = StatsBase.reconstruct(standardizer, collect(float(targets)))
    return sqrt(sum((p .- t) .^ 2)) / length(t)
end

"""Generate a trace for `data` constrained to `depth` hidden layers."""
function initial_boston_trace(
    data::BostonData,
    depth::Int;
    width::Int,
    maximum_depth::Int,
    likelihood_variance::Float64,
)
    1 <= depth <= maximum_depth ||
        throw(ArgumentError("depth must lie between 1 and maximum_depth"))
    observations = choicemap()
    observations[:y] = data.y_train
    observations[:l] = depth
    (trace,) = generate(
        DepthBNN.interpolator,
        (data.x_train, width, maximum_depth, likelihood_variance),
        observations,
    )
    return trace
end

"""
    best_initial_boston_trace(data, depth; candidates=1000, kwargs...)

Retain the candidate with the lowest [`scaled_mse`](@ref) against the training
responses, out of `candidates` draws plus one that is drawn but not compared.
"""
function best_initial_boston_trace(
    data::BostonData,
    depth::Int;
    candidates::Int=1000,
    width::Int,
    maximum_depth::Int,
    likelihood_variance::Float64,
)
    candidates >= 1 || throw(ArgumentError("candidates must be positive"))
    draw() = initial_boston_trace(data, depth; width=width,
        maximum_depth=maximum_depth, likelihood_variance=likelihood_variance)
    error_of(trace) = scaled_mse(
        boston_predictions(trace, data.x_train), data.y_train, data.response_standardizer)

    best = draw()
    best_error = error_of(best)
    draw()  # drawn but not compared
    for _ = 1:candidates
        trace = draw()
        candidate_error = error_of(trace)
        if candidate_error < best_error
            best_error = candidate_error
            best = trace
        end
    end
    return best
end

"""Advance one Boston chain, recording each state."""
function run_depth_chain(trace, iterations::Int, settings::DepthRJNUTS.DepthSettings;
                         chain::Int=1)
    traces = Any[]
    scores = Float64[]
    depths = Int[]
    across_acceptance = Int[]
    within_acceptance = Int[]

    for iteration in 1:iterations
        trace, accepted_across, accepted_within =
            DepthRJNUTS.RJNUTS_parallel(trace, settings, chain, iteration)
        push!(traces, trace)
        push!(scores, get_score(trace))
        push!(depths, trace[:l])
        push!(across_acceptance, accepted_across)
        push!(within_acceptance, accepted_within)
    end

    return traces, scores, depths, across_acceptance, within_acceptance
end

"""
    run_boston(; kwargs...)

Run one Boston chain. Chain `i` starts at depth `((i - 1) % maximum_depth) + 1`,
spreading 16 chains evenly over the available depths.
"""
function run_boston(;
    iterations::Int=1,
    chain::Int=1,
    width::Int=2,
    maximum_depth::Int=8,
    likelihood_variance::Float64=1.0,
    data_seed::Int=23,
    seed::Int=1,
    candidates::Int=1000,
    initial_depth::Union{Nothing,Int}=nothing,
    target_acceptance::Float64=0.65,
    nuts_samples::Int=1,
    nuts_adaptation::Int=1,
    divergence_threshold::Real=1000,
    within_dimension_update::Symbol=:mixed,
    directory::AbstractString=boston_directory(),
)
    data = load_boston(seed=data_seed, directory=directory)
    # NUTS reads this threshold from module state.
    global Δ_max = divergence_threshold
    settings = DepthRJNUTS.DepthSettings(
        target_acceptance, nuts_samples, nuts_adaptation, within_dimension_update)

    depth = initial_depth === nothing ? ((chain - 1) % maximum_depth) + 1 : initial_depth
    Random.seed!(seed)
    trace = best_initial_boston_trace(data, depth; candidates=candidates,
        width=width, maximum_depth=maximum_depth, likelihood_variance=likelihood_variance)
    traces, scores, depths, across_acceptance, within_acceptance =
        run_depth_chain(trace, iterations, settings, chain=chain)

    recorded = (;
        iterations, chain, width, maximum_depth, likelihood_variance, data_seed,
        seed, candidates, initial_depth=depth, target_acceptance, nuts_samples,
        nuts_adaptation, divergence_threshold, within_dimension_update,
    )
    return BostonResult(data, traces, scores, depths,
                        across_acceptance, within_acceptance, recorded)
end

"""Root mean squared error of one trace's predictions, on the original scale."""
function boston_rmse(trace, inputs::AbstractMatrix, targets, standardizer)
    p = StatsBase.reconstruct(standardizer, collect(float(boston_predictions(trace, inputs))))
    t = StatsBase.reconstruct(standardizer, collect(float(targets)))
    return sqrt(sum((p .- t) .^ 2) / length(t))
end
