"""Training and test data for the historical synthetic XOR experiment."""
struct XORData
    x_train::Matrix{Float64}
    y_train::Vector{Int}
    x_test::Matrix{Float64}
    y_test::Vector{Int}
end

"""Recorded output from one single-chain XOR run."""
struct XORResult
    data::XORData
    traces::Vector{Any}
    scores::Vector{Float64}
    widths::Vector{Int}
    across_acceptance::Vector{Int}
    within_acceptance::Vector{Int}
    settings::NamedTuple
end

function xor_modes(samples_per_mode::Int, bound::Float64, variance::Float64)
    means = (
        [-bound, -bound],
        [-bound, bound],
        [bound, bound],
        [bound, -bound],
    )
    covariance = [variance 0.0; 0.0 variance]
    samples = zeros(Float64, samples_per_mode * length(means), 2)
    classes = zeros(Int, samples_per_mode * length(means))

    for (index, mean) in enumerate(means)
        rows = ((index - 1) * samples_per_mode + 1):(index * samples_per_mode)
        samples[rows, :] = transpose(rand(MvNormal(mean, covariance), samples_per_mode))
        classes[rows] .= index
    end

    return samples, classes
end

function xor_labels(classes)
    return Int[(class + 1) % 2 + 1 for class in classes]
end

"""
    generate_xor_data(; samples_per_mode=50, variance=0.015, bound=0.5, seed=1)

Generate independent training and test sets using the four-mode construction
from `OptDigits/dockerxor`. The historical code called the parameter `σₐ`, but
passed it directly as the diagonal covariance of `MvNormal`; `variance` names
that implemented behavior explicitly.
"""
function generate_xor_data(;
    samples_per_mode::Int=50,
    variance::Float64=0.015,
    bound::Float64=0.5,
    seed::Int=1,
)
    Random.seed!(seed)
    x_train, train_classes = xor_modes(samples_per_mode, bound, variance)
    x_test, test_classes = xor_modes(samples_per_mode, bound, variance)
    return XORData(
        x_train,
        xor_labels(train_classes),
        x_test,
        xor_labels(test_classes),
    )
end

"""Install one XOR dataset and its historical sampler settings."""
function prepare_xor!(
    data::XORData;
    maximum_width::Int=16,
    target_acceptance::Float64=0.65,
    nuts_samples::Int=4,
    nuts_adaptation::Int=1,
    divergence_threshold::Real=1,
)
    global xt = transpose(data.x_train)
    global y = data.y_train
    global k_list = collect(1:maximum_width)
    global acc_prob = target_acceptance
    global m = nuts_samples
    global m2 = nuts_adaptation
    global Δ_max = divergence_threshold

    observations = choicemap()
    for index in eachindex(y)
        observations[(:y, index)] = y[index]
    end
    global obs_master = observations
    return data
end

"""Generate an initial width-model trace constrained to `hidden_width`."""
function initial_xor_trace(hidden_width::Int=1)
    hidden_width in k_list || throw(ArgumentError("hidden width is outside k_list"))
    observations = choicemap()
    for index in eachindex(y)
        observations[(:y, index)] = y[index]
    end
    observations[(:k, 1)] = hidden_width
    (trace,) = generate(classifier, (xt,), observations)
    return trace
end

"""Run a short, single-chain version of the historical XOR sampler."""
function run_xor(;
    iterations::Int=1,
    samples_per_mode::Int=50,
    variance::Float64=0.015,
    seed::Int=1,
    initial_width::Int=1,
    maximum_width::Int=16,
    target_acceptance::Float64=0.65,
    nuts_samples::Int=4,
    nuts_adaptation::Int=1,
)
    data = generate_xor_data(
        samples_per_mode=samples_per_mode,
        variance=variance,
        seed=seed,
    )
    prepare_xor!(
        data;
        maximum_width=maximum_width,
        target_acceptance=target_acceptance,
        nuts_samples=nuts_samples,
        nuts_adaptation=nuts_adaptation,
    )
    Random.seed!(seed)
    trace = initial_xor_trace(initial_width)
    traces = Any[]
    scores = Float64[]
    widths = Int[]
    across_acceptance = Int[]
    within_acceptance = Int[]

    for iteration in 1:iterations
        trace, accepted_across, accepted_within =
            RJNUTS_parallel(trace, 1, iteration)
        push!(traces, trace)
        push!(scores, get_score(trace))
        push!(widths, trace[(:k, 1)])
        push!(across_acceptance, accepted_across)
        push!(within_acceptance, accepted_within)
    end

    settings = (;
        iterations,
        samples_per_mode,
        variance,
        seed,
        initial_width,
        maximum_width,
        target_acceptance,
        nuts_samples,
        nuts_adaptation,
    )
    return XORResult(
        data,
        traces,
        scores,
        widths,
        across_acceptance,
        within_acceptance,
        settings,
    )
end

"""Serialize an XOR result, creating its parent directory when necessary."""
function save_xor_result(path::AbstractString, result::XORResult)
    mkpath(dirname(path))
    open(path, "w") do io
        serialize(io, result)
    end
    return path
end

"""Load a result produced by [`save_xor_result`](@ref)."""
function load_xor_result(path::AbstractString)
    open(path, "r") do io
        result = deserialize(io)
        result isa XORResult || error("file does not contain an XORResult")
        return result
    end
end

"""Return class probabilities for observations stored one per matrix row."""
predict_probabilities(trace, observations::AbstractMatrix) =
    G(transpose(observations), trace)

function predicted_labels(probabilities::AbstractMatrix)
    return [argmax(view(probabilities, :, index)) for index in axes(probabilities, 2)]
end

function classification_accuracy(trace, observations::AbstractMatrix, labels)
    predictions = predicted_labels(predict_probabilities(trace, observations))
    return sum(predictions .== labels) / length(labels)
end

function posterior_accuracy(traces, observations::AbstractMatrix, labels)
    isempty(traces) && throw(ArgumentError("at least one trace is required"))
    probabilities = zeros(Float64, 2, size(observations, 1))
    for trace in traces
        probabilities .+= predict_probabilities(trace, observations)
    end
    probabilities ./= length(traces)
    predictions = predicted_labels(probabilities)
    return sum(predictions .== labels) / length(labels)
end
