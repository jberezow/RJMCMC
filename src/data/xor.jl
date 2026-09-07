"""Training and test data for the synthetic XOR experiment."""
struct XORData <: ExperimentData
    x_train::Matrix{Float64}
    y_train::Vector{Int}
    x_test::Matrix{Float64}
    y_test::Vector{Int}
end

"""Recorded output from one single-chain XOR run."""
struct XORResult <: ChainResult
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
from `OptDigits/dockerxor`. `variance` is passed directly as the diagonal
covariance of `MvNormal`.
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

"""Install one XOR dataset and its sampler settings."""
function prepare_xor!(
    data::XORData;
    maximum_width::Int=16,
    target_acceptance::Float64=0.65,
    nuts_samples::Int=4,
    nuts_adaptation::Int=1,
    divergence_threshold::Real=1,
)
    install_data!(
        data.x_train,
        data.y_train;
        classes=2,
        maximum_width=maximum_width,
        scale=0.5,
        target_acceptance=target_acceptance,
        nuts_samples=nuts_samples,
        nuts_adaptation=nuts_adaptation,
        divergence_threshold=divergence_threshold,
    )
    return data
end

"""Run a single XOR chain."""
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
    trace = initial_trace(initial_width)
    traces, scores, widths, across_acceptance, within_acceptance =
        run_chain(trace, iterations)

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
