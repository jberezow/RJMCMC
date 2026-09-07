#--------------------------------
#Shared Width-Experiment Runtime
#--------------------------------

# TODO: consider migrating the width family to the depth pattern

"""Supertype for datasets that can be installed for the width sampler."""
abstract type ExperimentData end

"""Supertype for recorded single-chain sampler output."""
abstract type ChainResult end

"""Trace arguments implied by the currently installed dataset and settings."""
model_arguments() = (xt, n_classes, last(k_list), softmax_scale)

"""
    install_data!(inputs, labels; kwargs...)

Install one dataset and its sampler settings as the ambient state read by the
proposal and RJNUTS code. `inputs` holds one observation per row.
"""
function install_data!(
    inputs::AbstractMatrix,
    labels::AbstractVector{Int};
    classes::Int,
    maximum_width::Int,
    scale::Float64,
    target_acceptance::Float64,
    nuts_samples::Int,
    nuts_adaptation::Int,
    divergence_threshold::Real,
)
    maximum_width >= 2 || throw(ArgumentError("RJNUTS requires at least two possible widths"))
    size(inputs, 1) == length(labels) ||
        throw(ArgumentError("inputs and labels disagree on the number of observations"))

    global xt = transpose(inputs)
    global y = labels
    global n_classes = classes
    global softmax_scale = scale
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
    return nothing
end

"""Generate a model trace constrained to the installed data and `hidden_width`."""
function initial_trace(hidden_width::Int)
    hidden_width in k_list || throw(ArgumentError("hidden width is outside k_list"))
    observations = choicemap()
    for index in eachindex(y)
        observations[(:y, index)] = y[index]
    end
    observations[(:k, 1)] = hidden_width
    (trace,) = generate(classifier, model_arguments(), observations)
    return trace
end

"""
    best_initial_trace(hidden_width; candidates=1000, criterion=get_score)

Retain the best of `candidates` prior draws at a fixed hidden width, ranking by
`criterion` and keeping the largest value. Boston ranks by lowest error instead,
through [`best_initial_boston_trace`](@ref).
"""
function best_initial_trace(hidden_width::Int; candidates::Int=1000,
                            criterion=get_score)
    candidates >= 1 || throw(ArgumentError("candidates must be positive"))
    best = initial_trace(hidden_width)
    best_value = criterion(best)
    for _ in 2:candidates
        trace = initial_trace(hidden_width)
        value = criterion(trace)
        if value > best_value
            best, best_value = trace, value
        end
    end
    return best
end

"""Advance one chain for `iterations` RJNUTS steps, recording each state."""
function run_chain(trace, iterations::Int; chain::Int=1)
    traces = Any[]
    scores = Float64[]
    widths = Int[]
    across_acceptance = Int[]
    within_acceptance = Int[]

    for iteration in 1:iterations
        trace, accepted_across, accepted_within = RJNUTS_parallel(trace, chain, iteration)
        push!(traces, trace)
        push!(scores, get_score(trace))
        push!(widths, trace[(:k, 1)])
        push!(across_acceptance, accepted_across)
        push!(within_acceptance, accepted_within)
    end

    return traces, scores, widths, across_acceptance, within_acceptance
end

"""Serialize a chain result, creating its parent directory when necessary."""
function save_result(path::AbstractString, result::ChainResult)
    mkpath(dirname(path))
    open(path, "w") do io
        serialize(io, result)
    end
    return path
end

"""Load a result produced by [`save_result`](@ref)."""
function load_result(path::AbstractString)
    open(path, "r") do io
        result = deserialize(io)
        result isa ChainResult || error("file does not contain a chain result")
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

"""Accuracy of the posterior mean prediction over `traces`."""
function posterior_accuracy(traces, observations::AbstractMatrix, labels)
    isempty(traces) && throw(ArgumentError("at least one trace is required"))
    probabilities = predict_probabilities(first(traces), observations)
    for trace in Iterators.drop(traces, 1)
        probabilities .+= predict_probabilities(trace, observations)
    end
    probabilities ./= length(traces)
    predictions = predicted_labels(probabilities)
    return sum(predictions .== labels) / length(labels)
end
