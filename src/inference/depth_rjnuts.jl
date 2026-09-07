# Depth RJNUTS driver, from BostonHousing/docker-parallel2a/RJNUTS.jl.
module DepthRJNUTS
using Gen
using Distributions
using ..LayerProposals: layer_change
import ..NUTS

export DepthSettings

"""
Sampler settings for one Boston experiment.

`within_dimension_update` selects the within-dimension step: `:mixed` chooses
layer-wise or all-parameter NUTS with equal probability, `:all` always takes the
all-parameter step.
"""
struct DepthSettings
    target_acceptance::Float64
    nuts_samples::Int
    nuts_adaptation::Int
    within_dimension_update::Symbol

    function DepthSettings(target_acceptance, nuts_samples, nuts_adaptation,
                           within_dimension_update)
        within_dimension_update in (:mixed, :all) ||
            throw(ArgumentError("within_dimension_update must be :mixed or :all"))
        new(target_acceptance, nuts_samples, nuts_adaptation, within_dimension_update)
    end
end

function load_layer(l)
    layer_selection = select()
    push!(layer_selection, (:W,l))
    push!(layer_selection, (:b,l))
    return layer_selection
end

function nuts_parameters(trace, s::DepthSettings)
    l = trace[:l]
    param_selection = select()
    for i=1:l+1 #Number of Layers
        push!(param_selection, (:W,i))
        push!(param_selection, (:b,i))
    end

    prev_score = get_score(trace)

    new_trace = NUTS(trace, param_selection, s.target_acceptance,
                     s.nuts_samples, s.nuts_adaptation, false)[s.nuts_samples+1]
    new_score = get_score(new_trace)
    nuts_score = new_score - prev_score

    if exp(nuts_score) == 1
        accepted = 0
        return (trace, accepted)
    else
        accepted = 1
        return (new_trace, accepted)
    end
end

# The layer-wise pass passes `m` as both NUTS sample arguments.
function layer_nuts(trace, s::DepthSettings, mode="draw")
    prev_score = get_score(trace)
    new_trace = trace
    if mode == "draw"
        mode = bernoulli(0.5) ? "forward" : "backward"
    end

    #Backward Pass
    if mode == "backward"
        for j=1:new_trace[:l]+1
            v = new_trace[:l]+2 - j
            layer_selection = load_layer(v)
            new_trace = NUTS(new_trace, layer_selection, s.target_acceptance,
                             s.nuts_samples, s.nuts_samples, false)[s.nuts_samples+1]
        end

    #Forward Pass
    else
        for j=1:new_trace[:l]+1
            v = j
            layer_selection = load_layer(v)
            new_trace = NUTS(new_trace, layer_selection, s.target_acceptance,
                             s.nuts_samples, s.nuts_samples, false)[s.nuts_samples+1]
        end
    end

    new_score = get_score(new_trace)
    nuts_score = new_score - prev_score

    if exp(nuts_score) == 1
        accepted = 0
        return (trace, accepted)
    else
        accepted = 1
        return (new_trace, accepted)
    end
end

"""
Across-dimension move: a palindromic composition of NUTS steps around one
reversible-jump layer proposal.
"""
function layer_parameter(trace, s::DepthSettings)
    init_trace = trace

    #NUTS Step 1
    trace_tilde = trace
    for i=1:1
        (trace_tilde,) = nuts_parameters(trace_tilde, s)
        (trace_tilde,) = layer_nuts(trace_tilde, s, "forward")
    end

    #Reversible Jump Step
    (trace_prime, q_weight) = layer_change(trace_tilde)

    #NUTS Step 2
    trace_star = trace_prime
    for i=1:1
        (trace_star,) = layer_nuts(trace_star, s, "backward")
        (trace_star,) = nuts_parameters(trace_star, s)
    end

    model_score = -get_score(init_trace) + get_score(trace_star)
    across_score = model_score + q_weight

    if rand() < exp(across_score)
        println("********** Accepted: $(trace_star[:l]) **********")
        return (trace_star, 1)
    else
        return (init_trace, 0)
    end
end

"""One RJNUTS iteration: an across-dimension move then a within-dimension move."""
function RJNUTS_parallel(trace, s::DepthSettings, chain, ci)
    (trace, a_acc) = layer_parameter(trace, s)

    if s.within_dimension_update == :all
        (trace, w_acc) = nuts_parameters(trace, s)
    elseif rand(Uniform(0,1)) < 0.5
        (trace, w_acc) = layer_nuts(trace, s)
    else
        (trace, w_acc) = nuts_parameters(trace, s)
    end
    current_l = trace[:l]
    println("Chain $chain Iter $ci : $(get_score(trace)), Layer Count: $current_l")

    return trace, a_acc, w_acc
end

end
