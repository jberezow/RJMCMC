function load_layer(l)
    layer_selection = select()
    push!(layer_selection, (:W,l))
    push!(layer_selection, (:b,l))
    return layer_selection
end

function nuts_parameters(trace)
    
    l = trace[:l]
    param_selection = select()
    for i=1:l+1 #Number of Layers
        push!(param_selection, (:W,i))
        push!(param_selection, (:b,i))
    end
    
    prev_score = get_score(trace)

    new_trace = NUTS(trace, param_selection, acc_prob, m, m, false)[m+1]
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

function layer_nuts(trace,mode="draw")
    prev_score = get_score(trace)
    new_trace = trace
    if mode == "draw"
        mode = bernoulli(0.5) ? "for" : "back"
    end
    
    #Backward Pass
    if mode == "back"
        for j=1:new_trace[:l]+1
            v = new_trace[:l]+2 - j
            layer_selection = load_layer(v)
            #println("Current Layer: $v")
            new_trace = NUTS(new_trace, layer_selection, acc_prob, m, m2, false)[m+1]
        end
        
    #Forward Pass
    else
        for j=1:new_trace[:l]+1
            v = j
            layer_selection = load_layer(v)
            #println("Current Layer: $v")
            new_trace = NUTS(new_trace, layer_selection, acc_prob, m, m2, false)[m+1]
        end
    end
    
    new_score = get_score(new_trace)
    if prev_score != new_score
        return (new_trace, 1)
    else
        return (trace, 0)
    end
    
end

function node_parameter(trace)
    obs = obs_master
    
    init_trace = trace
    
    #################################################RJNUTS#################################################
    #NUTS Step 1
    trace_tilde = trace
    for i=1:1
        (trace_tilde,) = nuts_parameters(trace_tilde)
    end
    
    #Reversible Jump Step
    (trace_prime, q_weight) = node_change(trace_tilde)
    
    #NUTS Step 2
    trace_star = trace_prime
    for i=1:1
        (trace_star,) = nuts_parameters(trace_star)
    end
    #################################################RJNUTS#################################################
        
    model_score = -get_score(init_trace) + get_score(trace_star)
    across_score = model_score + q_weight

    if rand() < exp(across_score)
        println("********** Accepted: $(trace_star[(:k,1)]) **********")
        return (trace_star, 1)
    else
        return (init_trace, 0)
    end
end

function RJNUTS(trace, iters, chain)
    traces = []
    scores = []
    across_acceptance = []
    within_acceptance = []
    
    for i=1:iters
        (trace, accepted) = node_parameter(trace)
        push!(across_acceptance, accepted)
        (trace, accepted) = nuts_parameters(trace)
        push!(within_acceptance, accepted)
        push!(scores,get_score(trace))
        push!(traces, trace)
        println("Chain $chain Iter $i : $(get_score(trace))")
        if i%10 == 0
            a_acc = 100*(sum(across_acceptance)/length(across_acceptance))
            w_acc = 100*(sum(within_acceptance)/length(within_acceptance))
            println("Chain $chain Epoch $i A Acceptance Probability: $a_acc %")
            println("Chain $chain Epoch $i W Acceptance Probability: $w_acc %")
        end
        flush(stdout)
    end
    
    return traces, scores
end

function RJNUTS_parallel(trace, chain, ci)
    
    (trace, a_acc) = node_parameter(trace)
    (trace, w_acc) = nuts_parameters(trace)
    current_k = trace[(:k,1)]
    println("Chain $chain Iter $ci : $(get_score(trace)), Layer Count: $current_k")

    return trace, a_acc, w_acc
end
