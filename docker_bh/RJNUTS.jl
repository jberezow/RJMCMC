function propose_hyperparameters(trace)
    
    hyper_selection = select()
    push!(hyper_selection, :τ₁)
    push!(hyper_selection, :τ₂)
    push!(hyper_selection, :τᵧ)
    (new_trace, weight, retdiff) = regenerate(trace, hyper_selection)
    
    if log(rand()) < weight
        return (new_trace, 1)
    else
        return (trace, 0)
    end
end;

function nuts_parameters(trace)
    
    l = trace[:l]
    param_selection = select()
    for i=1:l+1 #Number of Layers
        #push!(param_selection, (:μ,i))
        #push!(param_selection, (:μb,i))
        push!(param_selection, (:W,i))
        push!(param_selection, (:b,i))
    end
    
    prev_score = get_score(trace)
    
    acc = 0
    new_trace = NUTS(trace, param_selection, 0.65, m, m, false)[m+1]
    new_score = get_score(new_trace)
    
    if prev_score != new_score
        return (new_trace, 1)
    else
        return (trace, 0)
    end
    
    return (trace, acc)
end

function nuts_hyperparameters2(trace)
    
    l = trace[:l]
    hyper2_selection = select()
    for i=1:l+1 #Number of Layers
        push!(hyper2_selection, (:μ,i))
        push!(hyper2_selection, (:μb,i))
    end
    
    prev_score = get_score(trace)
    new_trace = NUTS(trace, hyper2_selection, 0.65, m, m, false)[m+1]
    new_score = get_score(new_trace)
    weight = new_score - prev_score
    #println(weight)
    
    #(new_trace, weight, retdiff) = regenerate(trace, hyper2_selection)
    
    #if log(rand()) < weight
    #    return (new_trace, 1)
    #else
    #    return (trace, 0)
    #end
    
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

    (trace_tilde,) = propose_hyperparameters(trace_tilde)
    (trace_tilde,) = nuts_hyperparameters2(trace_tilde)
    (trace_tilde,) = nuts_parameters(trace_tilde)

    #Reversible Jump Step
    (trace_prime, q_weight) = layer_change(trace_tilde)
    
    #NUTS Step 2
    trace_star = trace_prime

    (trace_star,) = nuts_parameters(trace_star)
    (trace_star,) = nuts_hyperparameters2(trace_star)
    (trace_star,) = propose_hyperparameters(trace_star) 
    #################################################RJNUTS#################################################
        
    model_score = -get_score(init_trace) + get_score(trace_star)
    across_score = model_score + q_weight
    #println(across_score)
    #println(model_score)

    if rand() < exp(across_score)
        #println("********** Accepted: $(trace_star[:l]) **********")
        return (trace_star, 1)
    else
        return (init_trace, 0)
    end
end

function RJNUTS(trace, iters=10, chain=1)
    traces = []
    scores = []
    across_acceptance = []
    within_acceptance = []
    hyper1_acceptance = []
    hyper2_acceptance = []
    
    for i=1:iters
        (trace, accepted) = node_parameter(trace)
        push!(across_acceptance, accepted)
        (trace, accepted)  = propose_hyperparameters(trace)
        push!(hyper1_acceptance, accepted)
        (trace, accepted)  = nuts_hyperparameters2(trace)
        push!(hyper2_acceptance, accepted)
        (trace, accepted)  = nuts_parameters(trace)
        push!(within_acceptance, accepted)
        
        push!(scores,get_score(trace))
        push!(traces, trace)
        #println("$i : $(get_score(trace))")
        
        if i%5 == 0
            a_acc = 100*(sum(across_acceptance)/length(across_acceptance))
            w_acc = 100*(sum(within_acceptance)/length(within_acceptance))
            h_acc = 100*(sum(hyper1_acceptance)/length(hyper1_acceptance))
            h2_acc = 100*(sum(hyper2_acceptance)/length(hyper2_acceptance))
            println("Epoch $i A Acceptance Probability: $a_acc %")
            println("Epoch $i W Acceptance Probability: $w_acc %")
            println("Epoch $i H Acceptance Probability: $h_acc %")
            println("Epoch $i H2 Acceptance Probability: $h2_acc %")
        end
    end
    return traces, scores
end
