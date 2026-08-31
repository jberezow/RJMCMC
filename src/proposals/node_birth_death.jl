#------------------------
#Node Birth / Death Moves
#------------------------

function obs_maker(trace)
    obs_new = choicemap()::ChoiceMap
    obs_new[:l] = trace[:l]
    for i=1:length(y)
        obs_new[(:y,i)] = y[i]
    end
    return obs_new
end

function node_change(trace)
    current_k = trace[(:k,1)]
    if current_k == last(k_list)
        new_trace = node_death(trace)
    elseif current_k == k_list[1]
        new_trace = node_birth(trace)
    else
        new_trace = bernoulli(0.5) ? node_birth(trace) : node_death(trace)
    end
    return new_trace
end
    
@gen function node_birth(trace)
    previous_k = trace[(:k,1)]
    new_k = k_list[previous_k+1]
    difference = abs(new_k - previous_k)
        
    #Select Insertion Place for New Node
    insert = rand((1:new_k))
    
    obs_new = obs_maker(trace)
    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    obs_new[(:k,1)] = new_k
    obs_new[(:k,2)] = trace[(:k,2)]
        
    #Get the input and output dimensions of the edit layer
    in_dim, out_dim = layer_unpacker(1,trace[:l],[obs_new[(:k,i)] for i=1:obs_new[:l]])
    output = trace[:l] + 1
    
    #-------------------
    #Modify Hidden Layer
    #-------------------
    #Hidden Layer Weights
    W1 = [trace[(:W,1)][j] for j=1:length(trace[(:W,1)])]
    σ = 1.0
    hidden_weights = []
    for j=1:in_dim
        hidden_weight = normal(0,σ)
        push!(hidden_weights,hidden_weight)
        W1 = insert!(W1,insert+((j-1)*(out_dim)),hidden_weight)
    end
    obs_new[(:W,1)] = W1
    
    #Hidden Layer Bias
    b1 = [trace[(:b,1)][j] for j=1:length(trace[(:b,1)])]
    new_bias = normal(0,σ)
    obs_new[(:b,1)] = insert!(b1, insert, new_bias)

    #Output Layer Weights
    W2 = [trace[(:W,output)][j] for j=1:length(trace[(:W,output)])]
    output_weights = []
    for j=1:trace[(:k,2)]
        output_weight = normal(0,σ)
        push!(output_weights,output_weight)
        W2 = insert!(W2,(insert-1)*(trace[(:k,2)])+j,output_weight)
    end
    obs_new[(:W,2)] = W2
    
    #Hidden Layer Bias
    obs_new[(:b,output)] = trace[(:b,output)]
    #-------------------
    
    #Determine q_score
    q_score = 0
    for i=1:length(hidden_weights)
        q_score += log(pdf(Normal(0,σ),hidden_weights[i]))
    end
    q_score += log(pdf(Normal(0,σ),new_bias))
    for i=1:length(output_weights)
        q_score += log(pdf(Normal(0,σ),output_weights[i]))
    end
        
    #Update Trace and Return Trace and Weights
    (new_trace,) = generate(classifier, (xt,), obs_new)
    node_penalty = log(1/new_k)
    q = -q_score + node_penalty
        
    return (new_trace, q)
      
end

@gen function node_death(trace)
    previous_k = trace[(:k,1)]
    new_k = k_list[previous_k-1]
    difference = abs(new_k - previous_k)
        
    #Select Insertion Place for New Node
    delete = rand((1:previous_k))
    
    obs_new = obs_maker(trace)
    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    obs_new[(:k,1)] = new_k
    obs_new[(:k,2)] = trace[(:k,2)]
        
    #Get the input and output dimensions of the edit layer
    in_dim, out_dim = layer_unpacker(1,trace[:l],[obs_new[(:k,i)] for i=1:obs_new[:l]])
    output = trace[:l] + 1
    
    #-------------------
    #Modify Hidden Layer
    #-------------------
    #Hidden Layer Weights
    W1 = [trace[(:W,1)][j] for j=1:length(trace[(:W,1)])]
    σ = 1.0
    hidden_weights = []
    for j=1:in_dim
        hidden_weight = W1[delete+((j-1)*(out_dim))]
        push!(hidden_weights,hidden_weight)
        W1 = deleteat!(W1,delete+((j-1)*(out_dim)))
    end
    obs_new[(:W,1)] = W1
    
    #Hidden Layer Bias
    b1 = [trace[(:b,1)][j] for j=1:length(trace[(:b,1)])]
    new_bias = b1[delete]
    obs_new[(:b,1)] = deleteat!(b1, delete)

    #Output Layer Weights
    W2 = [trace[(:W,output)][j] for j=1:length(trace[(:W,output)])]
    output_weights = []
    for j=1:trace[(:k,2)]
        output_weight = W2[(delete-1)*(trace[(:k,2)])+1]
        push!(output_weights,output_weight)
        W2 = deleteat!(W2,(delete-1)*(trace[(:k,2)])+1)
    end
    obs_new[(:W,2)] = W2
    
    #Hidden Layer Bias
    obs_new[(:b,output)] = trace[(:b,output)]
    #-------------------
    
    #Determine q_score
    q_score = 0
    for i=1:length(hidden_weights)
        q_score += log(pdf(Normal(0,σ),hidden_weights[i]))
    end
    q_score += log(pdf(Normal(0,σ),new_bias))
    for i=1:length(output_weights)
        q_score += log(pdf(Normal(0,σ),output_weights[i]))
    end
        
    #Update Trace and Return Trace and Weights
    (new_trace,) = generate(classifier, (xt,), obs_new)
    node_penalty = log(previous_k)
    q = -q_score + node_penalty
    q = q_score
        
    return (new_trace, q)
end