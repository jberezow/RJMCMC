#-------------------------
#Node Birth / Death Moves
#-------------------------

function obs_maker(trace)
    obs_new = choicemap()::ChoiceMap
    if network == "classifier"
        for i=1:length(classes)
            obs_new[(:y,i)] = classes[i]
        end
    else
        obs_new[:y] = y
        obs_new[:τᵧ] = trace[:τᵧ]
    end
    obs_new[:τ₁] = trace[:τ₁]
    obs_new[:τ₂] = trace[:τ₂]
    return obs_new
end

function obs_maker_regression(trace)
    obs_new = choicemap()::ChoiceMap
    obs_new[:y] = y
    obs_new[:l] = trace[:l]
    obs_new[:τ₁] = trace[:τ₁]
    obs_new[:τ₂] = trace[:τ₂]
    obs_new[:τ₃] = trace[:τ₃]
    obs_new[:τᵧ] = trace[:τᵧ]
    return obs_new
end

@gen function node_change(trace)
    #current_l = 1#trace[:l]
    layer = 1 #rand((1:current_l))
    #println("Proposed layer: $layer")
    current_k = trace[(:k, layer)]
    if current_k == last(k_list)
        new_trace = death(trace)
    elseif current_k == k_list[1]
        new_trace = birth(trace)
    else
        new_trace = bernoulli(0.5) ? birth(trace) : death(trace)
    end
    return new_trace        
end

@gen function birth(trace)
    
    previous_k = trace[(:k,1)]
    new_k = k_list[previous_k + 1]
    difference = abs(new_k - previous_k) #Just one neuron for now
    obs_new = obs_maker(trace)
    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    
    #Select Insertion Place for New Neuron
    in_dim, out_dim = layer_unpacker(1,1,trace[(:k,1)])
    insert = rand((1:new_k))
    
    #Pull hyperparams from trace
    σ₁ = 1/obs_new[:τ₁]
    σ₂ = 1/obs_new[:τ₂]
    
    #Draw new hyperparams
    hw1_μ ~ normal(0, 1)
    hw2_μ ~ normal(0, 1)
    hb_μ ~ normal(0, 1)
    ow_μ ~ normal(0, 1)
    
    #Draw new params
    hw1 ~ normal(hw1_μ, σ₁)
    hw2 ~ normal(hw2_μ, σ₁)
    hb ~ normal(hb_μ, σ₁)
    ow ~ normal(ow_μ, σ₂)
    
    #Calculate proposal score for new params and hyperparams
    hw1_μ_score = Distributions.logpdf(Normal(0, 1), hw1_μ)
    hw2_μ_score = Distributions.logpdf(Normal(0, 1), hw2_μ)
    hb_μ_score = Distributions.logpdf(Normal(0, 1), hb_μ)
    ow_μ_score = Distributions.logpdf(Normal(0, 1), ow_μ)
    
    hw1_score = Distributions.logpdf(Normal(hw1_μ, σ₁), hw1)
    hw2_score = Distributions.logpdf(Normal(hw2_μ, σ₁), hw2)
    hb_score = Distributions.logpdf(Normal(hb_μ, σ₁), hb)
    ow_score = Distributions.logpdf(Normal(ow_μ, σ₂), ow)
    
    #Modify Weight Matrices and Bias Vector
    #-------------------------------------
    #Pull previous parameter vectors
    W1_μ = [trace[(:μ,1)][j] for j=1:length(trace[(:μ,1)])]
    b1_μ = [trace[(:μb,1)][j] for j=1:length(trace[(:μb,1)])]
    W1 = [trace[(:W,1)][j] for j=1:length(trace[(:W,1)])]
    b1 = [trace[(:b,1)][j] for j=1:length(trace[(:b,1)])]

    W2_μ = [trace[(:μ,2)][j] for j=1:length(trace[(:μ,2)])]
    W2 = [trace[(:W,2)][j] for j=1:length(trace[(:W,2)])]
    
    #Add the new parameters
    W1_μ = insert!(W1_μ,insert,hw1_μ)
    W1_μ = insert!(W1_μ,insert+out_dim+1,hw2_μ)
    b1_μ = insert!(b1_μ,insert,hb_μ)
    W1 = insert!(W1,insert,hw1)
    W1 = insert!(W1,insert+out_dim+1,hw2)
    b1 = insert!(b1,insert,hb)
    
    W2_μ = insert!(W2_μ,insert,ow_μ)
    W2 = insert!(W2,insert,ow)
    #---------------------------------------
    
    #Update the Trace
    obs_new[(:k,1)] = new_k
    obs_new[(:k,2)] = 1
    obs_new[(:μ,1)] = W1_μ
    obs_new[(:μ,2)] = W2_μ
    obs_new[(:μb,1)] = b1_μ
    obs_new[(:μb,2)] = trace[(:μb,2)]
    obs_new[(:W,1)] = W1
    obs_new[(:W,2)] = W2
    obs_new[(:b,1)] = b1
    obs_new[(:b,2)] = trace[(:b,2)]
    (new_trace,_,_,_) = update(trace, args, argdiffs, obs_new)
    
    #Update Trace and Calculate Weights
    q_forward = sum([hw1_μ_score, hw2_μ_score, hb_μ_score, ow_μ_score, hw1_score, hw2_score, hb_score, ow_score])
    q = -q_forward
    
    return(new_trace, q)
end

@gen function death(trace)
    
    previous_k = trace[(:k,1)]
    new_k = k_list[previous_k - 1]
    difference = abs(new_k - previous_k) #Just one neuron for now
    obs_new = obs_maker(trace)
    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    
    #Select Insertion Place for New Neuron
    in_dim, out_dim = layer_unpacker(1,1,trace[(:k,1)])
    delete = rand((1:previous_k))
    
    #Pull hyperparams from trace
    σ₁ = 1/obs_new[:τ₁]
    σ₂ = 1/obs_new[:τ₂]
    
    #Locate indices for parameters to be deleted
    hw1_μ = trace[(:μ,1)][delete]
    hw2_μ = trace[(:μ,1)][delete+out_dim]
    hb_μ = trace[(:μb,1)][delete]
    ow_μ = trace[(:μ,2)][delete]
    
    hw1 = trace[(:W,1)][delete]
    hw2 = trace[(:W,1)][delete+out_dim]
    hb = trace[(:b,1)][delete]
    ow = trace[(:W,2)][delete]
    
    #Calculate proposal score for deleted params and hyperparams
    hw1_μ_score = Distributions.logpdf(Normal(0, 1), hw1_μ)
    hw2_μ_score = Distributions.logpdf(Normal(0, 1), hw2_μ)
    hb_μ_score = Distributions.logpdf(Normal(0, 1), hb_μ)
    ow_μ_score = Distributions.logpdf(Normal(0, 1), ow_μ)
    
    hw1_score = Distributions.logpdf(Normal(hw1_μ, σ₁), hw1)
    hw2_score = Distributions.logpdf(Normal(hw2_μ, σ₁), hw2)
    hb_score = Distributions.logpdf(Normal(hb_μ, σ₁), hb)
    ow_score = Distributions.logpdf(Normal(ow_μ, σ₂), ow)
    
    #Modify Weight Matrices and Bias Vector
    #-------------------------------------
    #Pull previous parameter vectors
    W1_μ = [trace[(:μ,1)][j] for j=1:length(trace[(:μ,1)])]
    b1_μ = [trace[(:μb,1)][j] for j=1:length(trace[(:μb,1)])]
    W1 = [trace[(:W,1)][j] for j=1:length(trace[(:W,1)])]
    b1 = [trace[(:b,1)][j] for j=1:length(trace[(:b,1)])]

    W2_μ = [trace[(:μ,2)][j] for j=1:length(trace[(:μ,2)])]
    W2 = [trace[(:W,2)][j] for j=1:length(trace[(:W,2)])]
    
    #Add the new parameters
    W1_μ = deleteat!(W1_μ,delete)
    W1_μ = deleteat!(W1_μ,delete+out_dim-1)
    b1_μ = deleteat!(b1_μ,delete)
    W1 = deleteat!(W1,delete)
    W1 = deleteat!(W1,delete+out_dim-1)
    b1 = deleteat!(b1,delete)
    
    W2_μ = deleteat!(W2_μ,delete)
    W2 = deleteat!(W2,delete)
    #---------------------------------------

    #Update the Trace
    obs_new[(:k,1)] = new_k
    obs_new[(:k,2)] = 1
    obs_new[(:μ,1)] = W1_μ
    obs_new[(:μ,2)] = W2_μ
    obs_new[(:μb,1)] = b1_μ
    obs_new[(:μb,2)] = trace[(:μb,2)]
    obs_new[(:W,1)] = W1
    obs_new[(:W,2)] = W2
    obs_new[(:b,1)] = b1
    obs_new[(:b,2)] = trace[(:b,2)]
    (new_trace,_,_,_) = update(trace, args, argdiffs, obs_new)
    
    #Update Trace and Calculate Weights
    q_backward = sum([hw1_μ_score, hw2_μ_score, hb_μ_score, ow_μ_score, hw1_score, hw2_score, hb_score, ow_score])
    q = q_backward
    
    return(new_trace, q)
    
end

@gen function node_death(trace, layer)
    
    previous_k = trace[(:k,layer)]
    new_k = k_list[previous_k - 1]
    difference = abs(new_k - previous_k)
    
    #Select Deletion Place for New Neuron
    delete = rand((1:previous_k))
    #delete=5
    
    #Create new choicemap and fill with real Y values
    obs_new = obs_maker(trace)
    obs_new[:l] = trace[:l]
    
    #Fill k values in new ChoiceMap
    for i=1:trace[:l]+1
        if i == layer
            obs_new[(:k,i)] = new_k
        elseif i < layer
            obs_new[(:k,i)] = trace[(:k,i)]
        else
            obs_new[(:k,i)] = trace[(:k,i)]
        end
    end
    
    #Get the input and output dimensions of the edit layer
    in_dim, out_dim = layer_unpacker(layer,trace[:l],[obs_new[(:k,i)] for i=1:obs_new[:l]])
    output = trace[:l] + 1
    
    for i=1:trace[:l]
        if i == layer
            obs_new[(:k,i)] = new_k
            W1 = [trace[(:W,i)][j] for j=1:length(trace[(:W,i)])]
            b1 = [trace[(:b,i)][j] for j=1:length(trace[(:b,i)])]
            for j=1:in_dim
                W1 = deleteat!(W1,delete+((j-1)*(out_dim)))
            end
            obs_new[(:W,i)] = W1
            obs_new[(:b,i)] = deleteat!(b1, delete)
            
            if layer == trace[:l]
                W2 = [trace[(:W,output)][j] for j=1:length(trace[(:W,output)])]
                obs_new[(:W,output)] = deleteat!(W2,delete)
                obs_new[(:b,output)] = trace[(:b,output)]
            else
                new_dim = obs_new[(:k,i+1)]
                W2 = [trace[(:W,i+1)][j] for j=1:length(trace[(:W,i+1)])]
                b2 = [trace[(:b,i+1)][j] for j=1:length(trace[(:b,i+1)])]
                for j=1:new_dim
                    W2 = deleteat!(W2,new_dim*(delete-1)+1)
                end
                obs_new[(:W,i+1)] = W2
                obs_new[(:b,i+1)] = b2
                obs_new[(:W,output)] = trace[(:W,output)]
                obs_new[(:b,output)] = trace[(:b,output)]
            end
            
        elseif i == layer + 1
            continue
        elseif i < layer
            obs_new[(:W,i)] = trace[(:W,i)]
            obs_new[(:b,i)] = trace[(:b,i)]
        else
            obs_new[(:W,i)] = trace[(:W,i)]
            obs_new[(:b,i)] = trace[(:b,i)]
        end
    end
    
    (new_trace, weight) = generate(classifier, (x,), obs_new)
    return new_trace
    
end;

#-------------------------
#Layer Birth / Death Moves
#-------------------------

@gen function layer_birth(trace, layer)
    
    previous_l = trace[:l]
    new_l = l_list[l+1]
    difference = abs(new_l - previous_l)
    
    #Select Insertion Place for New Neuron
    insert = l + 1
    
    #Create new choicemap and fill with real Y values
    obs_new = obs_maker(trace)
    obs_new[:l] = trace[:l]
    
    #Fill k values in new ChoiceMap
    for i=1:trace[:l]+1
        obs_new[(:k,i)] = k_real
    end
    
    #Get the input and output dimensions of the edit layer
    in_dim, out_dim = layer_unpacker(layer,trace[:l],[obs_new[(:k,i)] for i=1:obs_new[:l]])
    output = trace[:l] + 1
    
    #Hyperparameters for new proposals
    σ₁ = 1/obs_new[:τ₁]
    σ₂ = 1/obs_new[:τ₂]
    σ₃ = 1/obs_new[:τ₃]
    
    
    #TO-DO Jan 15 10:00 PM: Write proper layer function
    #Tack on layer at the end and draw parameters from prior
    for i=1:trace[:l]
        if i == layer
            obs_new[(:k,i)] = new_k
            W1 = [trace[(:W,i)][j] for j=1:length(trace[(:W,i)])]
            b1 = [trace[(:b,i)][j] for j=1:length(trace[(:b,i)])]
            for j=1:in_dim
                W1 = insert!(W1,insert+((j-1)*(out_dim)),normal(0,σ₁))
            end
            obs_new[(:W,i)] = W1
            obs_new[(:b,i)] = insert!(b1, insert, normal(1,σ₂))
            
            if layer == trace[:l]
                W2 = [trace[(:W,output)][j] for j=1:length(trace[(:W,output)])]
                obs_new[(:W,output)] = insert!(W2,insert,normal(0,σ₃))
                obs_new[(:b,output)] = trace[(:b,output)]
            else
                new_dim = obs_new[(:k,i+1)]
                W2 = [trace[(:W,i+1)][j] for j=1:length(trace[(:W,i+1)])]
                b2 = [trace[(:b,i+1)][j] for j=1:length(trace[(:b,i+1)])]
                for j=1:new_dim
                    W2 = insert!(W2,new_dim*(insert-1)+1,normal(0,σ₁))
                end
                obs_new[(:W,i+1)] = W2
                obs_new[(:b,i+1)] = b2
                obs_new[(:W,output)] = trace[(:W,output)]
                obs_new[(:b,output)] = trace[(:b,output)]
            end
            
        elseif i == layer + 1
            continue
        elseif i < layer
            obs_new[(:W,i)] = trace[(:W,i)]
            obs_new[(:b,i)] = trace[(:b,i)]
        else
            obs_new[(:W,i)] = trace[(:W,i)]
            obs_new[(:b,i)] = trace[(:b,i)]
        end
    end
    
    (new_trace, weight) = generate(classifier, (x,), obs_new)
    return new_trace
    
end