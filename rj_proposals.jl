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
    obs_new[:l] = trace[:l]
    obs_new[:τ₁] = trace[:τ₁]
    obs_new[:τ₂] = trace[:τ₂]
    obs_new[:τ₃] = trace[:τ₃]
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

function node_change(trace)
    current_l = trace[:l]
    layer = rand((1:current_l))
    #println("Proposed layer: $layer")
    current_k = trace[(:k, layer)]
    if current_k == last(k_list)
        new_trace = node_death(trace,layer)
    elseif current_k == k_list[1]
        new_trace = node_birth(trace,layer)
    else
        new_trace = bernoulli(0.5) ? node_birth(trace,layer) : node_death(trace,layer)
    end
    return new_trace        
end

@gen function node_birth(trace, layer)
    
    previous_k = trace[(:k,layer)]
    new_k = k_list[previous_k + 1]
    difference = abs(new_k - previous_k)
    
    #Select Insertion Place for New Neuron
    insert = rand((1:new_k))
    
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
    
    #Hyperparameters for new proposals
    σ₁ = sqrt(1/obs_new[:τ₁])
    σ₂ = sqrt(1/obs_new[:τ₂])
    σ₃ = sqrt(1/obs_new[:τ₃])
    
    #Q vector is new parameters
    
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

@gen function node_death(trace, layer)
    
    previous_k = trace[(:k,layer)]
    new_k = k_list[previous_k - 1]
    difference = abs(new_k - previous_k)
    
    #Select Insertion Place for New Neuron
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