#-------------------------
#Layer Birth / Death Moves
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
    #obs_new[:τ₃] = trace[:τ₃]
    return obs_new
end

function obs_maker_regression(trace)
    obs_new = choicemap()::ChoiceMap
    obs_new[:y] = y
    obs_new[:l] = trace[:l]
    obs_new[:τ₁] = trace[:τ₁]
    obs_new[:τ₂] = trace[:τ₂]
    obs_new[:τᵧ] = trace[:τᵧ]
    return obs_new
end

function layer_change(trace)
    current_l = trace[:l]
    if current_l == last(l_list)
        new_trace = layer_death(trace)
    elseif current_l == l_list[1]
        new_trace = layer_birth(trace)
    else
        new_trace = bernoulli(0.5) ? layer_birth(trace) : layer_death(trace)
    end
    return new_trace        
end

@gen function layer_birth(trace)
    
    previous_l = trace[:l]
    new_l = l_list[previous_l+1]
    difference = abs(new_l - previous_l)
    
    #Select Insertion Place for New Layer
    insert = previous_l + 1
    
    obs_new = obs_maker(trace)
    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    obs_new[:l] = new_l
    
    #Pull hyperparams from trace
    σ₁ = 1/obs_new[:τ₁]
    σ₂ = 1/obs_new[:τ₂]
    
    #Recast Output Layer
    obs_new[(:k,new_l+1)] = trace[(:k,previous_l+1)]
    obs_new[(:μ,new_l+1)] = trace[(:μ,previous_l+1)]
    obs_new[(:μb,new_l+1)] = trace[(:μb,previous_l+1)]
    obs_new[(:W,new_l+1)] = trace[(:W,previous_l+1)]
    obs_new[(:b,new_l+1)] = trace[(:b,previous_l+1)]
    
    #Modify Weight Matrices and Bias Vector
    #------------------------------------------------
    q_score = 0
    
    for i=1:new_l
        if i == new_l
            k = trace[(:k,i-1)]
            obs_new[(:k,i)] = k
            h = Int(k * k)
            
            #Hidden Weights
            u = zeros(h) 
            S = Diagonal([1 for i=1:length(u)])
            μ = @trace(mvnormal(u,S), (:μ,i))
            Σ = Diagonal([σ₁ for i=1:length(μ)])
            W = @trace(mvnormal(μ,Σ), (:W,i))

            #Hidden Biases
            ub = zeros(k)
            Sb = Diagonal([1 for i=1:length(ub)])    
            μb = @trace(mvnormal(ub,Sb), (:μb,i))
            Σ2 = Diagonal([σ₁ for i=1:length(μb)])
            b = @trace(mvnormal(μb,Σ2), (:b,i))
            
            obs_new[(:μ,i)] = μ
            obs_new[(:μb,i)] = μb
            obs_new[(:W,i)] = W
            obs_new[(:b,i)] = b
            
            q_score = (
                log(pdf(MvNormal(u,S),μ)) + 
                log(pdf(MvNormal(μ,Σ),W)) + 
                log(pdf(MvNormal(ub,Sb),μb)) + 
                log(pdf(MvNormal(μb,Σ2),b))
                )
        else
            obs_new[(:k,i)] = trace[(:k,i)]
            obs_new[(:μ,i)] = trace[(:μ,i)]
            obs_new[(:μb,i)] = trace[(:μb,i)]
            obs_new[(:W,i)] = trace[(:W,i)]
            obs_new[(:b,i)] = trace[(:b,i)]
        end
    end
    #------------------------------------------------
    
    #Update Trace and Return Trace and Weights
    (new_trace,_,_,_) = update(trace, args, argdiffs, obs_new)
    #(new_trace,) = generate(interpolator, (x,), obs_new)
    q = -q_score
        
    return (new_trace, q)
    
end

@gen function layer_death(trace)
    
    previous_l = trace[:l] #4
    new_l = l_list[previous_l-1] #3 
    difference = abs(new_l - previous_l) #1
    
    #Select Insertion Place for New Layer
    output = previous_l + 1 #5
    
    obs_new = obs_maker_regression(trace)
    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    obs_new[:l] = new_l
    
    #Pull hyperparams from trace
    σ₁ = 1/obs_new[:τ₁]
    σ₂ = 1/obs_new[:τ₂]
    
    #Recast Output Layer
    obs_new[(:k,new_l+1)] = trace[(:k,output)] #Layer 4 = Layer 5
    obs_new[(:μ,new_l+1)] = trace[(:μ,output)]
    obs_new[(:μb,new_l+1)] = trace[(:μb,output)]
    obs_new[(:W,new_l+1)] = trace[(:W,output)]
    obs_new[(:b,new_l+1)] = trace[(:b,output)]
    
    #Capture Deleted Layer Weight Matrices and Bias Vector
    #-----------------------------------------------------
    q_score = 0
    
    for i=1:new_l #4
        if i == new_l
            k = trace[(:k,i+1)]
            h = Int(k * k)
            
            #Hidden Weights
            μ = trace[(:μ,output)]
            u = zeros(length(μ))
            S = Diagonal([1 for i=1:length(u)])
            Σ = Diagonal([σ₁ for i=1:length(μ)])
            W = trace[(:W,output)]

            #Hidden Biases
            μb = trace[(:μb,output)]
            ub = zeros(length(μb))
            Sb = Diagonal([1 for i=1:length(ub)])    
            Σ2 = Diagonal([σ₁ for i=1:length(μb)])
            b = trace[(:b,output)]
            
            q_score = (
                log(pdf(MvNormal(u,S),μ)) + 
                log(pdf(MvNormal(μ,Σ),W)) + 
                log(pdf(MvNormal(ub,Sb),μb)) + 
                log(pdf(MvNormal(μb,Σ2),b))
                )
        else
            obs_new[(:k,i)] = trace[(:k,i)]
            obs_new[(:μ,i)] = trace[(:μ,i)]
            obs_new[(:μb,i)] = trace[(:μb,i)]
            obs_new[(:W,i)] = trace[(:W,i)]
            obs_new[(:b,i)] = trace[(:b,i)]
        end
    end
    #-----------------------------------------------------
    
    #Update Trace and Return Trace and Weights
    (new_trace,_,_,_) = update(trace, args, argdiffs, obs_new)
    #(new_trace,) = generate(interpolator, (x,), obs_new)
    q = q_score
        
    return (new_trace, q)
    
end