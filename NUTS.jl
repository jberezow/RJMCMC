function sample_momenta(n::Int)
    Float64[random(normal, 0, 1) for _=1:n]
end

function assess_momenta(momenta)
    logprob = 0.
    for val in momenta
        logprob += Gen.logpdf(normal, val, 0, 1)
    end
    logprob
end

function select_selection_NUTS(trace)
    l = 1
    selection = select()
    for i=1:l+1
        push!(selection, (:W,i))
        push!(selection, (:b,i))
    end
    return selection
end

function find_reasonable_epsilon(trace,selection,vals,θ)
    ϵ = 1 #initialize ϵ
    r = sample_momenta(length(θ))
    θ¹,r¹,new_trace = leapfrog(trace,selection,vals,θ,r,ϵ)
    score_comparison = exp(get_score(new_trace) - get_score(trace))
    score_indicator = (score_comparison > 0.5) ? 1.0 : 0.0
    a = 2.0*(score_indicator) - 1.0
    while score_comparison^a > 2^(-a)
        ϵ = (2^a)*ϵ
        θ¹,r¹,new_trace2 = leapfrog(new_trace,selection,vals,θ,r,ϵ)
        score_comparison = exp(get_score(new_trace2) - get_score(new_trace))
    end
    return ϵ
end

function L(trace,θ,selection)
    new_trace = trace
    args = get_args(new_trace)
    argdiffs = map((_) -> NoChange(), args)
    retval_grad = accepts_output_grad(get_gen_fn(trace)) ? zero(get_retval(trace)) : nothing
    (_, vals, gradient_trie) = choice_gradients(trace, selection, retval_grad)
    θ_update = from_array(vals, θ)
    (new_trace, _, _) = update(new_trace, args, argdiffs, θ_update)
    get_score(new_trace)
end

#Tree Building for Efficient NUTS
function build_tree(trace,selection,vals,θ,r,u,v,j,ϵ,θ⁰,r⁰)
    #Base case - take one leapfrog step in the direction of v
    if j == 0
        θ¹,r¹,tree_trace = leapfrog(trace,selection,vals,θ,r,v*ϵ)
        prev_score = L(trace,θ⁰,selection) - 0.5(dot(r⁰,r⁰))
        score = get_score(tree_trace) - 0.5(dot(r¹,r¹))

        n¹ = (log(u) ≤ score) ? 1 : 0
        s¹ = (log(u) < Δmax + score) ? 1 : 0
        α¹ = min(1, exp(score-prev_score))
        return θ¹,r¹,θ¹,r¹,θ¹,n¹,s¹,α¹,1
    #Recursion - build left and right subtrees
    else
        θ⁻,r⁻,θ⁺,r⁺,θ¹,n¹,s¹,α¹,nᵅ¹ = build_tree(trace,selection,vals,θ,r,u,v,j-1,ϵ,θ⁰,r⁰)
        if s¹ == 1
            if v == -1
                θ⁻,r⁻,_,_,θ²,n²,s²,α²,nᵅ² = build_tree(trace,selection,vals,θ⁻,r⁻,u,v,j-1,ϵ,θ⁰,r⁰)
            else
                _,_,θ⁺,r⁺,θ²,n²,s²,α²,nᵅ² = build_tree(trace,selection,vals,θ⁺,r⁺,u,v,j-1,ϵ,θ⁰,r⁰)
            end
            met_ind = n²/(n¹+n²)
            #println("Acceptance prob: $met_ind")
            if bernoulli(met_ind) == true
                θ¹ = θ²
            else
                θ¹ = θ¹
            end
            α¹ = α¹ + α²
            nᵅ¹ = nᵅ¹ + nᵅ²
            i¹ = (dot((θ⁺ - θ⁻),r⁻) ≥ 0) ? 1 : 0
            i² = (dot((θ⁺ - θ⁻),r⁺) ≥ 0) ? 1 : 0
            s¹ = s²*i¹*i²
            n¹ = n¹ + n²
        end
       return θ⁻,r⁻,θ⁺,r⁺,θ¹,n¹,s¹,α¹,nᵅ¹
    end
end;

function leapfrog(trace,selection,vals,θ,r,ϵ)
    #Prep trace and gradient
    new_trace = trace
    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    retval_grad = accepts_output_grad(get_gen_fn(trace)) ? zero(get_retval(trace)) : nothing
    θtrace = from_array(vals, θ)
    (new_trace, _, _) = update(new_trace, args, argdiffs, θtrace)
    (_, values_trie, gradient_trie) = choice_gradients(new_trace, selection, retval_grad)
    gradient = to_array(gradient_trie, Float64)
    
    #LEAPFROG
    
    r += (ϵ / 2) * gradient # half step on momenta
    θ += ϵ .* r # full step on positions

    # get new gradient
    θnew = from_array(values_trie, θ)
    (new_trace, _, _) = update(new_trace, args, argdiffs, θnew)
    (_, _, gradient_trie) = choice_gradients(new_trace, selection, retval_grad)
    gradient = to_array(gradient_trie, Float64)
    
    r += (ϵ / 2) * gradient # half step on momenta

    return θ,r,new_trace
end

#Dual averaging constants for finding good ϵ
γ = 0.05
t₀ = 10
κ = 0.75
δ = 0.65

function NUTS(trace, selection::Selection, check, observations, M, Madapt, prev_trace)
    #Get vals structure
    args = get_args(trace)
    retval_grad = accepts_output_grad(get_gen_fn(trace)) ? zero(get_retval(trace)) : nothing
    argdiffs = map((_) -> NoChange(), args)
    (_, vals, gradient_trie) = choice_gradients(trace, selection, retval_grad)
    
    #Initialize θ, r
    θ⁰ = to_array(vals, Float64)
    r⁰ = sample_momenta(length(θ⁰))
    θ = θ⁰
    θᵐ = θ
    r = r⁰
    new_trace = trace
    
    #Previous scores
    prev_model_score = get_score(prev_trace)
    prev_momenta_score = assess_momenta(r⁰)
        
    #Initialize ϵ
    ϵ⁰ = find_reasonable_epsilon(trace,selection,vals,θ⁰)
    μ = log(10*ϵ⁰);ϵ¹=1;H⁰=0;γ=0.05;t⁰=10;κ=0.75;δ=0.65
    #println("Epsilon: $ϵ")
        
    #Loop M times
    for m=1:M
        #Resample momentum Variables
        println("Epsilon for iter $m: $ϵ⁰")
        m == 1 ? (r = r⁰) : (r = sample_momenta(length(θ)))
        params = from_array(vals, θ)
        (new_trace, _, _) = update(new_trace, args, argdiffs, params)
        score = exp(get_score(new_trace) - 0.5(dot(r,r)))
        
        if score <= 0
            u = 0
        else
            u = rand(Uniform(0,score))
        end
        
        #Initialize
        θ¹ = θ; θ⁻ = θ; θ⁺ = θ; r⁻ = r; r⁺ = r; j = 0; s = 1; n = 1; nᵅ = 1
        
        while s == 1
            vⱼ = rand([-1,1])
            if vⱼ == -1
                θ⁻,r⁻,_,_,θ¹,n¹,s¹,α,nᵅ = build_tree(new_trace,selection,vals,θ⁻,r⁻,u,vⱼ,j,ϵ⁰,θ,r)
            else
                _,_,θ⁺,r⁺,θ¹,n¹,s¹,α,nᵅ = build_tree(new_trace,selection,vals,θ⁺,r⁺,u,vⱼ,j,ϵ⁰,θ,r)
            end
            if s¹ == 1
                met_ind = (n¹/n > 1.0) ? 1.0 : n¹/n
                if bernoulli(met_ind) == true
                    θᵐ = θ¹
                else
                    θᵐ = θᵐ
                end
            end
            n = n + n¹
            i¹ = (dot((θ⁺ - θ⁻),r⁻) ≥ 0) ? 1 : 0
            i² = (dot((θ⁺ - θ⁻),r⁺) ≥ 0) ? 1 : 0
            s = s¹*i¹*i²
            j += 1
        end
            
        if m ≤ Madapt
            H⁰ = ((1-(1/(m+t₀)))*H⁰)+((1/(m+t₀))*(δ-(α/nᵅ)))
            ϵ⁰ = exp(μ - (√m)/γ * H⁰)
            ϵ¹ = exp(m^(-κ)*log(ϵ⁰) + (1-m^(-κ))*log(ϵ¹))
            #ϵ⁰ = ϵ⁰
        else
            ϵ⁰ = ϵ⁰
        end
        
            
    end
    
    θ = from_array(vals, θᵐ)
    momenta = r
    (new_trace, _, _) = update(new_trace, args, argdiffs, θ)
    
    new_model_score = get_score(new_trace) # assess new model score (negative potential energy)
    new_momenta_score = assess_momenta(-momenta) # assess new momenta score (negative kinetic energy)

    # accept or reject
    alpha = new_model_score - prev_model_score + new_momenta_score - prev_momenta_score
    (new_trace, alpha)
end