function build_tree(trace,selection,θ,r,u,v,j,ϵ)
    #Base case - take one leapfrog step in the direction of v
    if j == 0
        θ¹,r¹ = leapfrog(trace,selection,θ,r,v*ϵ)
        args = get_args(trace)
        argdiffs = map((_) -> NoChange(), args)
        retval_grad = accepts_output_grad(get_gen_fn(trace)) ? zero(get_retval(trace)) : nothing
        (_, values_trie, _) = choice_gradients(trace, selection, retval_grad)
        θm = from_array(values_trie, θ¹)
        (tree_trace, _, _) = update(trace, args, argdiffs, θm)
        score = get_score(tree_trace) - 0.5(dot(r¹,r¹))
        if u ≤ exp(score)
            C¹ = Set(tuple([θ¹,r¹]))
        else
            C¹ = Set()
        end
        s¹ = (score > log(u) - Δmax) ? 1 : 0
        return θ¹,r¹,θ¹,r¹,C¹,s¹
    #Recursion - build left and right subtrees
    else
        θ⁻,r⁻,θ⁺,r⁺,C¹,s¹ = build_tree(trace,selection,θ,r,u,v,j-1,ϵ)
        if v == -1
            θ⁻,r⁻,_,_,C²,s² = build_tree(trace,selection,θ⁻,r⁻,u,v,j-1,ϵ)
        else
            _,_,θ⁺,r⁺,C²,s² = build_tree(trace,selection,θ⁺,r⁺,u,v,j-1,ϵ)
        end
        i¹ = (dot((θ⁺ - θ⁻),r⁻) ≥ 0) ? 1 : 0
        i² = (dot((θ⁺ - θ⁻),r⁺) ≥ 0) ? 1 : 0
        s¹ = s¹*s²*i¹*i²
        C¹ = union(C¹,C²)
       return θ⁻,r⁻,θ⁺,r⁺,C¹,s¹
    end
end;

function leapfrog(trace,selection,θ,r,ϵ)
    new_trace = trace
    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    retval_grad = accepts_output_grad(get_gen_fn(trace)) ? zero(get_retval(trace)) : nothing
    (_, values_trie, _) = choice_gradients(new_trace, selection, retval_grad)
    θtrace = from_array(values_trie, θ)
    (new_trace, _, _) = update(new_trace, args, argdiffs, θtrace)
    (_, values_trie, gradient_trie) = choice_gradients(new_trace, selection, retval_grad)
    gradient = to_array(gradient_trie, Float64)
    for step=1:1
        # half step on momenta
        r += (ϵ / 2) * gradient

        # full step on positions
        θ += ϵ .* r

        # get new gradient
        values_trie = from_array(values_trie, θ)
        (new_trace, _, _) = update(new_trace, args, argdiffs, values_trie)
        (_, _, gradient_trie) = choice_gradients(new_trace, selection, retval_grad)
        gradient = to_array(gradient_trie, Float64)

        # half step on momenta
        r += (ϵ / 2) * gradient
    end
    return θ,r
end

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

function NUTS(trace, selection::Selection, ϵ, check, observations, M)
    #Get θ₀
    Θ = []
    args = get_args(trace)
    retval_grad = accepts_output_grad(get_gen_fn(trace)) ? zero(get_retval(trace)) : nothing
    argdiffs = map((_) -> NoChange(), args)
    #selection = select_selection_NUTS(trace)
    (_, values_trie, gradient_trie) = choice_gradients(trace, selection, retval_grad)
    θ₀ = to_array(values_trie, Float64)
    push!(Θ, θ₀)
    C_choice = 0
    
    prev_model_score = get_score(trace)
    r₀ = sample_momenta(length(θ₀))
    prev_momenta_score = assess_momenta(r₀)
        
    #Loop M times
    for m=1:M
        #Resample Position Variables
        r = r₀
        θ = from_array(values_trie, θ₀)
        (new_trace, _, _) = update(trace, args, argdiffs, θ)
        score = exp(get_score(new_trace) - 0.5(dot(r,r)))
        if score <= 0
            u = 0
        else
            u = rand(Uniform(0,score))
        end
        
        #Initialize
        θ⁻ = Θ[m]
        θ⁺ = Θ[m]
        r⁻ = r₀
        r⁺ = r₀
        j = 0
        C = Set(tuple([Θ[m], r₀]))
        s = 1
        
        while s == 1
            vⱼ = rand([-1,1])
            if vⱼ == -1
                θ⁻,r⁻,_,_,C¹,s¹ = build_tree(trace,selection,θ⁻,r⁻,u,vⱼ,j,ϵ)
            else
                _,_,θ⁺,r⁺,C¹,s¹ = build_tree(trace,selection,θ⁺,r⁺,u,vⱼ,j,ϵ)
            end
            if s¹ == 1
                C = union(C,C¹)
            end
            i¹ = (dot((θ⁺ - θ⁻),r⁻) ≥ 0) ? 1 : 0
            i² = (dot((θ⁺ - θ⁻),r⁺) ≥ 0) ? 1 : 0
            s = s¹*i¹*i²
            j += 1
        end
        C_choice = rand(unique(C))
    end
    θ = from_array(values_trie, C_choice[1])
    print
    momenta = C_choice[2]
    (new_trace, _, _) = update(trace, args, argdiffs, θ)
    
    # assess new model score (negative potential energy)
    new_model_score = get_score(new_trace)

    # assess new momenta score (negative kinetic energy)
    new_momenta_score = assess_momenta(-momenta)

    # accept or reject
    alpha = new_model_score - prev_model_score + new_momenta_score - prev_momenta_score
    (new_trace, alpha)
end