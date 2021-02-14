#Helper Functions
function plot_data_classifier(data,classes,alpha=1.0)
    markers = ["o","*"]
    colors = ["blue","green"]
    for i=1:2
        mask = [classes[j] == i for j in 1:length(classes)]
        scatter(data[:,1][mask],data[:,2][mask],c=colors[i],alpha=alpha,marker=markers[i],zorder=3)
    end
end;

function data_labeller(y::Array{Float64})
    labels = [y[i] > 0.5 ? 2 : 1 for i=1:length(y)]
    return labels
end

function mse_regression(y_pred::Array{Float64},y_real::Array{Float64})
    sum((y_pred .- y_real).^2)/length(y_real)
end

function mse_scaled(y_pred::Array{Float64},y_real::Array{Float64})
    y_pred = StatsBase.reconstruct(dy,y_pred)
    y_real = StatsBase.reconstruct(dy,y_real)
    √(sum((y_pred .- y_real).^2)/length(y_real))
end

function trace_acc(trace)
    pred_y = G(x, trace)
    pred_labels = data_labeller(pred_y)
    acc = sum([classes[i] == pred_labels[i] for i=1:length(classes)])
    return acc
end

mₖ(k) = k*4 + 1;

function layer_unpacker(x,i,l,k)
    d = length(x[:,1])
    if i == 1
        input_dim = d
        output_dim = k[i]
    else
        input_dim = k[i-1]
        output_dim = k[i]
    end
    return input_dim, output_dim
end

#Generate XOR Data
function real_data_classifier(N::Int, modes::Int, bound::Float64, σ::Float64)
    μ₁ = [-bound, -bound]
    μ₂ = [-bound, bound]
    μ₃ = [bound, bound]
    μ₄ = [bound, -bound]
    #μ₅ = [1.25, 1.25]
    #μ₆ = [1.25, 1.75]
    #μ₇ = [1.75, 1.75]
    #μ₈ = [1.75, 1.25]
    μ = [μ₁, μ₂, μ₃, μ₄]
    Σ = [[σ, 0] [0, σ]]
    
    all_samples = zeros(Float64, (N*modes, 2))
    classes = zeros(Int, (N*modes))
    
    for i = 1:modes
        dist = MvNormal(μ[i], Σ)
        sample = rand(dist, N)::Matrix
        #scatter(sample[1,:],sample[2,:])
        all_samples[(i-1)*N+1:i*N,:] = transpose(sample)
        classes[(i-1)*N+1:i*N] = fill(i, N)
        classes = float(classes)
    end
    return all_samples, classes
end;

#Generative Function for Assessing Likelihood
@gen function lh(x::Array{Float64}, trace)
    scores = G(x,trace)
    scores = Flux.σ.(scores)
    y = zeros(length(scores))
    for j=1:N
        y[j] = @trace(categorical([1-scores[j],scores[j]]), (:y,j))
    end

    return scores
end;

##################
#RJ Help Functions
##################

function select_hyperparameters(trace, obs)
    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    (new_trace,weight,retdiff) = regenerate(trace, args, argdiffs, select(:τ₁,:τ₂,:τ₃))
    obs[:τ₁] = new_trace[:τ₁]
    obs[:τ₂] = new_trace[:τ₂]
    obs[:τ₃] = new_trace[:τ₃]
    if network == "interpolator"
        obs[:τᵧ] = new_trace[:τᵧ]
    end
    return new_trace, obs
end

function select_selection(trace)
    l = trace[:l]
    selection = select()
    for i=1:l+1
        push!(selection, (:W,i))
        push!(selection, (:b,i))
    end
    return selection
end

function select_selection_fixed(trace)
    l = trace[:l]
    selection = select()
    for i=1:l+1
        push!(selection, (:W,i))
        push!(selection, (:b,i))
    end
    push!(selection, (:τ₁))
    push!(selection, (:τ₂))
    push!(selection, (:τ₃))
    return selection
end

##################
#Likelihood Tests#
##################

function likelihood_regression(iters)
    obs = obs_master;
    scores = []
    mses = []
    ls = []
    best_traces = []
    (best_trace,) = generate(interpolator, (x,), obs)
    best_score = get_score(best_trace)
    best_pred_y = transpose(G(x, best_trace))[:,1]
    best_mse = mse_scaled(best_pred_y, y)
    
    (trace,) = generate(interpolator, (x,), obs)
    score = get_score(trace)
    pred_y = transpose(G(x, trace))[:,1]
    mse = mse_scaled(pred_y, y)
    
    for i=1:iters
        (trace,) = generate(interpolator, (x,), obs)
        score = get_score(trace)
        pred_y = transpose(G(x, trace))[:,1]
        mse = mse_scaled(pred_y, y)
        push!(scores,score)
        push!(mses,mse)
        if mse < best_mse
            best_mse = mse
            best_score = score
            best_trace = trace
            best_pred_y = pred_y
        end
    end
    return(best_trace, scores, mses)
end;