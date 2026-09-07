# BostonHousing/docker-parallel2a/BNN.jl at
# 3a2c64d0b8c0f1483f018ab6db0fcb2365a91021 (also the surviving 2B model).
# Width, maximum depth, and likelihood variance parameterize the 4A/4B variants.
module DepthBNN
using Gen
using Distributions
using LinearAlgebra
using Flux

export interpolator, G, layer_unpacker

#---------------
#I help the BNN
#---------------
function layer_unpacker(i,l,k)
    if i == 1
        input_dim = 13
        output_dim = k[i]
    else
        input_dim = k[i-1]
        output_dim = k[i]
    end
    return input_dim, output_dim
end

#-------------------
#Bayesian Neural Net
#-------------------
function G(x, trace)
    activation = relu
    l = trace[:l]
    ks = [trace[(:k,i)] for i=1:l]

    for i=1:l
        in_dim, out_dim = layer_unpacker(i, l, ks)
        W = reshape(trace[(:W,i)], out_dim, in_dim)
        b = reshape(trace[(:b,i)], trace[(:k,i)])
        nn = Dense(W, b, activation)
        x = nn(x)
    end

    Wₒ = reshape(trace[(:W,l+1)], 1, ks[l])
    bₒ = reshape(trace[(:b,l+1)], 1)

    nn_out = Dense(Wₒ, bₒ)
    return nn_out(x)

end;

#-------------------
#Probabilistic Model
#-------------------
"""
    interpolator(x, width=2, maximum_depth=8, likelihood_variance=1.0)

Boston Housing regression model with variable hidden-layer depth. Columns of
`x` are observations with 13 features. `likelihood_variance` is the diagonal
observation covariance; the historical `:τᵧ` draw remains in the trace but does
not control that covariance. Defaults match the surviving 2A model.
"""
@gen function interpolator(x, width=2, maximum_depth=8, likelihood_variance=1.0)
    size(x, 1) == 13 || throw(ArgumentError("Boston inputs must have 13 feature rows"))
    width isa Integer && width >= 1 || throw(ArgumentError("width must be a positive integer"))
    maximum_depth isa Integer && maximum_depth >= 1 ||
        throw(ArgumentError("maximum_depth must be a positive integer"))
    isfinite(likelihood_variance) && likelihood_variance > 0 ||
        throw(ArgumentError("likelihood_variance must be finite and positive"))

    #Node hyperparameters
    k_real = width #Number of hidden nodes per layer
    k_vector = [0.0 for i=1:k_real]
    k_vector[k_real] = 1.0

    #Layer hyperparameters
    l_range = maximum_depth #Maximum number of layers in the network
    l_list = [Int(i) for i in 1:l_range]

    #Hyperpriors
    αᵧ = 1 #Regression Noise Shape
    βᵧ = 1 #Regression Noise Scale/Rate
    α₁ = 1 #Input Weights, Biases Shape
    β₁ = 1 #Input Weights, Biases Scale/Rate
    α₂ = 1 #Hidden & Output Weights Shape
    β₂ = k_real; #Hidden & Output Weights Scale

    d = length(x[:,1])

    #Create a blank choicemap
    obs = choicemap()::ChoiceMap

    #Draw number of layers
    l ~ categorical([1/length(l_list) for i=1:length(l_list)])
    l_real = l
    obs[:l] = l

    #Create individual weight and bias vectors
    #Loop through hidden layers
    k = [Int(0) for i=1:l+1]
    for i=1:l
        k[i] = @trace(categorical(k_vector), (:k,i))
        obs[(:k,i)] = k[i]
    end
    k[l+1] = @trace(categorical([1.0]), (:k,l+1))
    obs[(:k,l+1)] = k[l+1]

    #####################################
    #New hyperparameter schedule - Mar 8#
    #####################################

    τ = [0.0 for i=1:l+1]
    τᵦ = [0.0 for i=1:l+1]
    σ = [1.0 for i=1:l+1]
    σᵦ = [1.0 for i=1:l+1]

    #for i=1:l+1
        #if i==1
            #τ[i] = @trace(gamma(α₁,β₁), (:τ,i))
            #τᵦ[i] = @trace(gamma(α₁,β₁), (:τᵦ,i))
        #else
            #τ[i] = @trace(gamma(α₂,β₂), (:τ,i))
            #τᵦ[i] = @trace(gamma(α₁,β₁), (:τᵦ,i))
        #end
        #σ[i] = 1/τ[i]
        #σᵦ[i] = 1/τᵦ[i]
    #end

    #Noise Variance
    τᵧ ~ gamma(αᵧ,βᵧ)
    σᵧ = likelihood_variance

    #Sample weight and bias vectors
    W = [zeros(k[i]) for i=1:l+1]
    b = [zeros(k[i]) for i=1:l+1]

    for i=1:l+1
        if i == 1
            h = Int(d * k[i])
        else
            h = Int(k[i-1] * k[i])
        end

        if i<=l
            #Hidden Weights
            u = zeros(h)
            S = Diagonal([σ[i] for j=1:length(u)])
            W[i] = @trace(mvnormal(u,S), (:W,i))
            obs[(:W,i)] = W[i]

            #Hidden Biases
            ub = zeros(k[i])
            Sb = Diagonal([σᵦ[i] for j=1:length(ub)])
            b[i] = @trace(mvnormal(ub,Sb), (:b,i))
            obs[(:b,i)] = b[i]
        else
            #Output Weights
            u = zeros(k[l])
            S = Diagonal([σ[i] for j=1:length(u)])
            W[i] = @trace(mvnormal(u,S), (:W,i))
            obs[(:W,i)] = W[i]

            #Output Bias
            ub = zeros(1)
            Sb = Diagonal([σᵦ[i] for j=1:length(ub)])
            b[i] = @trace(mvnormal(ub,Sb), (:b,i))
            obs[(:b,i)] = b[i]
        end
    end

    #Return Network Scores for X
    scores = transpose(G(x,obs))[:,1]

    #Regression Likelihood
    Sy = Diagonal([σᵧ for i=1:length(x[1,:])])
    y = @trace(mvnormal(vec(scores), Sy), (:y))

    return scores

end;

end;
