# Extracted from OptDigits/dockerxor/BNN.jl; the OptDigits variants at
# ff0b6e3f8ddb15326bd8ddd8f65aecf4b264954e supply the class, width, and scale settings.
module BNN
using Gen
using Distributions
using LinearAlgebra
using Flux

export classifier, G, layer_unpacker, softmax_

#---------------
#I help the BNN
#---------------
function layer_unpacker(i,l,k,d)
    if i == 1
        input_dim = d
        output_dim = k[i]
    else
        input_dim = k[i-1]
        output_dim = k[i]
    end
    return input_dim, output_dim
end

#New Softmax
function softmax_(arr::AbstractArray, scale=0.5)
    ex = mapslices(x -> exp.(scale*x),arr,dims=1)
    rows, cols = size(arr)
    val = similar(ex)
    for i in 1:cols
        s = sum(ex[:,i])
        for j in 1:rows
            val[j,i] = ex[j,i]/s
        end
    end
    return val
end;

#Bayesian Neural Net
# Old serialized XOR traces have only the input argument.
function G(x, trace::Gen.Trace)
    args = get_args(trace)
    scale = length(args) >= 4 ? args[4] : 0.5
    return G(x, get_choices(trace), scale)
end

function G(x, trace, scale=0.5)
    activation = tanh
    layers = trace[:l]
    ks = [trace[(:k,i)] for i=1:layers]
    
    c = trace[(:k,layers+1)]
    d = length(x[:,1])
    
    for i=1:layers
        in_dim, out_dim = layer_unpacker(i, layers, ks, d)
        W = reshape(trace[(:W,i)], out_dim, in_dim)
        b = reshape(trace[(:b,i)], trace[(:k,i)])
        nn = Dense(W, b, activation)
        x = nn(x)
    end
    
    Wₒ = reshape(trace[(:W,layers+1)], c, ks[layers])
    bₒ = reshape(trace[(:b,layers+1)], c)
    
    nn_out = Dense(Wₒ, bₒ)
    x = nn_out(x)
    
    return softmax_(x, scale)
end;

#-------------------
#Probabilistic Model
#-------------------
# Defaults reproduce XOR. Settings are trace arguments, not mutable model globals.
@gen function classifier(x, classes=2, maximum_width=16, softmax_scale=0.5)
    classes >= 2 || throw(ArgumentError("classes must be at least 2"))
    maximum_width >= 1 || throw(ArgumentError("maximum_width must be positive"))
    isfinite(softmax_scale) && softmax_scale > 0 ||
        throw(ArgumentError("softmax_scale must be finite and positive"))

    c = classes
    d = length(x[:,1])
    
    #Create a blank choicemap
    obs = choicemap()::ChoiceMap
    
    #Draw number of layers - 1 for Classifier Net
    l ~ categorical([1.0])
    l_real = l
    obs[:l] = l
    
    k_range = maximum_width #Maximum number of neurons per layer
    k_list = [Int(i) for i in 1:k_range]
    
    #Create individual weight and bias vectors
    #Loop through hidden layers
    k = [Int(0) for i=1:l+1]
    for i=1:l
        k[i] = @trace(categorical([1/length(k_list) for i=1:length(k_list)]), (:k,i))
        obs[(:k,i)] = k[i]
    end
    output_array = zeros(Float64, c)
    output_array[c] = 1.0

    k[l+1] = @trace(categorical(output_array), (:k,l+1))
    obs[(:k,l+1)] = k[l+1]
    
    ########################################
    #Fixed Hyperparameter schedule - Apr 26#
    ########################################
    
    σ = 1.0
    
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
            S = Diagonal([σ for j=1:length(u)])
            W[i] = @trace(mvnormal(u,S), (:W,i))
            obs[(:W,i)] = W[i]
            
            #Hidden Biases
            ub = zeros(k[i])
            Sb = Diagonal([σ for j=1:length(ub)])   
            b[i] = @trace(mvnormal(ub,Sb), (:b,i))
            obs[(:b,i)] = b[i]
        else
            #Output Weights
            u = zeros(h)
            S = Diagonal([σ for j=1:length(u)])
            W[i] = @trace(mvnormal(u,S), (:W,i))
            obs[(:W,i)] = W[i]

            #Output Bias
            ub = zeros(c)
            Sb = Diagonal([σ for j=1:length(ub)]) 
            b[i] = @trace(mvnormal(ub,Sb), (:b,i))
            obs[(:b,i)] = b[i]
        end
    end
    
    #Return Network Scores for X
    scores = G(x,obs,softmax_scale)
    
    #Logistic Classification Likelihood
    y = zeros(length(scores))
    for j=1:length(x[1,:])
        score_vec = scores[:,j]
        #println(score_vec)
        y[j] = @trace(categorical(score_vec), (:y,j))
    end
    
    return scores
    
end;

end;
