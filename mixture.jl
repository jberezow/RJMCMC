struct NeuralDist <: Gen.Distribution{Vector{Float64}} end

const neural_dist = NeuralDist()

function Gen.logpdf(
        dist::NeuralDist, x::Array{Float64,1},
        k::Int, G, params::Vector{Float64})
    mu1 = [0.25, 0.25]
    mu2 = [0.25, 0.75]
    means = [mu1, mu2]
    sig = 0.002
    sigma = [[sig, 0] [0, sig]]::Matrix
    dist1 = MvNormal(mu1, sigma)
    dist2 = MvNormal(mu2, sigma)
    score = 1
    
    for i in 1:length(x[1,:])
        dist1score = (pdf(dist1,data[:,i]))
        dist2score = (pdf(dist2,data[:,i]))
        update = log((dist1score + dist2score)/2)
        score += update
    end
    return score
end

function Gen.random(
        dist::NeuralDist,
        k::Int, G, params::Vector{Float64})
    weights = [1/k for i=1:k]
    cuts::Int = k * 2 - 1
    i = Gen.categorical(weights)
    a = (2*(i-1)) / cuts
    b = (2*(i-1) + 1) / cuts
    return G(Gen.random(uniform, a, b),params)
end

function Gen.logpdf_grad(
        dist::NeuralDist, x::Real,
        k::Int, G, params::Vector{Float64})
    return (nothing, nothing, nothing, nothing)
end

(dist::NeuralDist)(k, G, params) = Gen.random(dist, k, G, params)
Gen.is_discrete(dist::NeuralDist) = false
Gen.has_output_grad(dist::NeuralDist) = false
Gen.has_argument_grads(dist::NeuralDist) = (false, false, false)