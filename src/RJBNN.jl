module RJBNN

using Gen
using Distributions
using Flux
using LinearAlgebra
using MultivariateStats
using Random
using Serialization
using StatsBase

include("models/depth_bnn.jl")
include("proposals/layer_birth_death.jl")

include("models/width_bnn.jl")
using .BNN: G, classifier, softmax_

# These names were ambient globals in the historical experiment scripts. They
# remain module state for the faithful baseline and will be replaced by explicit
# sampler state only after parity has been established.
Δ_max = 1
acc_prob = 0.65
m = 4
m2 = 1
k_list = collect(1:16)
n_classes = 2
softmax_scale = 0.5
y = Int[]
xt = zeros(2, 0)
obs_master = choicemap()

# The historical `utils.jl` supplied this three-argument adapter for proposal
# code, while the BNN module owned the input-dimension-aware implementation.
layer_unpacker(index, layers, widths) =
    BNN.layer_unpacker(index, layers, widths, size(xt, 1))

include("inference/nuts.jl")
include("inference/chains.jl")
include("data/xor.jl")
include("data/optdigits.jl")
include("proposals/node_birth_death.jl")
include("inference/width_rjnuts.jl")

export DepthBNN,
       LayerProposals,
       ExperimentData,
       ChainResult,
       XORData,
       XORResult,
       OptDigitsData,
       OptDigitsResult,
       generate_xor_data,
       prepare_xor!,
       run_xor,
       balanced_set,
       load_optdigits,
       prepare_optdigits!,
       run_optdigits,
       optdigits_directory,
       initial_trace,
       best_initial_trace,
       run_chain,
       save_result,
       load_result,
       predict_probabilities,
       classification_accuracy,
       posterior_accuracy,
       NUTS,
       RJNUTS,
       RJNUTS_parallel

end
