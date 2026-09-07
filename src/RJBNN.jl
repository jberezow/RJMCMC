module RJBNN

using Gen
using Distributions
using Flux
using JLD
using LinearAlgebra
using MultivariateStats
using Random
using Serialization
using StatsBase

include("models/depth_bnn.jl")
include("proposals/layer_birth_death.jl")

include("models/width_bnn.jl")
using .BNN: G, classifier, softmax_

# Ambient state read by the width model, proposals and RJNUTS driver.
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

# Three-argument form used by the width proposals.
layer_unpacker(index, layers, widths) =
    BNN.layer_unpacker(index, layers, widths, size(xt, 1))

include("inference/nuts.jl")
include("inference/depth_rjnuts.jl")
include("inference/chains.jl")
include("data/xor.jl")
include("data/optdigits.jl")
include("data/boston.jl")
include("proposals/node_birth_death.jl")
include("inference/width_rjnuts.jl")

export DepthBNN,
       LayerProposals,
       DepthRJNUTS,
       ExperimentData,
       ChainResult,
       XORData,
       XORResult,
       OptDigitsData,
       OptDigitsResult,
       BostonData,
       generate_xor_data,
       prepare_xor!,
       run_xor,
       balanced_set,
       load_optdigits,
       prepare_optdigits!,
       run_optdigits,
       optdigits_directory,
       load_boston,
       boston_directory,
       BostonResult,
       run_boston,
       run_depth_chain,
       initial_boston_trace,
       best_initial_boston_trace,
       boston_predictions,
       boston_rmse,
       scaled_mse,
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
