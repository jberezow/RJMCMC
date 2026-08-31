module RJBNN

using Gen
using Distributions
using Flux
using LinearAlgebra
using Random

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
y = Int[]
xt = zeros(2, 0)
obs_master = choicemap()

# The historical `utils.jl` supplied this three-argument adapter for proposal
# code, while the BNN module owned the input-dimension-aware implementation.
layer_unpacker(index, layers, widths) =
    BNN.layer_unpacker(index, layers, widths, size(xt, 1))

include("inference/nuts.jl")
include("data/xor.jl")
include("proposals/node_birth_death.jl")
include("inference/width_rjnuts.jl")

export XORData,
       generate_xor_data,
       prepare_xor!,
       initial_xor_trace,
       run_xor,
       NUTS,
       RJNUTS,
       RJNUTS_parallel

end
