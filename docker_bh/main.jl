#Library calls

current_dir = pwd()
app_dir = "/app"
#cd(app_dir)
push!(LOAD_PATH, app_dir)
push!(LOAD_PATH, current_dir)

#println(LOAD_PATH)
using Gen
using Distributions
using LinearAlgebra
using Flux
using Random
using Distances
using JLD
using Serialization
using StatsBase
using BNN

include("NUTS.jl")
include("RJNUTS.jl")
include("utils.jl")
include("rj_proposals_layers.jl")
include("BostonHousing.jl");

println("Packages Loaded")

#---------------
#Hyperparameters
#---------------

show(ARGS)

filename = "Run1.jld"
ITERS = parse(Int, ARGS[1])
CHAINS = 1

#NUTS
Δ_max = 1000
m = parse(Int, ARGS[2])

#Select Network Goal
network = "interpolator"

#Data hyperparameters
n = nrow #Number of samples per mode (classifier)
d = ncol-1 #Input dimension

#Network hyperparameters
k_real = parse(Int, ARGS[3]) #Number of hidden nodes per layer
k_vector = [0.0 for i=1:k_real]
k_vector[k_real] = 1.0

#Layer hyperparameters
l_range = parse(Int, ARGS[4]) #Maximum number of layers in the network
l_list = [Int(i) for i in 1:l_range]
l_real = 1

obs_master = choicemap()::ChoiceMap
obs_master[:y] = y
obs = obs_master;

#--------------
#Run Inference
#--------------
println("Beginning Inference")
println("-------------------")
(trace,) = generate(interpolator, (x,), obs)
traces, scores = RJNUTS(trace, ITERS)

cd(current_dir)
serialize(filename, traces)
