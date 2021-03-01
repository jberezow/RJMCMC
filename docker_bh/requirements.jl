using Pkg

metadata_packages = [
    "Gen",
    "LinearAlgebra",
    "Random",
    "Distributions",
    "Distances",
    "Flux",
    "JLD",
    "Serialization",
    "StatsBase"]

#Pkg.update()

for package=metadata_packages
    Pkg.add(package)
end

app_dir = "/app"
push!(LOAD_PATH, app_dir)

using Gen
using LinearAlgebra
using Random
using Distributions
using Distances
using Flux
using JLD
using Serialization
using StatsBase
using BNN
