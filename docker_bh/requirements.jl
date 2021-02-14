using Pkg

metadata_packages = [
    "Gen",
    "LinearAlgebra",
    "Random",
    "Distributions",
    "Distances",
    "Flux",
    "JLD",
    "PyPlot",
    "Serialization",
    "StatsBase"]

Pkg.update()

for package=metadata_packages
    Pkg.add(package)
end