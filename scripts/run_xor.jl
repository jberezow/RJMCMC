using RJBNN
using TOML

function argument(name, default, convert=identity)
    prefix = "--$(name)="
    match = findfirst(value -> startswith(value, prefix), ARGS)
    match === nothing && return default
    return convert(split(ARGS[match], "="; limit=2)[2])
end

default_config = normpath(joinpath(@__DIR__, "..", "experiments", "xor", "low-noise.toml"))
config_path = argument("config", default_config)
config = TOML.parsefile(config_path)

iterations = argument("iterations", Int(config["iterations"]), value -> parse(Int, value))
samples_per_mode = argument(
    "samples-per-mode",
    Int(config["samples_per_mode"]),
    value -> parse(Int, value),
)
variance = argument("variance", Float64(config["mode_variance"]), value -> parse(Float64, value))
seed = argument("seed", 1, value -> parse(Int, value))
initial_width = argument("initial-width", 1, value -> parse(Int, value))

result = run_xor(
    iterations=iterations,
    samples_per_mode=samples_per_mode,
    variance=variance,
    seed=seed,
    initial_width=initial_width,
    maximum_width=Int(config["maximum_width"]),
    target_acceptance=Float64(config["target_acceptance"]),
    nuts_samples=Int(config["nuts_samples"]),
    nuts_adaptation=Int(config["nuts_adaptation"]),
)

println("Configuration: $(config["name"]) ($config_path)")
println("Completed $(length(result.traces)) XOR iterations")
println("Final log score: $(last(result.scores))")
println("Final hidden width: $(last(result.traces)[(:k, 1)])")
