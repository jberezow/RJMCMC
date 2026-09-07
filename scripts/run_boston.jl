using RJBNN
using TOML

function argument(name, default, convert=identity)
    prefix = "--$(name)="
    match = findfirst(value -> startswith(value, prefix), ARGS)
    match === nothing && return default
    return convert(split(ARGS[match], "="; limit=2)[2])
end

default_config = normpath(joinpath(@__DIR__, "..", "experiments", "boston", "2-node-a.toml"))
config_path = argument("config", default_config)
config = TOML.parsefile(config_path)

iterations = argument("iterations", Int(config["iterations"]), value -> parse(Int, value))
chain = argument("chain", 1, value -> parse(Int, value))
seed = argument("seed", 1, value -> parse(Int, value))
candidates = argument("candidates", Int(config["initial_candidates"]), value -> parse(Int, value))
initial_depth = argument("initial-depth", nothing, value -> parse(Int, value))
data_directory = argument("data", boston_directory())
slug = replace(lowercase(config["name"]), r"[^a-z0-9]+" => "-")
default_output = normpath(joinpath(
    @__DIR__, "..", "results", "boston", "$slug-seed-$seed-chain-$chain.jls",
))
output_path = argument("output", default_output)

result = run_boston(
    iterations=iterations,
    chain=chain,
    width=Int(config["hidden_width"]),
    maximum_depth=Int(config["maximum_depth"]),
    likelihood_variance=Float64(config["likelihood_variance"]),
    data_seed=Int(config["data_seed"]),
    seed=seed,
    candidates=candidates,
    initial_depth=initial_depth,
    target_acceptance=Float64(config["target_acceptance"]),
    nuts_samples=Int(config["nuts_samples"]),
    nuts_adaptation=Int(config["nuts_adaptation"]),
    divergence_threshold=Float64(config["nuts_delta_max"]),
    within_dimension_update=Symbol(config["within_dimension_update"]),
    directory=data_directory,
)
save_result(output_path, result)

final = last(result.traces)
standardizer = result.data.response_standardizer
println("Configuration: $(config["name"]) ($config_path)")
println("Completed $(length(result.traces)) Boston iterations on chain $chain")
println("Initial depth: $(result.settings.initial_depth)")
println("Final log score: $(last(result.scores))")
println("Final depth: $(last(result.depths))")
println("Across-dimension acceptance: $(100 * sum(result.across_acceptance) / length(result.across_acceptance))%")
println("Within-dimension acceptance: $(100 * sum(result.within_acceptance) / length(result.within_acceptance))%")
println("Train RMSE: $(boston_rmse(final, result.data.x_train, result.data.y_train, standardizer))")
println("Test RMSE: $(boston_rmse(final, result.data.x_test, result.data.y_test, standardizer))")
println("Saved result: $output_path")
