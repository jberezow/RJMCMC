using RJBNN
using TOML

function argument(name, default, convert=identity)
    prefix = "--$(name)="
    match = findfirst(value -> startswith(value, prefix), ARGS)
    match === nothing && return default
    return convert(split(ARGS[match], "="; limit=2)[2])
end

default_config = normpath(joinpath(@__DIR__, "..", "experiments", "optdigits", "5-class-a.toml"))
config_path = argument("config", default_config)
config = TOML.parsefile(config_path)

iterations = argument("iterations", Int(config["iterations"]), value -> parse(Int, value))
seed = argument("seed", 1, value -> parse(Int, value))
chain = argument("chain", 1, value -> parse(Int, value))
initial_width = argument(
    "initial-width",
    Int(config["initial_width_stride"]) * chain,
    value -> parse(Int, value),
)
candidates = argument("candidates", Int(config["initial_candidates"]), value -> parse(Int, value))
data_directory = argument("data", optdigits_directory())
slug = replace(lowercase(config["name"]), r"[^a-z0-9]+" => "-")
default_output = normpath(joinpath(
    @__DIR__,
    "..",
    "results",
    "optdigits",
    "$slug-seed-$seed-chain-$chain.jls",
))
output_path = argument("output", default_output)

result = run_optdigits(
    classes=Int(config["classes"]),
    iterations=iterations,
    samples_per_class=Int(config["samples_per_class"]),
    test_samples_per_class=Int(config["test_samples_per_class"]),
    training_seed=Int(config["training_seed"]),
    test_seed=Int(config["test_seed"]),
    pca_dimensions=Int(config["pca_dimensions"]),
    seed=seed,
    initial_width=initial_width,
    candidates=candidates,
    maximum_width=Int(config["maximum_width"]),
    softmax_scale=Float64(config["softmax_scale"]),
    target_acceptance=Float64(config["target_acceptance"]),
    nuts_samples=Int(config["nuts_samples"]),
    nuts_adaptation=Int(config["nuts_adaptation"]),
    divergence_threshold=Float64(config["nuts_delta_max"]),
    chain=chain,
    directory=data_directory,
)
save_result(output_path, result)

println("Configuration: $(config["name"]) ($config_path)")
println("Completed $(length(result.traces)) OptDigits iterations on chain $chain")
println("Final log score: $(last(result.scores))")
println("Final hidden width: $(last(result.widths))")
println("Across-dimension acceptance: $(100 * sum(result.across_acceptance) / length(result.across_acceptance))%")
println("Within-dimension acceptance: $(100 * sum(result.within_acceptance) / length(result.within_acceptance))%")
println("Training accuracy: $(classification_accuracy(last(result.traces), result.data.x_train, result.data.y_train))")
println("Test accuracy: $(classification_accuracy(last(result.traces), result.data.x_test, result.data.y_test))")
println("Saved result: $output_path")
