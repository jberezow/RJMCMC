using Test
using TOML

@testset "Boston experiment configurations" begin
    directory = joinpath(@__DIR__, "..", "experiments", "boston")
    expected = Dict(
        "2-node-a" => (2, 8, 1.0, 1, 1, "mixed"),
        "2-node-b" => (2, 8, 0.8, 1, 1, "all"),
        "4-node-a" => (4, 4, 1.0, 1, 1, "mixed"),
        "4-node-b" => (4, 4, 0.8, 2, 2, "mixed"),
    )

    for (name, settings) in expected
        config = TOML.parsefile(joinpath(directory, name * ".toml"))
        width, depth, variance, samples, adaptation, update = settings
        @test config["hidden_width"] == width
        @test config["maximum_depth"] == depth
        @test config["likelihood_variance"] == variance
        @test config["nuts_samples"] == samples
        @test config["nuts_adaptation"] == adaptation
        @test config["within_dimension_update"] == update
        @test config["data_seed"] == 23
        @test config["iterations"] == 1000
        @test config["chains"] == 16
        @test config["initial_candidates"] == 1000
        @test config["target_acceptance"] == 0.65
        @test config["nuts_delta_max"] == 1000
        @test config["chains"] % config["maximum_depth"] == 0
    end
end
