using Test
using RJBNN
using JLD
using Random
using Statistics
using StatsBase

@testset "Boston Housing data loader" begin
    data = load_boston()
    repeated = load_boston()

    @test size(data.x_train) == (13, 253)
    @test size(data.x_test) == (13, 253)
    @test length(data.y_train) == 253
    @test length(data.y_test) == 253
    @test data.row_order[1:10] == [169, 455, 227, 291, 10, 432, 88, 451, 271, 150]
    @test data.x_train == repeated.x_train
    @test data.x_test == repeated.x_test
    @test data.y_train == repeated.y_train
    @test data.y_test == repeated.y_test

    stored = JLD.load(joinpath(boston_directory(), "boston.jld"), "boston")
    @test size(stored) == (506, 14)
    historical = stored[shuffle(MersenneTwister(23), 1:506), :]
    historical_inputs = historical[:, 1:13]
    historical_responses = historical[:, 14]
    input_transform = fit(ZScoreTransform, historical_inputs, dims=1)
    response_transform = fit(ZScoreTransform, historical_responses, dims=1)
    StatsBase.transform!(input_transform, historical_inputs)
    StatsBase.transform!(response_transform, historical_responses)
    @test data.x_train == transpose(historical_inputs[1:253, :])
    @test data.x_test == transpose(historical_inputs[254:506, :])
    @test data.y_train == historical_responses[1:253]
    @test data.y_test == historical_responses[254:506]

    standardized_inputs = transpose(hcat(data.x_train, data.x_test))
    standardized_responses = vcat(data.y_train, data.y_test)
    @test vec(mean(standardized_inputs; dims=1)) ≈ zeros(13) atol=1e-12
    @test vec(std(standardized_inputs; dims=1)) ≈ ones(13) atol=1e-12
    @test mean(standardized_responses) ≈ 0.0 atol=1e-12
    @test std(standardized_responses) ≈ 1.0 atol=1e-12

    other_seed = load_boston(seed=3)
    @test other_seed.row_order != data.row_order
    @test sort(other_seed.row_order) == collect(1:506)

    mktempdir() do directory
        @test_throws ArgumentError load_boston(directory=directory)
        # A 490-row file is rejected by the size guard.
        JLD.save(joinpath(directory, "boston.jld"), "boston", zeros(490, 14))
        @test_throws ArgumentError load_boston(directory=directory)
    end
end
