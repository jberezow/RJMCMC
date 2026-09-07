using Test
using RJBNN
using Gen
using Random
using StatsBase

@testset "Boston depth sampler" begin
    @testset "settings" begin
        for mode in (:mixed, :all)
            @test DepthRJNUTS.DepthSettings(0.65, 1, 1, mode).within_dimension_update == mode
        end
        @test_throws ArgumentError DepthRJNUTS.DepthSettings(0.65, 1, 1, :other)
    end

    data = load_boston()

    @testset "initialization" begin
        model = (width=2, maximum_depth=8, likelihood_variance=1.0)
        trace = initial_boston_trace(data, 3; model...)
        @test trace[:l] == 3
        @test trace[:y] == data.y_train
        @test get_args(trace)[2:4] == (2, 8, 1.0)
        @test isfinite(get_score(trace))
        @test length(boston_predictions(trace, data.x_train)) == 253

        @test_throws ArgumentError initial_boston_trace(data, 0; model...)
        @test_throws ArgumentError initial_boston_trace(data, 9; model...)

        # Selection minimizes scaled MSE, unlike the width path's posterior score.
        errors(trace) = scaled_mse(boston_predictions(trace, data.x_train),
                                   data.y_train, data.response_standardizer)
        Random.seed!(9)
        best = best_initial_boston_trace(data, 2; candidates=4, model...)
        Random.seed!(9)
        drawn = [initial_boston_trace(data, 2; model...) for _ = 1:6]
        # The historical routine draws one extra candidate it never compares.
        compared = errors.(drawn[[1; 3:6]])
        @test errors(best) == minimum(compared)
        @test best[:l] == 2
    end

    @testset "scaled_mse" begin
        standardizer = data.response_standardizer
        targets = data.y_train[1:4]
        @test scaled_mse(targets, targets, standardizer) == 0
        shifted = targets .+ 1
        expected = sqrt(sum((StatsBase.reconstruct(standardizer, collect(shifted)) .-
                             StatsBase.reconstruct(standardizer, collect(targets))) .^ 2)) / 4
        @test scaled_mse(shifted, targets, standardizer) ≈ expected
    end

    @testset "chain runs" begin
        for mode in (:mixed, :all)
            result = run_boston(iterations=1, candidates=2, chain=1,
                                within_dimension_update=mode)
            @test length(result.traces) == 1
            @test isfinite(only(result.scores))
            @test 1 <= only(result.depths) <= 8
            @test only(result.across_acceptance) in (0, 1)
            @test only(result.within_acceptance) in (0, 1)
            @test result.settings.within_dimension_update == mode
            @test isfinite(boston_rmse(only(result.traces), result.data.x_test,
                                       result.data.y_test, result.data.response_standardizer))
        end
    end

    @testset "chain initial depth schedule" begin
        # Chain i starts at ((i - 1) % maximum_depth) + 1.
        for (chain, maximum_depth, expected) in ((1, 8, 1), (9, 8, 1), (16, 8, 8),
                                                 (5, 4, 1), (16, 4, 4))
            # Zero iterations: this checks the schedule, not the sampler, and
            # initializing at depth 8 is expensive.
            result = run_boston(iterations=0, candidates=1, chain=chain,
                                maximum_depth=maximum_depth,
                                width=maximum_depth == 4 ? 4 : 2)
            @test result.settings.initial_depth == expected
        end
    end

    @testset "result round trip" begin
        result = run_boston(iterations=1, candidates=1)
        mktempdir() do directory
            path = joinpath(directory, "boston-result.jls")
            save_result(path, result)
            restored = load_result(path)
            @test restored isa BostonResult
            @test restored.scores == result.scores
            @test restored.depths == result.depths
            @test restored.settings == result.settings
        end
    end
end
