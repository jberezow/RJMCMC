using Test
using RJBNN

@testset "XOR data" begin
    first = generate_xor_data(samples_per_mode=3, seed=7)
    second = generate_xor_data(samples_per_mode=3, seed=7)

    @test size(first.x_train) == (12, 2)
    @test size(first.x_test) == (12, 2)
    @test first.x_train == second.x_train
    @test first.x_test == second.x_test
    @test Set(first.y_train) == Set([1, 2])
end

@testset "XOR model initialization" begin
    data = generate_xor_data(samples_per_mode=2, seed=11)
    prepare_xor!(data)
    trace = initial_xor_trace(1)

    @test trace[:l] == 1
    @test trace[(:k, 1)] == 1
    @test trace[(:k, 2)] == 2
end

@testset "XOR RJNUTS smoke run" begin
    result = run_xor(iterations=1, samples_per_mode=2, seed=11)

    @test length(result.traces) == 1
    @test length(result.scores) == 1
    @test isfinite(only(result.scores))
    @test 1 <= only(result.traces)[(:k, 1)] <= 16
    @test length(result.across_acceptance) == 1
    @test length(result.within_acceptance) == 1
    @test 0 <= classification_accuracy(
        only(result.traces),
        result.data.x_test,
        result.data.y_test,
    ) <= 1

    mktempdir() do directory
        path = joinpath(directory, "xor-result.jls")
        save_xor_result(path, result)
        restored = load_xor_result(path)
        @test restored.scores == result.scores
        @test restored.widths == result.widths
        @test restored.settings == result.settings
    end
end

using Gen
using Random
using TOML

@testset "OptDigits width models" begin
    x = reshape(collect(range(-0.5, 0.5; length=40)), 20, 2)
    config_dir = joinpath(@__DIR__, "..", "experiments", "optdigits")
    for name in ("5-class-a", "5-class-b", "10-class-a", "10-class-b")
        config = TOML.parsefile(joinpath(config_dir, name * ".toml"))
        classes = config["classes"]
        maximum_width = config["maximum_width"]
        scale = config["softmax_scale"]
        args = (x, classes, maximum_width, scale)
        observations = choicemap(((:k, 1), 2), ((:y, 1), 1), ((:y, 2), classes))
        Random.seed!(21)
        trace, _ = generate(RJBNN.classifier, args, observations)
        @test trace[(:k, 2)] == classes
        @test length(trace[(:W, 1)]) == 40
        @test length(trace[(:W, 2)]) == 2 * classes
        @test isfinite(get_score(trace))
        probabilities = predict_probabilities(trace, transpose(x))
        @test size(probabilities) == (classes, 2)
        @test probabilities == get_retval(trace)
        @test vec(sum(probabilities; dims=1)) ≈ ones(2)

        # With every choice fixed, changing only the uniform width prior changes
        # the score by its known normalization constant.
        wider, _ = generate(RJBNN.classifier,
            (x, classes, 2 * maximum_width, scale), get_choices(trace))
        @test get_score(wider) - get_score(trace) ≈ -log(2)

        # The historical proposal helpers still use ambient data and width bounds.
        @eval RJBNN begin
            xt = $x
            y = $([1, classes])
            k_list = collect(1:$maximum_width)
        end
        born, _ = RJBNN.node_birth(trace)
        restored, _ = RJBNN.node_death(born)
        @test born[(:k, 1)] == 3
        @test restored[(:k, 1)] == 2
        for moved in (born, restored)
            @test get_args(moved) == args
            @test moved[(:k, 2)] == classes
            @test isfinite(get_score(moved))
        end
        _, _, gradients = choice_gradients(trace, select((:W, 1), (:W, 2)), nothing)
        @test all(isfinite, to_array(gradients, Float64))
    end
    prepare_xor!(generate_xor_data(samples_per_mode=2))
end

@testset "XOR width bound" begin
    data = generate_xor_data(samples_per_mode=2)
    prepare_xor!(data; maximum_width=32)
    trace = initial_xor_trace(32)
    @test isfinite(get_score(trace))
    @test trace[(:k, 1)] == 32
    @test_throws ArgumentError prepare_xor!(data; maximum_width=1)
    prepare_xor!(data)
end
