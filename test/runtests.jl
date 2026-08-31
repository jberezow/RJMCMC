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
end
