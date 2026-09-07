using Test
using RJBNN
using Gen
using Random
using Distributions
using LinearAlgebra

@testset "Boston depth model and layer proposals" begin
    x = reshape(collect(range(-0.5, 0.5; length=39)), 13, 3)
    observations = [0.2, -0.3, 1.0]
    for (width, maximum_depth, variance) in ((2, 8, 1.0), (2, 8, 0.8),
                                           (4, 4, 1.0), (4, 4, 0.8))
        args = (x, width, maximum_depth, variance)
        Random.seed!(17)
        trace, _ = generate(DepthBNN.interpolator, args,
            choicemap((:l, 2), (:y, observations), (:τᵧ, 1.5)))
        @test isfinite(get_score(trace))
        @test trace[:y] == observations
        @test trace[(:k, 1)] == trace[(:k, 2)] == width
        @test trace[(:k, 3)] == 1
        @test length(trace[(:W, 1)]) == 13 * width
        @test length(trace[(:W, 2)]) == width^2
        @test length(trace[(:W, 3)]) == width
        @test get_retval(trace) == vec(DepthBNN.G(x, trace))

        deeper, _ = generate(DepthBNN.interpolator,
            (x, width, 2 * maximum_depth, variance), get_choices(trace))
        @test get_score(deeper) - get_score(trace) ≈ -log(2)
        other_variance = 0.6
        changed, _ = generate(DepthBNN.interpolator,
            (x, width, maximum_depth, other_variance), get_choices(trace))
        likelihood(v) = Distributions.logpdf(MvNormal(get_retval(trace),
            Diagonal(fill(v, length(observations)))), observations)
        @test get_score(changed) - get_score(trace) ≈
            likelihood(other_variance) - likelihood(variance)
        @test get_retval(changed) == get_retval(trace)

        # The auxiliary Gamma draw remains a choice, but is not the noise variance.
        changed_tau, _, _, _ = update(trace, args, map(_ -> NoChange(), args),
            choicemap((:τᵧ, 2.0)))
        @test get_retval(changed_tau) == get_retval(trace)
        @test get_score(changed_tau) - get_score(trace) ≈ -0.5

        born, q_birth = LayerProposals.layer_birth(trace)
        restored, q_death = LayerProposals.layer_death(born)
        @test born[:l] == 3
        @test restored[:l] == 2
        @test get_args(born) == get_args(restored) == args
        @test born[:τᵧ] == restored[:τᵧ] == trace[:τᵧ]
        @test born[:y] == restored[:y] == observations
        @test born[(:W, 4)] == trace[(:W, 3)]
        @test born[(:b, 4)] == trace[(:b, 3)]
        @test q_birth ≈ -q_death
        @test get_score(restored) == get_score(trace)
        for layer in 1:3, parameter in (:k, :W, :b)
            @test restored[(parameter, layer)] == trace[(parameter, layer)]
        end
        for depth in (1, maximum_depth)
            boundary, _ = generate(DepthBNN.interpolator, args,
                choicemap((:l, depth), (:y, observations)))
            moved, _ = LayerProposals.layer_change(boundary)
            @test moved[:l] == (depth == 1 ? 2 : maximum_depth - 1)
            @test isfinite(get_score(moved))
        end
        _, _, gradients = choice_gradients(trace,
            select((:W, 1), (:b, 1), (:W, 2), (:b, 2), (:W, 3), (:b, 3)), nothing)
        @test all(isfinite, to_array(gradients, Float64))
    end
    @test_throws ArgumentError generate(DepthBNN.interpolator, (zeros(12, 3),))
    @test_throws ArgumentError generate(DepthBNN.interpolator, (x, 0, 8, 1.0))
    @test_throws ArgumentError generate(DepthBNN.interpolator, (x, 2, 0, 1.0))
    @test_throws ArgumentError generate(DepthBNN.interpolator, (x, 2, 8, 0.0))
    fixed, _ = generate(DepthBNN.interpolator, (x, 2, 1, 1.0))
    @test_throws ArgumentError LayerProposals.layer_change(fixed)
    @test_throws ArgumentError LayerProposals.layer_birth(fixed)
    @test_throws ArgumentError LayerProposals.layer_death(fixed)
end
