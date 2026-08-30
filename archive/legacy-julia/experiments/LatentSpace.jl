using Random

function latent_sampler(samples::Int64, classes::Int64, dimension::Int64)
    return randn(Float64,(samples,dimension))
end

z = latent_sampler(10, 4, 2)
print(z)