# An Julia Implementation of No-U-Turn Sampler with Dual Averaging described in Algorithm 6 in Hoffman et al. (2011)
# Author: Kai Xu
# Date: 06/10/2016

function NUTS(trace, selection, δ, M, Madapt, verbose=true)

  args = get_args(trace)
  retval_grad = accepts_output_grad(get_gen_fn(trace)) ? zero(get_retval(trace)) : nothing
  argdiffs = map((_) -> NoChange(), args)
  (_, vals, gradient_trie) = choice_gradients(trace, selection, retval_grad)
  θ0 = to_array(vals, Float64)
    
  function L(θ)
    θtrace = from_array(vals, θ)
    (new_trace, _, _) = update(trace, args, argdiffs, θtrace)
    score = get_score(new_trace)
    return score
  end
    
  function ∇L(θ)
    θtrace = from_array(vals, θ)
    (new_trace, _, _) = update(trace, args, argdiffs, θtrace)
    (_, values_trie, gradient_trie) = choice_gradients(new_trace, selection, retval_grad)
    gradient = to_array(gradient_trie, Float64)
    return gradient
  end 

  function leapfrog(θ, r, ϵ)

    r̃ = r + 0.5 * ϵ * ∇L(θ)
    θ̃ = θ + ϵ * r̃
    r̃ = r̃ + 0.5 * ϵ * ∇L(θ̃)
    return θ̃, r̃
  end

  function find_reasonable_ϵ(θ)
    ϵ, r = 1, Float64[random(normal, 0, 1.0) for _=1:length(θ)]#randn(length(θ))
    θ′, r′ = leapfrog(θ, r, ϵ)

    # This trick prevents the log-joint or its graident from being infinte
    # Ref: code start from Line 111 in https://github.com/mfouesneau/NUTS/blob/master/nuts.py
    # QUES: will this lead to some bias of the sampler?
    while isinf(L(θ′)) || any(isinf.(∇L(θ′)))
      println("Inf warning")
      ϵ = ϵ * 0.5
      θ′, r′ = leapfrog(θ, r, ϵ)
    end

    a = 2 * (exp(L(θ′) - 0.5 * dot(r′, r′)) / exp(L(θ) - 0.5 * dot(r, r)) > 0.5) - 1
    while (exp(L(θ′) - 0.5 * dot(r′, r′)) / exp(L(θ) - 0.5 * dot(r, r)))^float(a) > 2^float(-a)
      ϵ = 2^float(a) * ϵ
      θ′, r′ = leapfrog(θ, r, ϵ)
    end
    return ϵ
  end

  function build_tree(θ, r, u, v, j, ϵ, θ0, r0)

    if j == 0
      # Base case - take one leapfrog step in the direction v.
      θ′, r′ = leapfrog(θ, r, v * ϵ)
      # NOTE: this trick prevents the log-joint or its graident from being infinte
      while L(θ′) == -Inf || ∇L(θ′) == -Inf
        ϵ = ϵ * 0.5
        θ′, r′ = leapfrog(θ, r, v * ϵ)
        println("Inf warning")
      end
      n′ = u <= exp(L(θ′) - 0.5 * dot(r′, r′))
      s′ = u < exp(Δ_max + L(θ′) - 0.5 * dot(r′, r′))
      return θ′, r′, θ′, r′, θ′, n′, s′, min(1, exp(L(θ′) - 0.5 * dot(r′, r′) - L(θ0) + 0.5 * dot(r0, r0))), 1
    else
      # Recursion - build the left and right subtrees.
      θm, rm, θp, rp, θ′, n′, s′, α′, n′_α = build_tree(θ, r, u, v, j - 1, ϵ, θ0, r0)
      if s′ == 1
        if v == -1
          θm, rm, _, _, θ′′, n′′, s′′, α′′, n′′_α = build_tree(θm, rm, u, v, j - 1, ϵ, θ0, r0)
        else
          _, _, θp, rp, θ′′, n′′, s′′, α′′, n′′_α = build_tree(θp, rp, u, v, j - 1, ϵ, θ0, r0)
        end
        if rand() < n′′ / (n′ + n′′)
          θ′ = θ′′
        end
        α′ = α′ + α′′
        n′_α = n′_α + n′′_α
        s′ = s′′ & (dot(θp - θm, rm) >= 0) & (dot(θp - θm, rp) >= 0)
        n′ = n′ + n′′
      end
      return θm, rm, θp, rp, θ′, n′, s′, α′, n′_α
    end
  end

  #∇L = θ -> ForwardDiff.gradient(L, θ)  # generate gradient function

  θs = [zeros(length(θ0)) for i=1:M+1]

  θs[1] = θ0
  ϵ = find_reasonable_ϵ(θ0)
  μ, γ, t_0, κ = log(10 * ϵ), 0.05, 10, 0.75
  ϵ̄, H̄ = 1, 0

  if verbose println("[NUTS] start sampling for $M samples with inital ϵ=$ϵ") end

  for m = 1:M
    if verbose print('.') end
    r0 = Float64[random(normal, 0, 1.0) for _=1:length(θ0)]
    #r0 = randn(length(θ0))
    u = rand() * exp(L(θs[m]) - 0.5 * dot(r0, r0)) # Note: θ^{m-1} in the paper corresponds to
                                                   #       `θs[m]` in the code
    θm, θp, rm, rp, j, θs[m + 1], n, s = θs[m], θs[m], r0, r0, 0, θs[m], 1, 1
    α, n_α = NaN, NaN
    while s == 1
      v = rand([-1, 1])
      if v == -1
        θm, rm, _, _, θ′, n′, s′, α, n_α = build_tree(θm, rm, u, v, j, ϵ, θs[m], r0)
      else
        _, _, θp, rp, θ′, n′, s′, α, n_α = build_tree(θp, rp, u, v, j, ϵ, θs[m], r0)
      end

      if s′ == 1
        if rand() < min(1, n′ / n)
          θs[m + 1] = θ′
        end
      end
      n = n + n′
      s = s′ & (dot(θp - θm, rm) >= 0) & (dot(θp - θm, rp) >= 0)
      j = j + 1
    end
    if m + 1 <= Madapt + 1
      # NOTE: H̄ goes to negative when δ - α / n_α < 0
      H̄ = (1 - 1 / (m + t_0)) * H̄ + 1 / (m + t_0) * (δ - α / n_α)
      ϵ = exp(μ - sqrt(m) / γ * H̄)
      ϵ̄ = exp(m^float(-κ) * log(ϵ) + (1 - m^float(-κ)) * log(ϵ̄))
    else
      ϵ = ϵ̄
    end
  end

  if verbose println() end
  if verbose println("[NUTS] sampling complete with final apated ϵ = $ϵ") end

  traces = []
  for i = 1:length(θs)
    θ = from_array(vals, θs[i])
    #println(θ)
    (trace, _, _) = update(trace, args, argdiffs, θ)
    push!(traces, trace)
  end
    
  return traces
end