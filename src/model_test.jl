#23/04/2023
"""
    zernikeapprox(n, m, ρ, θ, model_name::String)

Approximates the first kind Bessel function centered at ``a`` given ``x``. `model_name` determines the model to use.

Do not include the .bson suffix in `model_name`, as the function already appends it.
"""
function zernikeapprox(n::Int, m::Int, r::AbstractFloat, th::AbstractFloat, model_name::String)::Tuple
    BSON.@load model_name*".bson" model
    X = Array{Float32}(undef, (4, 1))
    X[:, 1] = Float32[n, m, r, th]
    out::Float32 = model(X)[end]
    return out, out - zernikerec(n, m, r, th)
end
#@time appx11 = bessel_approx(2.5, 0., s)
#@code_warntype bessel_approx(x_test, a_test, s)

"""
    zernikeapproxgpu(x, a, model_name)

Approximate the Zernike polynomials with indeces `n`, `m` at `(ρ, θ)`. `model_name` determines the model to use.\\
Uses CUDA to compute on the GPU.

Do not include the .bson suffix, as the function already appends it.
"""
function zernikeapproxGPU(n::Int, m::Int, r::AbstractFloat, th::AbstractFloat, model_name::String)::Tuple
    BSON.@load model_name*".bson" model
    model = model |> gpu
    X = Array{Float32}(undef, (4, 1))
    X[:, 1] = Float32[n, m, r, th]
    v = model(X |> gpu) |> cpu
    out::Float32 = v[end]
    return out, out - zernikerec(n, m, r, th)
end
#@time appx12 = bessel_approx_gpu(x_test, a_test, s)
#@code_warntype bessel_approx_gpu(x_test, a_test, s)
