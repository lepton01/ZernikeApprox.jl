#23/04/2023
x_max::Float32 = 50
y_max::Float32 = 50
ρ_max = 1.5
θ_max = 2π
n::Int = 1_000

#model_create(s)

#@time bessel_train!(x_max*rand32(n), a_max*rand32(n), s, 10_000)
#@code_warntype bessel_train!(x_max*rand32(n), a_max*rand32(n), s, 10_000)

#@time bessel_train!(x_max, a_max, n, s, 10_000)
#@code_warntype bessel_train!(x_max, a_max, n, s, 10_000)

#@time bessel_train_cpu!(x_max*rand32(n), a_max*rand32(n), s, 10_000)
#@code_warntype bessel_train_cpu!(x_max*rand32(n), a_max*rand32(n), s, 10_000)

#@time bessel_train!(x_max, a_max, n, s, 10_000)
#@code_warntype bessel_train!(x_max, a_max, n, s, 10_000)


x_test::Float32 = x_max*rand32()
a_test::Float32 = a_max*rand32()

"""
    bessel_approx(x, a, model_name)

Approximates the first kind Bessel function centered at ``a`` given ``x``. `model_name` determines the model to use.

Do not include the .bson suffix in `model_name`, as the function already appends it.
"""
function bessel_approx(x::AbstractFloat, a::AbstractFloat, model_name::String)
    BSON.@load model_name*".bson" model
    X = Array{Float32}(undef, (2, 1))
    X[:, 1] = Float32.([x, a])
    out::Float32 = model(X)[end]
    return out, out - besselj(a, x)
end
#@time appx11 = bessel_approx(2.5, 0., s)
#@code_warntype bessel_approx(x_test, a_test, s)

"""
    bessel_approx_gpu(x, a, model_name)

Approximate the first kind Bessel function centered at `a` given `x`. `model_name` determines the model to use.\\
Uses CUDA to compute on the GPU.

Do not include the .bson suffix, as the function already appends it.
"""
function bessel_approx_gpu(x::AbstractFloat, a::AbstractFloat, model_name::String)
    BSON.@load model_name*".bson" model
    model = model |> gpu
    X = Array{Float32}(undef, (2, 1))
    X[:, 1] = Float32.([x, a])
    v = model(X |> gpu) |> cpu
    out::Float32 = v[end]
    return out, out - besselj(a, x)
end
#@time appx12 = bessel_approx_gpu(x_test, a_test, s)
#@code_warntype bessel_approx_gpu(x_test, a_test, s)
