function greet(s::String)
    s = "Hello, "*s
    println(s)
    s
end
function zernike_first_cart(n::Integer, m::Integer, x::AbstractFloat, y::AbstractFloat)
    @assert abs(m) ≤ n "m ∉ [-n, n]."
    @assert mod(n - abs(m), 2) ≠ 0.0 "n - abs(m) should be an even number."
    @assert √(x^2 + y^2) ≤ one(x) "Outside the unit circle."

    ρ = √(x^2 + y^2)
    θ = atan(y, x)
    if abs(m) == zero(m)
        norm_f = √(n + one(n))
    else
        norm_f = √(oftype(n, 2)*(n + one(n)))
    end
    if m ≥ zero(m)  
        ang_f = cos(m*θ)
    else
        ang_f = sin(m*θ)
    end
    rad_poly = radial(n, m, ρ)
    return norm_f*rad_poly*ang_f
end
function zernike_pol(n::Integer, m::Integer, ρ::AbstractFloat, θ::AbstractFloat)
    @assert ρ ≤ one(ρ) "ρ must be ≤ 1."
    @assert ρ ≥ zero(ρ) "ρ must be ≥ 0."
    @assert zero(m) ≤ m && m ≤ n "m ∉ [0, n]."
    @assert mod(n - abs(m), 2) ≠ 0 "n - abs(m) should be an even number."

    if abs(m) == zero(m)
        norm_f = sqrt(n + one(n))
    else
        norm_f = sqrt(oftype(n, 2)*(n + one(n)))
    end
    if m ≥ zero(m)
        ang_f = cos(m*θ)
    else
        ang_f = -sin(m*θ)
    end
    if n == zero(n) && m == zero(m)
        return 1.0
    end
    R = radial(n, m, ρ)
    return norm_f*R*ang_f
end
function zernike_rec(n::Integer, m::Integer, ρ::AbstractFloat, θ::AbstractFloat)
    @assert ρ ≤ one(ρ) "ρ must be ≤ 1."
    @assert ρ ≥ zero(ρ) "ρ must be ≥ 0."
    #@assert zero(m) ≤ m && m ≤ n "m ∉ [0, n]."
    @assert mod(n - abs(m), 2) ≠ 0 "n - abs(m) should be an even number."

    if abs(m) == zero(m)
        norm_f = sqrt(n + one(n))
    else
        norm_f = sqrt(oftype(n, 2)*(n + one(n)))
    end
    if m ≥ zero(m)
        ang_f = cos(m*θ)
    else
        ang_f = -sin(m*θ)
    end
    A = recursive(n, m, A)
    B = [ρ^i for i in 0:n + 1]
    return A.*B
end
function recursive(n, m, n_max)
    V = zeros(Float64, n_max + 1)
    if n == zero(n) && m == zero(m)
        V[1] = one(m)
        return V
    elseif n < m
        return V
    elseif n == m
        V[m + 1] += one(m)
        return V
    end

    left = recursive(n - 1, abs(m - 1), n_max) + recursive(n - 1, m + 1, n_max)
    left = circshift(left, 1)
    right = recursive(n - 2, m, n_max)
    return left - right
end
