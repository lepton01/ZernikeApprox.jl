function greet(s::String)
    s = "Hello, "*s
    println(s)
    s
end
"""
    norma(n, m)

Compute the corresponding normalization factor for the Zernike polynomials.
"""
norma(n::Integer, m::Integer) = √(2(n + one(n))/(one(n) + Kδ0(m)))

Kδ0(m::Integer) = m == zero(m) ? one(m) : zero(m)

"""
    zernike_first_cart(n::Integer, m::Integer, x, y)

Compute the explicit Zernike polynomials for a given order of n at (x, y).
"""
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

"""
    zernike_pol(n, m, ρ, θ)

Compute the explicit Zernike polynomials for a given order of n at (ρ, θ).
"""
function zernike_pol(n::Integer, m::Integer, ρ::AbstractFloat, θ::AbstractFloat)
    @assert ρ ≤ one(ρ) "ρ must be ≤ 1."
    @assert ρ ≥ zero(ρ) "ρ must be ≥ 0."
    #@assert zero(m) ≤ m && m ≤ n "m ∉ [0, n]."
    @assert iseven(n - abs(m)) "n - abs(m) should be an even number."

    ang_f = m ≥ zero(m) ? cos(m*θ) : -sin(m*θ)
    if n == zero(n) && m == zero(m)
        return ang_f*norma(n, m)*one(ρ)
    end
    R = radial(n, m, ρ)
    return norma(n, m)*ang_f*R
end

"""
    zernike_rec(n, m, ρ, θ)

Compute the recurrent Zernike polynomials up to the given order. First compute the recurrent coefficient relations, then evaluate at ρ.
"""
function zernike_rec(n::Integer, m::Integer, ρ::AbstractFloat, θ::AbstractFloat)
    @assert ρ ≤ one(ρ) "ρ must be ≤ 1."
    @assert ρ ≥ zero(ρ) "ρ must be ≥ 0."
    #@assert zero(m) ≤ m && m ≤ n "m ∉ [0, n]."
    @assert iseven(n - abs(m)) "n - abs(m) should be an even number."

    ang_f = m ≥ zero(m) ? cos(m*θ) : -sin(m*θ)
    if ρ == 1
        return norma(n, m)*ang_f*one(ρ)
    end
    A = recursive(n, abs(m), n)
    B = [ρ^i for i in 0:n]
    return norma(n, m)*ang_f*(A'*B)
end
