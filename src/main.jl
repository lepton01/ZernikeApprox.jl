#12/04/2023
"""
    norma(n, m)

Compute the corresponding normalization factor for the Zernike polynomials.
"""
norma(n::Int, m::Int) = √(2(n + one(n))/(one(n) + Kδ0(m)))

"""
    Kδ0(m::Int)

Compute the Kroenecker delta δ(m, 0) return `1` if m = 0, `0` if m ≠ 0. The value returned is `oftype(m)`.
"""
Kδ0(m::Int) = m == zero(m) ? one(m) : zero(m)

"""
    zernikepol(n, m, ρ, θ)

Compute the explicit Zernike polynomials for a given order of n at (ρ, θ).
"""
function zernikepol(n::Int, m::Int, ρ::Real, θ::Real)
    @assert ρ ≥ zero(ρ) "ρ must be ≥ 0."
    @assert iseven(n - abs(m)) "n - abs(m) should be an even number."

    if ρ > one(ρ)
        return zero(ρ)
    end
    ang_f = m ≥ zero(m) ? cos(m*θ) : -sin(m*θ)
    if n == zero(n) && m == zero(m)
        return ang_f*norma(n, m)*one(ρ)
    end
    rad_f = radial(n, m, ρ)
    return norma(n, m)*ang_f*rad_f
end

"""
    zernikecart(n::Int, m::Int, x, y)

Compute the explicit Zernike polynomials for a given order of n at (x, y).
"""
function zernikecartrec(n::Int, m::Int, x::Real, y::Real)
    r = √(x^2 + y^2)
    th = atan(y, x)
    return zernikerec(n, m, r, th)
end

"""
    zernikerec(n, m, ρ, θ)

Compute the recurrent Zernike polynomials up to the given order. First compute the recurrent coefficient relations, then evaluate at ρ.
"""
function zernikerec(n::Int, m::Int, ρ::Real, θ::Real)
    @assert ρ ≥ zero(ρ) "ρ must be ≥ 0."
    @assert iseven(n - abs(m)) "n - abs(m) should be an even number."

    if ρ > one(ρ)
        return zero(ρ)
    end
    ang_f = m ≥ zero(m) ? cos(m*θ) : -sin(m*θ)
    if ρ == one(ρ)
        return norma(n, m)*ang_f*one(ρ)
    end
    C = recursive(n, abs(m), n)
    R = [ρ^i for i in 0:n]
    return norma(n, m)*ang_f*(C'*R)
end
