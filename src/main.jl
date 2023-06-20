#12/04/2023
"""
    norma(n, m)

Compute the corresponding normalization factor for the Zernike polynomials.
"""
norma(n::Int, m::Int) = √(2(n + one(n)) / (one(n) + Kδ0(m)))

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

    ρ > one(ρ) && return zero(ρ)
    ang_f = m ≥ zero(m) ? cos(m * θ) : -sin(m * θ)
    if n == zero(n) && m == zero(m)
        return ang_f * norma(n, m) * one(ρ)
    end
    rad_f = radial(n, m, ρ)
    return norma(n, m) * ang_f * rad_f
end

"""
    zernikecart(n::Int, m::Int, x, y; rec=true)

Compute the Zernike polynomials for a given order of n at (x, y). ``rec`` is a bool indicating the use of the recursive algorithm.
"""
function zernikecart(m::Int, n::Int, x::Real, y::Real; rec::Bool=true)
    r = √(x^2 + y^2)
    th = atan(y, x)
    rec == true ? zernikerec(n, m, r, th) : zernikepol(n, m, r, th)
end

"""
    zernikerec(m, n, ρ, θ)

Compute the recurrent Zernike polynomials up to the given order.
"""
function zernikerec(m::Int, n::Int, ρ::Real, θ::Real)
    @assert ρ ≥ zero(ρ) "ρ must be ≥ 0."
    @assert iseven(n - abs(m)) "n - abs(m) should be an even number."

    if ρ > one(ρ)
        return zero(ρ)
    end
    ang_f = m ≥ zero(m) ? cos(m * θ) : -sin(m * θ)
    if ρ == one(ρ)
        return norma(n, m) * ang_f * one(ρ)
    end
    C = recursive(n, m |> abs, n)
    R = [ρ^i for i in 0:n]
    return norma(n, m) * ang_f * (C' * R)
end

"""
    evalzern(N::Int, J::Vector{Int}, C::Vector{<:AbstractFloat})

Evaluate the Zernike polynomials on an N-by-N grid as specified by the Zernike coefficients of the polynomials J. Mod of the ZernikePolynomials.jl package's function ``evaluateZernike``.
"""
function evalzern(N::Int, J::Vector{Int}, C::Vector{<:AbstractFloat})
    X = range(-1, 1, N)
    A = Array{ComplexF64,3}(undef, (N, N, length(J)))
    for ii in J
        A[:, :, ii] = [zernikecart(OSA2mn(ii)..., x, y, rec=false) * (C[ii] + im * (C[ii] + length(C))) for x in X, y in X]
    end
    return reduce(+, A, dims=3)[:, :]
end
