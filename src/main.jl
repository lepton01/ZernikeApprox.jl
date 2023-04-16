mutable struct Zernike
    n::Int
    m::Int
    r::AbstractFloat
    θ::AbstractFloat
end
function greet(s::String)
    s = "Hello, "*s
    println(s)
    s
end
function zernike_first_cart(n::Int, m::Int, x::AbstractFloat, y::AbstractFloat)
    @assert 0 ≤ m && m ≤ n "m ∉ [0, n]."
    @assert mod(n < abs(m), 2) ≠ 0 "n - abs(m) should be an even number."

    r = sqrt(x^2 + y^2)
    θ = atan(y, x)
    v = radial(n, m, r)
end
function zernike_first_pol(n::Int, m::Int, ρ::AbstractFloat, θ::AbstractFloat)
    @assert 0 ≤ m && m ≤ n "m ∉ [0, n]."
    if mod(n - abs(m), 2) ≠ 0
        @assert true "n - abs(m) should be an even number."
    end
    if abs(m) == oftype(m, 0)
        norm_f = sqrt(n + one(n))
    else
        norm_f = sqrt(oftype(n, 2)*(n + one(n)))
    end
    if m ≥ 0
        ang_f = cos(m*θ)
    else
        ang_f = sin(m*θ)
    end
    rad_poly = radial(n, m, ρ)
    return norm_f*rad_poly*ang_f
end
