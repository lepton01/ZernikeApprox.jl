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
function zernike_first_pol(n::Int, m::Int, ρ::AbstractFloat, θ::AbstractFloat)
    @assert ρ ≤ one(ρ)
    @assert 0 ≤ m && m ≤ n "m ∉ [0, n]."
    @assert mod(n - abs(m), 2) ≠ 0 "n - abs(m) should be an even number."

    if abs(m) == zero(m)
        norm_f = sqrt(n + one(n))
    else
        norm_f = sqrt(oftype(n, 2)*(n + one(n)))
    end
    if m ≥ zero(m)
        ang_f = cos(m*θ)
    else
        ang_f = sin(m*θ)
    end
    rad_poly = radial(n, m, ρ)
    return norm_f*rad_poly*ang_f
end
