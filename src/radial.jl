#12/04/2023
"""
    radial(n::Int, m::Int, r::Real)

Compute the radial function of the Zernike polynomials in explicit factorial form.
"""
function radial(n::Int, m::Int, r::Real)
    r == 1 && return one(r)
    n < m && return zero(r)
    n == zero(n) && return r^m
    N = typeof(r)[]
    μ = n - abs(m)
    if iseven(μ)
        for k in 0:Int(μ / 2)
            v = (-1)^k * r^(n - 2k) * factorial(big(n - k)) / (factorial(big(k)) * factorial(big(Int(0.5(n - m)) - k)) * factorial(big(Int(0.5(n + m)) - k)))
            push!(N, v)
        end
    else
        return zero(r)
    end
    s = sum(N)
    return s
end
function radial_2(n::Int, m::Int, r::Real)
    r == 1 && return one(r)
    N = typeof(r)[]
    for k in 0:n
        v = (-1)^(n - k) * r^k * factorial(m + n + k) / (factorial(k) * factorial((m + k) * factorial((n - k))))
        push!(N, v)
    end
    s = sum(N)
    return s
end

"""
    recursive(n::Int, m::Int, n_max::Int)

Computes the polynomial coefficients recursively.
"""
function recursive(n::Int, m::Int, n_max::Int)
    V = zeros(Float64, n_max + 1)
    if n == zero(n) && m == zero(m)
        V[1] = one(m)
        return V
    elseif n < m
        return V
    elseif n == m
        V[m+1] += one(m)
        return V
    end

    left = recursive(n - 1, abs(m - 1), n_max) + recursive(n - 1, m + 1, n_max)
    left = circshift(left, 1)
    right = recursive(n - 2, m, n_max)
    return left - right
end
