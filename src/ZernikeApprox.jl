module ZernikeApprox

export greet
include("main.jl")
include("radial.jl")
include("deps.jl")

end
using Plots
#=
ρ = LinRange(0, 1, 100)
θ = 0
n, m = 0, 
p = plot(r, zernike_pol(n, m, ρ, θ))
=#

#n = 3
#m = 3
#a = ZernikeApprox.recursive(n, m, n)
