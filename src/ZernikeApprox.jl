module ZernikeApprox
using LinearAlgebra, Statistics, Random
using Plots
using Flux, BSON
export greet, zernike_first_pol
include("main.jl")
include("radial.jl")

end
