module ZernikeApprox
export zernikerec, zernikepol, zernikecart, evalzern
export modeltrain!, modeltrainCPU!
export zernikeapprox, zernikeapproxGPU
include("deps.jl")
include("main.jl")
include("radial.jl")
include("model_creation.jl")
include("model_test.jl")
include("model_train.jl")
include("train2.jl")
end# end module
