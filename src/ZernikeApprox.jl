module ZernikeApprox
export zernikerec, zernikepol
export modelcreate, modeltrain!, modeltrainCPU!
export zernikeapprox, zernikeapproxGPU
include("deps.jl")
include("main.jl")
include("radial.jl")
include("model_creation.jl")
include("model_test.jl")
include("model_train.jl")
#MODEL_NAME::String = "model_0"
end
