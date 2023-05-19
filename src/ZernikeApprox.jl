module ZernikeApprox
export zernikerec, zernikepol, zernikecartrec
export modeltrain!, modeltrainCPU!
export zernikeapprox, zernikeapproxGPU
include("deps.jl")
include("main.jl")
include("radial.jl")
include("model_creation.jl")
include("model_test.jl")
include("model_train.jl")
include("train2.jl")
include("zern.jl")
#MODEL_NAME::String = "model_0"
end # module
