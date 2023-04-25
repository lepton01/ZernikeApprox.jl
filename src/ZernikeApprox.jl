module ZernikeApprox
export greet, zernike_pol, zernike_rec
export model_create, model_train!, model_train_CPU!, zernike_approx, zernike_approx_gpu
include("deps.jl")
include("main.jl")
include("radial.jl")
include("model_creation.jl")
include("model_test.jl")
include("model_train.jl")

MODEL_NAME::String = "model_0"

end
