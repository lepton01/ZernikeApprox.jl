#23/04/2023
"""
    modelcreate(n_in::Int, n_out::Int, model_name::String)

Create a NN model with `n_in` inputs, `n_out` outputs, and its parameters to save on file: `model_name.bson`.
Do not add `.bson` to the string input.
"""
function modelcreate(n_in::Int, n_out::Int, model_name::String)
    model::Chain = Chain(
        BatchNorm(n_in),
        Dense(n_in => 1024, relu),
        Dense(1024 => 1024, relu),
        Dense(1024 => 1024, relu),
        Dense(1024 => 1024, relu),
        Dense(1024 => n_out)
    )
    BSON.@save model_name*".bson" model
    return
end
