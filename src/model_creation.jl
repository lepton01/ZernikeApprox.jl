#23/04/2023
"""
    modelcreate(n_in::Int, n_out::Int, model_name::String)

Create a NN model with `n_in` inputs, `n_out` outputs, and its parameters to save on file: `model_name.bson`.
Do not add `.bson` to the string input.
"""
function modelcreate(n_in::Int, n_out::Int, n::Int, model_name::String)
    nn = 2^n
    model::Chain = Chain(
        BatchNorm(n_in),
        Dense(n_in => nn, relu),
        Dense(nn => nn, relu),
        Dense(nn => n_out)
    )
    BSON.@save model_name * ".bson" model
    return
end
