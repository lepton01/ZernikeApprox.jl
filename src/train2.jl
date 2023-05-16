function prep(M::AbstractArray)

    r = Flux.flatten(M)
    return r
end
"""
    coefftrain!(IM, ind, name, ep)

Train the `name` model to extract the Zernike coefficients and return the percentage of succesful testset approximation.
"""
function coefftrain!(DATA_train, DATA_test, ind, model_name::String, ep::Integer = 5_000)::Float32
    X_train = prep(DATA_train)

    Y_train::Vector{Vector} = [Zernikecoefficients(DATA_train[:, :, ii], ind) for ii in 1:axes(DATA_train, 3)]
    #X_train::Array{Float32, 2} = IMr
    train_SET::Array{Tuple} = [(X_train, Y_train')] |> gpu
    BSON.@load model_name*".bson" model
    model = model |> gpu
    opt = Flux.setup(Flux.Adam(), model)
    #loss_log = Float32[]
    for i in 1:ep
        losses = Float32[]
        for data in train_SET
            input, label = data
            l, grads = Flux.withgradient(model) do m
                result = m(input)
                mae(result, label)
            end
            push!(losses, l)
            Flux.update!(opt, model, grads[1])
        end
        l2 = sum(losses)
        #push!(loss_log, l2)
        if ep > 2_000
            if rem(i, 500) == 0
                println("Epoch = $i. Training loss = $l2")
            end
        else
            if rem(i, 100) == 0
                println("Epoch = $i. Training loss = $l2")
            end
        end
        #=
        # Stop training when some criterion is reached
        acc = mean(isapprox.(model(X_train), Y_train'; atol = 0.05))
        if acc > 0.95
            println("Stopped after $epoch epochs.")
            break
        end
        =#
    end
    X_test = prep(DATA_test)
    Y_test::Vector{Vector} = [Zernikecoefficients(DATA_test[:, :, ii], ind) for ii in 1:axes(DATA_test, 3)]
    #X_test::Array{Float32, 2} = vcat(N_test', M_test', Ρ_test', Θ_test')
    Y_hat::AbstractArray{Float32} = model(X_test |> gpu) |> cpu
    model = model |> cpu
    BSON.@save model_name*".bson" model
    return mean(isapprox.(Y_hat', Y_test; atol = 0.015))*100
end
