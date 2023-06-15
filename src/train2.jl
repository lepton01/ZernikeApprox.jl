"""
    coefftrain!(train_data, test_data, name, ep; bat=false, bs=2)

Train the `name` model to extract the Zernike coefficients and return the percentage of succesful testset approximation.

bat is `Bool` for optio of using batching data ingestion (faster training) and bs is `Int` for batch size.
"""
function coefftrain!(DATA_train, DATA_test, model_name::String, ep::Integer=5_000; bat=false, bs=2)::Float32
    X_train = Flux.flatten(DATA_train[1])
    Y_train = DATA_train[2]
    X_test = Flux.flatten(DATA_test[1])
    Y_test = DATA_test[2]
    if bat==false
        train_SET = Flux.DataLoader((X_train, Y_train), batchsize=bs, parallel=true, shuffle=true) |> gpu
    else
        train_SET = [(X_train, Y_train)] |> gpu
    end
    BSON.@load model_name * ".bson" model
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
    #X_test::Array{Float32, 2} = vcat(N_test', M_test', Ρ_test', Θ_test')
    Y_hat = model(X_test |> gpu) |> cpu
    model = model |> cpu
    BSON.@save model_name * ".bson" model
    CUDA.reclaim()
    return mean(isapprox.(Y_hat, Y_test; atol=0.015)) * 100
end
