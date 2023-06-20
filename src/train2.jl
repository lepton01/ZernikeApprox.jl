"""
    coefftrain!(DATA::Tuple, name; ep=100, bs=2)

Train the `name` model with the DATA tuple of X and Y to extract the Zernike coefficients and return the cost function progression through training.

bat is `Bool` for optio of using batching data ingestion (faster training) and ``bs`` is `Int` for batch size.
"""
function coefftrain!(DATA, name; ep::Int=100, bs::Int=2)
    train_SET = Flux.DataLoader(DATA |> gpu, batchsize=bs, parallel=true, shuffle=true)
    BSON.@load name * ".bson" model
    model = model |> gpu
    opt = Optimisers.setup(Optimisers.Adam(), model)
    loss_log = Float32[]
    @showprogress 1 "Training..." for i in 1:ep::Int
        l1 = Float32[]
        for (input, label) in train_SET
            loss, grads = Flux.withgradient(model) do m
                result = m(input)
                mae(result, label)
            end
            push!(l1, loss)
            Flux.update!(opt, model, grads[1])
        end
        l2 = sum(l1)
        push!(loss_log, l2)
        if ep > 2_000
            if rem(i, 500) == 0
                println("Epoch = $i. Training loss = $l2")
            end
        else
            if rem(i, 10) == 0
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
    #Y_hat1 = model(X_train |> gpu) |> cpu
    #Y_hat2 = model(X_test |> gpu) |> cpu
    model = model |> cpu
    BSON.@save name * ".bson" model
    CUDA.reclaim()
    return loss_log
end
function accuracygpu(A, B, name)
    BSON.@load name * ".bson" model
    model = model |> gpu
    X1, Y1 = A
    X2, Y2 = B
    Y_tr_r = model(X1 |> gpu) |> cpu
    Y_te_r = model(X2 |> gpu) |> cpu
    a = mean(isapprox.(Y_tr_r, Y1; atol=0.015)) * 100
    b = mean(isapprox.(Y_te_r, Y2; atol=0.015)) * 100
    return a, b
end
function accuracycpu(A, B, name)
    BSON.@load name * ".bson" model
    X1, Y1 = A
    X2, Y2 = B
    Y_tr_r = model(X1 .|> Float32)
    Y_te_r = model(X2 .|> Float32)
    a = mean(isapprox.(Y_tr_r, Y1; atol=0.015)) * 100
    b = mean(isapprox.(Y_te_r, Y2; atol=0.015)) * 100
    return a, b
end
