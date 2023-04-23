#23/04/2023
"""
    model_train!(x::Vector{Float32}, a::Vector{Float32}, model_name, ep = 5_000)
    model_train!(x::Real, a::Real, n::Int, model_name, ep = 5_000)

Trains the `model_name.bson` model to approximate the Zernike polynomials with indeces `n, m` at `()`

When given `Vector`s these are used directly to train for all epochs.

When given `Real`s as inputs, every epoch generates a new random training data
where `x` and `a` are the upper boundary for these new values.

It is required to specify `n`, as it determines the length of the random `Vector`s
generated to train.

`ep` is the number of epochs to train for.
"""
function bessel_train!(x::Vector{Float32}, a::Vector{Float32}, model_name::String, ep::Int = 5_000)
    @assert x isa Vector "x must be of type Vector for training."
    @assert a isa Vector "a must be of type Vector for training."
    @assert length(x) == length(a) "must be of the same length."

    Y_train = map(x, a) do i, j
        besselj(j, i) |> real
    end
    X_train = vcat(x', a')
    train_SET = [(X_train, Y_train')] |> gpu
    BSON.@load model_name*".bson" model
    model = model |> gpu
    opt = Flux.setup(Flux.Adam(), model)
    loss_log = Float32[]
    for i ∈ 1:ep
        losses = Float32[]
        for data ∈ train_SET
            input, label = data
            l, grads = Flux.withgradient(model) do m
                result = m(input)
                mae(result, label)
            end
            push!(losses, l)
            Flux.update!(opt, model, grads[1])
        end
        l2 = sum(losses)
        push!(loss_log, l2)
        if rem(i, 1000) == 0
            println("Epoch = $i. Training loss = $l2")
        end
        #=
        # Stop training when some criterion is reached
        acc = mean(isapprox.(model(X_train), Y_train'; atol = 0.05))
        if acc > 0.95
            println("stopping after $epoch epochs.")
            break
        end
        =#
    end
    x_test = maximum(x)*rand32(length(x))
    a_test = maximum(a)*rand32(length(a))
    X_test = vcat(x_test', a_test')
    Y_test = map(x_test, a_test) do i, j
        besselj(j, i) |> real
    end
    Y_hat::AbstractArray = model(X_test |> gpu) |> cpu
    model = model |> cpu
    BSON.@save model_name*".bson" model
    return mean(isapprox.(Y_hat', Y_test; atol = 0.015))*100
end
function bessel_train!(x::Real, a::Real, n::Int, model_name::String, ep::Int = 5_000)
    BSON.@load model_name*".bson" model
    model = model |> gpu
    opt = Flux.setup(Flux.Adam(), model)
    loss_log = Float32[]
    for i ∈ 1:ep
        losses = Float32[]
        x_train = Float32(x)*rand32(n)
        a_train = Float32(a)*rand32(n)
        Y_train = map(x_train, a_train) do i, j
            besselj(j, i) |> real
        end
        X_train = vcat(x_train', a_train')
        train_SET = [(X_train, Y_train')] |> gpu
        for data ∈ train_SET
            input, label = data
            l, grads = Flux.withgradient(model) do m
                result = m(input)
                mae(result, label)
            end
            push!(losses, l)
            Flux.update!(opt, model, grads[1])
        end
        l2 = sum(losses)
        push!(loss_log, l2)
        if rem(i, 1000) == 0
            println("Epoch = $i. Training loss = $l2")
        end
        #=
        # Stop training when some criterion is reached
        acc = mean(isapprox.(model(X_train), Y_train'; atol = 0.05))
        if acc > 0.95
            println("stopping after $epoch epochs.")
            break
        end
        =#
    end
    x_test = Float32(x)*rand32(n)
    a_test = Float32(a)*rand32(n)
    X_test = vcat(x_test', a_test')
    Y_test = map(x_test, a_test) do i, j
        besselj(j, i) |> real
    end
    Y_hat::AbstractArray = model(X_test |> gpu) |> cpu
    model = model |> cpu
    BSON.@save model_name*".bson" model
    return mean(isapprox.(Y_hat', Y_test; atol = 0.015))*100
end
function bessel_train_cpu!(x::Vector{Float32}, a::Vector{Float32}, model_name::String, ep::Int = 5_000)
    @assert x isa Vector "x must be of type Vector for training."
    @assert a isa Vector "a must be of type Vector for training."
    @assert length(x) == length(a) "must be of the same length."

    Y_train = map(x, a) do i, j
        besselj(j, i) |> real
    end
    X_train = vcat(x', a')
    train_SET = [(X_train, Y_train')]
    BSON.@load model_name*".bson" model
    model = model
    opt = Flux.setup(Flux.Adam(), model)
    loss_log = Float32[]
    for i ∈ 1:ep
        losses = Float32[]
        for data ∈ train_SET
            input, label = data
            l, grads = Flux.withgradient(model) do m
                result = m(input)
                mae(result, label)
            end
            push!(losses, l)
            Flux.update!(opt, model, grads[1])
        end
        l2 = sum(losses)
        push!(loss_log, l2)
        if rem(i, 1000) == 0
            println("Epoch = $i. Training loss = $l2")
        end
        #=
        # Stop training when some criterion is reached
        acc = mean(isapprox.(model(X_train), Y_train'; atol = 0.05))
        if acc > 0.95
            println("stopping after $epoch epochs.")
            break
        end
        =#
    end
    x_test = maximum(x)*rand32(length(x))
    a_test = maximum(a)*rand32(length(a))
    X_test = vcat(x_test', a_test')
    Y_test = map(x_test, a_test) do i, j
        besselj(j, i) |> real
    end
    Y_hat::AbstractArray = model(X_test)
    model = model
    BSON.@save model_name*".bson" model
    return mean(isapprox.(Y_hat', Y_test; atol = 0.015))*100
end
function bessel_train_cpu!(x::Real, a::Real, n::Int, model_name::String, ep::Int = 5_000)
    BSON.@load model_name*".bson" model
    model = model
    opt = Flux.setup(Flux.Adam(), model)
    loss_log = Float32[]
    for i ∈ 1:ep
        losses = Float32[]
        x_train = Float32(x)*rand32(n)
        a_train = Float32(a)*rand32(n)
        Y_train = map(x_train, a_train) do i, j
            besselj(j, i) |> real
        end
        X_train = vcat(x_train', a_train')
        train_SET = [(X_train, Y_train')]
        for data ∈ train_SET
            input, label = data
            l, grads = Flux.withgradient(model) do m
                result = m(input)
                mae(result, label)
            end
            push!(losses, l)
            Flux.update!(opt, model, grads[1])
        end
        l2 = sum(losses)
        push!(loss_log, l2)
        if rem(i, 1000) == 0
            println("Epoch = $i. Training loss = $l2")
        end
        #=
        # Stop training when some criterion is reached
        acc = mean(isapprox.(model(X_train), Y_train'; atol = 0.05))
        if acc > 0.95
            println("stopping after $epoch epochs.")
            break
        end
        =#
    end
    x_test = Float32(x)*rand32(n)
    a_test = Float32(a)*rand32(n)
    X_test = vcat(x_test', a_test')
    Y_test = map(x_test, a_test) do i, j
        besselj(j, i) |> real
    end
    Y_hat::AbstractArray = model(X_test)
    model = model
    BSON.@save model_name*".bson" model
    return mean(isapprox.(Y_hat', Y_test; atol = 0.015))*100
end