#23/04/2023
"""
    model_train!(model_name::String, ep::Int = 5_000)
    model_train!(n::Int, ρ::Real, θ::Real, num_L::Int, model_name, ep = 5_000)

Trains the `model_name.bson` model to approximate the Zernike polynomials with indeces `n, m` at `(ρ, θ)` using the CUDA GPU.

When given `Vector`s, these are used directly to train for all epochs.

When given `Real` scalars as inputs, every epoch generates a new random training data
where `n`, `m`, `ρ`, and `θ` are the upper boundary for these new values. If this is the case, it is required to specify `num_L`, as it determines the length of the random `Vector`s
generated to train.

`ep` is the number of epochs to train for.
"""
function model_train!(N::Vector{Int}, M::Vector{Int}, Ρ::Vector{AbstractFloat}, Θ::Vector{AbstractFloat}, model_name::String, ep::Int = 5_000)
    @assert N isa Vector "x must be of type Vector for training."
    @assert M isa Vector "x must be of type Vector for training."
    @assert Ρ isa Vector "x must be of type Vector for training."
    @assert Θ isa Vector "a must be of type Vector for training."
    @assert length(N) == length(M) && length(Ρ) == length(Θ) && length(Ρ) == length(M) "All vectors supplied must have the same length."

    Y_train = map(N, M, Ρ, Θ) do h, i, j, k
        zernike_rec(h, i, j, k) |> Float32
    end
    X_train = vcat(N', M', Ρ', Θ')
    train_SET = [(X_train, Y_train')] |> gpu
    BSON.@load model_name*".bson" model
    model = model |> gpu
    opt = Flux.setup(Flux.Adam(), model)
    loss_log = Float32[]
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
        push!(loss_log, l2)
        if rem(i, 1_000) == 0
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
    N_test = rand(0:maximum(N), length(N))
    M_test = rand(-maximum(N):2:maximum(N), length(N))
    Ρ_test = Float32(ρ)*rand32(length(N))
    Θ_test = Float32(θ)*rand32(length(N))
    X_test = vcat(N_test', M_test', Ρ_test', Θ_test')
    Y_test = map(N_test, M_test, Ρ_test, Θ_test) do h, i, j, k
        zernike_rec(h, i, j, k) |> Float32
    end
    Y_hat::AbstractArray = model(X_test |> gpu) |> cpu
    model = model |> cpu
    BSON.@save model_name*".bson" model
    return mean(isapprox.(Y_hat', Y_test; atol = 0.015))*100
end
function model_train!(n::Int, ρ::Real, θ::Real, num_L::Int, model_name::String, ep::Int = 5_000)
    BSON.@load model_name*".bson" model
    model = model |> gpu
    opt = Flux.setup(Flux.Adam(), model)
    loss_log = Float32[]
    for i in 1:ep
        losses = Float32[]
        n_train = rand(0:n, num_L)
        m_train = rand(-n:2:n, num_L)
        ρ_train = Float32(ρ)*rand32(num_L)
        θ_train = Float32(θ)*rand32(num_L)
        Y_train = map(n_train, m_train, ρ_train, θ_train) do h, i, j, k
            zernike_rec(h, i, j, k)
        end
        X_train = vcat(n_train', m_train', ρ_train', θ_train')
        train_SET = [(X_train, Y_train')] |> gpu
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
        push!(loss_log, l2)
        if rem(i, 1_000) == 0
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
    n_test = rand(0:n, num_L)
    m_test = rand(-n:2:n, num_L)
    ρ_test = Float32(ρ)*rand32(num_L)
    θ_test = Float32(θ)*rand32(num_L)
    X_test = vcat(n_test', m_test', ρ_test', θ_test')
    Y_test = map(n_test, m_test, ρ_test, θ_test) do h, i, j, k
        zernike_rec(h, i, j, k)
    end
    Y_hat::AbstractArray = model(X_test |> gpu) |> cpu
    model = model |> cpu
    BSON.@save model_name*".bson" model
    return mean(isapprox.(Y_hat', Y_test; atol = 0.015))*100
end

"""
    model_train_cpu!(model_name::String, ep::Int = 5_000)
    model_train_cpu!(n::Int, m::Int, ρ::Real, θ::Real, num_L::Int, model_name, ep = 5_000)

Trains the `model_name.bson` model to approximate the Zernike polynomials with indeces `n, m` at `(ρ, θ)` using the CPU.

When given `Vector`s, these are used directly to train for all epochs.

When given `Real` scalars as inputs, every epoch generates a new random training data
where `n`, `m`, `ρ`, and `θ` are the upper boundary for these new values. If this is the case, it is required to specify `num_L`, as it determines the length of the random `Vector`s
generated to train.

`ep` is the number of epochs to train for.
"""
function model_train_CPU!(N::Vector{Int}, M::Vector{Int}, Ρ::Vector{AbstractFloat}, Θ::Vector{AbstractFloat}, model_name::String, ep::Int = 5_000)
    @assert N isa Vector "x must be of type Vector for training."
    @assert M isa Vector "x must be of type Vector for training."
    @assert Ρ isa Vector "x must be of type Vector for training."
    @assert Θ isa Vector "a must be of type Vector for training."
    @assert length(N) == length(M) && length(Ρ) == length(Θ) && length(Ρ) == length(M) "All vectors supplied must have the same length."

    Y_train = map(N, M, Ρ, Θ) do h, i, j, k
        zernike_rec(h, i, j, k) |> Float32
    end
    X_train = vcat(N', M', Ρ', Θ')
    train_SET = [(X_train, Y_train')]
    BSON.@load model_name*".bson" model
    opt = Flux.setup(Flux.Adam(), model)
    loss_log = Float32[]
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
        push!(loss_log, l2)
        if rem(i, 1_000) == 0
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
    N_test = rand(0:maximum(N), length(N))
    M_test = rand(-maximum(N):2:maximum(N), length(N))
    Ρ_test = Float32(ρ)*rand32(length(N))
    Θ_test = Float32(θ)*rand32(length(N))
    X_test = vcat(N_test', M_test', Ρ_test', Θ_test')
    Y_test = map(N_test, M_test, Ρ_test, Θ_test) do h, i, j, k
        zernike_rec(h, i, j, k) |> Float32
    end
    Y_hat::AbstractArray = model(X_test)
    BSON.@save model_name*".bson" model
    return mean(isapprox.(Y_hat', Y_test; atol = 0.015))*100
end
function model_train_CPU!(n::Int, ρ::Real, θ::Real, num_L::Int, model_name::String, ep::Int = 5_000)
    BSON.@load model_name*".bson" model
    opt = Flux.setup(Flux.Adam(), model)
    loss_log = Float32[]
    for i in 1:ep
        losses = Float32[]
        n_train = rand(0:n, num_L)
        m_train = rand(-n:n, num_L)
        ρ_train = Float32(ρ)*rand32(num_L)
        θ_train = Float32(θ)*rand32(num_L)
        Y_train = map(n_train, m_train, ρ_train, θ_train) do h, i, j, k
            zernike_rec(h, i, j, k)
        end
        X_train = vcat(n_train', m_train', ρ_train', θ_train')
        train_SET = [(X_train, Y_train')]
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
        push!(loss_log, l2)
        if rem(i, 1_000) == 0
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
    n_test = rand(0:n, num_L)
    m_test = rand(-n:2:n, num_L)
    ρ_test = Float32(ρ)*rand32(num_L)
    θ_test = Float32(θ)*rand32(num_L)
    X_test = vcat(n_test', m_test', ρ_test', θ_test')
    Y_test = map(n_test, m_test, ρ_test, θ_test) do h, i, j, k
        zernike_rec(h, i, j, k)
    end
    Y_hat::AbstractArray = model(X_test)
    BSON.@save model_name*".bson" model
    return mean(isapprox.(Y_hat', Y_test; atol = 0.015))*100
end
