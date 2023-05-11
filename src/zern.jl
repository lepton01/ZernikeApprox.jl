#03/05/2023
function zmat(IM::AbstractArray, ind::Array{Int})
    dims_IM = size(IM)
    x = LinRange(-1, 1, dims_IM[2])
    y = LinRange(1, -1, dims_IM[1])
    mat = Array{Float64}(undef, (dims_IM[1], dims_IM[2], size(ind, 1)))
    for ii in axes(ind, 1)
        mat[:, :, ii] = @. zernikecartrec(ind[ii, 1], ind[ii, 2], x', y)
    end
    return mat
end
function zmoment(IM::AbstractArray, ind::Array{Int})
    dims_IM = size(IM)
    z_mat = zmat(IM, ind)
    z_reshape = Array{Float64}(undef, (dims_IM[2]*dims_IM[1], size(ind, 1)))
    for ii in axes(ind, 1)
        z_reshape[:, ii] = reshape(z_mat[:, :, ii], dims_IM[2]*dims_IM[1], :)
    end
    for ii in eachindex(IM)
        if isnan(IM[ii])
            IM[ii] = 0
        end
    end
    for ii in eachindex(z_reshape)
        if isnan(z_reshape[ii])
            z_reshape[ii] = 0
        end
    end
    IM_reshape = reshape(IM, dims_IM[1]*dims_IM[2], :)
    return (z_reshape'*z_reshape)^(-1)*(z_reshape'*IM_reshape)
end
function zernrecreation(IM::AbstractArray, ind::Array{Int}, d_ind::AbstractVector)
    dimsIM = size(IM)
    z_mats = zmat(IM, ind)
    a_c = zmoment(IM, ind)
    recr = zeros(Float64, (dimsIM[1], dimsIM[2]))
    for ii in eachindex(d_ind)
        recr = recr + a_c[d_ind[ii]].*z_mats[:, :, d_ind[ii]]
    end
    return recr
end
#=
function cart2pol(x::Real, y::Real)
    r = sqrt(x^2 + y^2)
    th = atan(y, x)
    return r, th
end
=#
#=
function A(n, m)
    a = (n+1)/π
    s1 = []
    s2 = []
end
=#
