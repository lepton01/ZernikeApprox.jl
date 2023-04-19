using ZernikeApprox
using Test

@testset "ZernikeApprox.jl" begin
    @test greet("Angel") == "Hello, Angel"
    @test greet("Angel") != "Hello, angel"
    @test ZernikeApprox.zernike_pol(0, 0, 0., 0.) == 1
    @test ZernikeApprox.zernike_rec(0, 0, 0., 0.) == 1
end
