using ZernikeApprox, Test

@testset "ZernikeApprox.jl" begin
    @test greet("Angel") == "Hello, Angel"
    @test greet("Angel") != "Hello, angel"
    @test zernike_pol(0, 0, 0., 0.) == 1
    @test zernike_rec(0, 0, 0., 0.) == 1
    @test zernike_pol(0, 0, 1., 0.) == 1
    @test zernike_rec(0, 0, 1., 0.) == 1
end
