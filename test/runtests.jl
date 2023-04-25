using ZernikeApprox, Test

@testset "ZernikeApprox.jl" begin
    @test zernike_pol(0, 0, 0., 0.) == 1
    @test zernike_rec(0, 0, 0., 0.) == 1
    @test zernike_pol(0, 0, 1., 0.) == 1
    @test zernike_rec(0, 0, 1., 0.) == 1

    @test isapprox(zernike_pol(7, -1, 0.5, 0), zernike_rec(7, -1, 0.5, 0.); atol = 0.01)
    @test isapprox(zernike_pol(10, -2, 0.5, 0), zernike_rec(10, -2, 0.5, 0); atol = 0.01)
    @test isapprox(zernike_pol(1, -1, 0.5, 0), zernike_rec(1, -1, 0.5, 0); atol = 0.01)
    @test isapprox(zernike_pol(5, -3, 0.5, 0), zernike_rec(5, -3, 0.5, 0); atol = 0.01)
end
