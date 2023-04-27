using ZernikeApprox, Test

@testset "ZernikeApprox.jl" begin
    @test zernikepol(0, 0, 0., 0.) == 1
    @test zernikerec(0, 0, 0., 0.) == 1
    @test zernikepol(0, 0, 1., 0.) == 1
    @test zernikerec(0, 0, 1., 0.) == 1

    @test isapprox(zernikepol(7, -1, 0.5, 0), zernikerec(7, -1, 0.5, 0.); atol = 0.01)
    @test isapprox(zernikepol(10, -2, 0.5, 0), zernikerec(10, -2, 0.5, 0); atol = 0.01)
    @test isapprox(zernikepol(1, -1, 0.5, 0), zernikerec(1, -1, 0.5, 0); atol = 0.01)
    @test isapprox(zernikepol(5, -3, 0.5, 0), zernikerec(5, -3, 0.5, 0); atol = 0.01)
end
