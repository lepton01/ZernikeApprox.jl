using ZernikeApprox
using Test

@testset "ZernikeApprox.jl" begin
    @test greet("Angel") == "Hello, Angel"
    @test greet("Angel") != "Hello, angel"
end
