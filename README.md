# ZernikeApprox.jl

[![Build Status](https://github.com/lepton01/ZernikeApprox.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/lepton01/ZernikeApprox.jl/actions/workflows/CI.yml?query=branch%3Amain)

The main goal of this package is the implementation of Artificial Neural Networks to approximate Zernike polynomials `Z_n^m`, replacing the need for computing them recursively. However, this package also contains a recursive zernike coefficients computation, these results are used to train the ANN.

## Instructions

To install use Julia's Pkg manager, access it by typing `]` on your environment's REPL and then type `add ZernikeApprox`, or alternatively type in your script `using Pkg; Pkg.add("ZernikeApprox")`. Then simply command `using ZernikeApprox` in your script.

## Recursion

The recursion algorithm is based on [Honarvar and Paramesran's work](https://doi.org/10.1364/OL.38.002487), Eq. 18 to be more precise.

## System compatibility

I personally code on Windows 10 64-bit. CI should check for ubuntu compatibility.
