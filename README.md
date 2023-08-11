# SKMtools: a toolbox for adaptive tau-leaping algorithm.

Stochastic kinetic models have received attention in the domain of systems biology, for its ability to capture the stochastic nature of many biochemical reaction systems. An adaptive tau-leaping method is an accelerated approximation method to the exact Gillespie algorithm, for simulation of the population dynamics. This package is designed for solving the inverse problem of parameter estimation by combining the said adaptive tau-leaping method with bootstrap particle filter to estimate the marginal loglikelihood efficiently. A full implementation of the adaptive tau-leaping algorithm is given in Julia, as well as preliminary scripts of the bootstrap particle filter and particle marginal Metropolis-Hastings in Turing.jl.

To explore the Lotka-Volterra model, please see the `example` folder.

