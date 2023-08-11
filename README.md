# SKMtools: a toolbox for adaptive tau-leaping algorithm.

A Julia package for adaptive tau-leaping algorithm for stochastic kinetic models.

Stochastic kinetic models have received attention in the domain of systems biology, for its ability to capture the stochastic nature of many biochemical reaction systems. An adaptive tau-leaping method is an accelerated approximation method to the exact Gillespie algorithm, for simulation of the population dynamics. This package is designed to provide an efficient implementation of the said adaptive tau-leaping method, and furthermore solve the inverse problem of parameter inference by combining with the bootstrap particle filter to estimate the marginal loglikelihood. 

To explore the Lotka-Volterra model, please see the `example` folder.

