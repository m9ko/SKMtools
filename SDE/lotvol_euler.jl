using Turing
using StatsPlots
using Distributions

"""
Lotka-Volterra particle filter using Euler approximation. The model itself can be
readily used for a general multivariate SDE, with modification of the prior distribution.
"""

@model LotkaVolterra(Xt, dt, ninter, npar, mu, sigma) = begin
    # Memory allocation
	nagent, nobs = size(Xt)
    theta = Vector{Real}(undef, npar)
    Xlat = Array{Real}(undef, (nagent, nobs-1, ninter))
    dtl = dt/(ninter+1)

    # Prior distribution of log theta (this is VERY specific, to avoid non semidefinite matrix)
	theta[1] ~ Uniform(0.2,0.8)
	theta[2] ~ Uniform(0.001, 0.010)
	theta[3] ~ Uniform(0.1,0.5)

    for i in 2:nobs
        # Latent variables
        if ninter > 0
            Xlat[:,i-1, 1] ~ MvNormal(Xt[:,i-1] + mu(Xt[:,i-1], theta)*dtl,
                                  	  sigma(Xt[:,i-1], theta)*dtl)
            for j in 2:ninter
                Xlat[:,i-1,j] ~ MvNormal(Xlat[:,i-1,j-1] + mu(Xlat[:,i-1,j-1], theta)*dtl,
                                     	 sigma(Xlat[:,i-1,j-1], theta)*dtl)
            end
        end

        # observed variables
        Xt[:,i] ~ MvNormal(Xlat[:,i-1,ninter] + mu(Xlat[:,i-1,ninter], theta)*dtl,
                           sigma(Xlat[:,i-1,ninter], theta)*dtl)
    end
end

"""
Drift and diffusion of Lotka-Volterra, given in Golightly & Wilkinson (2010).
"""

function mu(X, theta)
	[theta[1]*X[1] - theta[2]*X[1]*X[2], theta[2]*X[1]*X[2] - theta[3]*X[2]]
end

function sigma(X, theta)
	[[theta[1]*X[1] + theta[2]*X[1]*X[2], - theta[2]*X[1]*X[2]] [- theta[2]*X[1]*X[2], theta[2]*X[1]*X[2] + theta[3]*X[2]]]
end

"""
Testing!

Some of the functions are required from other files:
	Gillespie from src/gillespie.jl
	DiscretizePath from examples/lotka_volterra.jl
	HazardFun and StoichMatrix from src/initialization.jl

With 11 observations, 4 subintervals, 50 particles and 1000 iterations,
PG() completes sampling after 1.5 ~ 2 minutes.
"""
c = [0.5, 0.0025, 0.3]
X0 = [100.0, 100.0]

t_path, X_path = Gillespie(c,X0,HazardFun,StoichMatrix,0.0,10.0)
t, Xt = DiscretizePath(t_path,X_path,1,10)

dt = 1
ninter = 4
npar = 3
npart = 50
iter = 1000

chn = sample(LotkaVolterra(Xt, dt, ninter, npar, mu, sigma), PG(npart), iter)

estims = Array(chn)

plot(1:40, estims[1000,2:2:80])
plot!(1:4:41, Xt[2,:])
