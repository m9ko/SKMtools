using Turing
using Distributions
using LinearAlgebra
using StatsPlots
using Random

@model Euler_SDE(Xt, dt, ninter, npar, mu, sigma) = begin
    # Memory allocation
	nagent, nobs = size(Xt)
    theta = tzeros(Float64, npar)
    Xlat = Array{Real}(undef, (nagent, nobs-1, ninter))
    dtl = dt/(ninter+1)

    # Prior distribution of theta (variance is completely arbitrary)
	# theta ~ MvNormal(zeros(npar), 5)

	# Specific to Lotka-Volterra
	theta[1] ~ Uniform(0.1,1)
	theta[2] ~ Uniform(0.001, 0.05)
	theta[3] ~ Uniform(0.1,1)

    for i in 2:nobs
        # Latent variables
        if ninter > 0
			μ = Xt[:,i-1] + mu(Xt[:,i-1], theta)*dtl
			Σ = sigma(Xt[:,i-1], theta)*dtl
			Xlat[:,i-1, 1] ~ CustomMvN(μ, Σ)

            for j in 2:ninter

				μ = Xlat[:,i-1,j-1] + mu(Xlat[:,i-1,j-1], theta)*dtl
				Σ = sigma(Xlat[:,i-1,j-1], theta)*dtl
				Xlat[:,i-1,j] ~ CustomMvN(μ, Σ)

            end
        end

        # observed variables
		Xt[:,i] ~ CustomMvN(Xlat[:,i-1,ninter] + mu(Xlat[:,i-1,ninter], theta)*dtl,
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


# Custom Multivariate Normal to deal with negative values and nonposdef matrices
struct CustomMvN <: ContinuousMultivariateDistribution
	μ::Array
	Σ::Array
end

# Components required for sampling
Distributions.length(d::CustomMvN) = length(d.μ)
Distributions._rand!(rng::Random._GLOBAL_RNG, d::CustomMvN, x::Array{Float64,1}) = (sum(d.μ .> 0) == length(d.μ)) & isposdef(d.Σ) ? x = rand(MvNormal(d.μ, d.Σ)) : x = zero(x)
Distributions._logpdf(d::CustomMvN, x::Array{Float64,1}) = (sum(d.μ .> 0) == length(d.μ)) & isposdef(d.Σ) ? logpdf(MvNormal(d.μ, d.Σ), x) : -Inf

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

t_path, X_path = Gillespie(c,X0,HazardFun,StoichMatrix,0.0,20.0)
t, Xt = DiscretizePath(t_path,X_path,1,20)

dt = 1
ninter = 4
npar = 3
npart = 50
iter = 500

chn = sample(Euler_SDE(Xt, dt, ninter, npar, mu, sigma), PG(npart), iter)

chn_compGibbs = sample(Euler_SDE(Xt, dt, ninter, npar, mu, sigma),
					   Gibbs(HMC(0.001, 10, :theta), PG(npart, :Xlat)), 500)
chn_NutsGibbs = sample(Euler_SDE(Xt, dt, ninter, npar, mu, sigma),
					   Gibbs(NUTS(1000, 0.65, :theta), PG(npart, :Xlat)), 500)

estims = Array(chn)

estims[500,:]

plot(estims[:,end-2])
plot(estims[:,end-1])
plot(estims[:,end])

plot(t_path, X_path[2,:])
plot!(range(0,t_path[end], length=length(estims[iter,1:2:end-3])), estims[iter,2:2:end-3])
