using Turing
using StatsPlots
using Distributions
using LinearAlgebra

@model Euler_SDE(Xt, dt, ninter, npar, mu, sigma) = begin
    # Memory allocation
	nagent, nobs = size(Xt)
    theta = Vector{Real}(undef, npar)
    Xlat = Array{Real}(undef, (nagent, nobs-1, ninter))
    dtl = dt/(ninter+1)

    # Prior distribution of theta (variance of 5 is completely arbitrary)
	theta ~ MvNormal(zeros(npar), 5)

    for i in 2:nobs
        # Latent variables
        if ninter > 0
			μ = Xt[:,i-1] + mu(Xt[:,i-1], theta)*dtl
			Σ = sigma(Xt[:,i-1], theta)*dtl

			# If parameters are valid, sample from the Multivariate normal
			if is_valid(μ, Σ, nagent)
				Xlat[:,i-1, 1] ~ MvNormal(μ, Σ)
			else
				Turing.@addlogprob! -Inf
				return
			end

            for j in 2:ninter

				μ = Xlat[:,i-1,j-1] + mu(Xlat[:,i-1,j-1], theta)*dtl
				Σ = sigma(Xlat[:,i-1,j-1], theta)*dtl

				if is_valid(μ, Σ, nagent)
					Xlat[:,i-1,j] ~ MvNormal(μ, Σ)
                else
					Turing.@addlogprob! -Inf
					return
				end
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

function is_valid(μ, Σ, nagent)
	return (sum(μ .>= 0) == nagent) & isposdef(Σ)
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

t_path, X_path = Gillespie(c,X0,HazardFun,StoichMatrix,0.0,20.0)
t, Xt = DiscretizePath(t_path,X_path,1,20)

dt = 1
ninter = 4
npar = 3
npart = 50
iter = 1000

chn = sample(LotkaVolterra(Xt, dt, ninter, npar, mu, sigma), PG(npart), iter)

chn_compGibbs2 = sample(LotkaVolterra(Xt, dt, ninter, npar, mu, sigma),
					   Gibbs(HMC(0.001, 10, :theta), PG(npart, :Xlat)), 500)
chn_NutsGibbs = sample(LotkaVolterra(Xt, dt, ninter, npar, mu, sigma),
					   Gibbs(NUTS(1000, 0.65, :theta), PG(npart, :Xlat)), 500)

estims = Array(chn)
estims_compGibbs = Array(chn_compGibbs)
estims_compGibbs2 = Array(chn_compGibbs2)

plot(estims_compGibbs2[:,161])
hline!([0.5])
density(estims_compGibbs2[200:end,161])
vline!([0.5])

plot(1:40, estims[1000,1:2:80])
plot!(1:4:41, Xt[1,:])
