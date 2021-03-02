using Turing
using StatsPlots
using Distributions

"""
Euler approximated SDE model using particle filter.

Tested with simple drift-diffusion model:

	Based on (# observation, # interval, # particle, # iteration)

	(50, 50, 50, 1000): 30 minutes
	(100, 5, 50, 1000): 12 minutes
	(100, 10, 50, 1000): 20 minutes
	(100, 20, 50, 1000): 35 minutes
	(100, 40, 50, 1000): ~ 1 hour
"""

@model SDEEuler(Xt, dt, ninter, npar, mu, sigma) = begin
    # Memory allocation
    nobs = length(Xt)
    theta = Vector{Real}(undef, npar)
    Xlat = Matrix{Real}(undef, (nobs-1, ninter))
    dtl = dt/(ninter+1)

    # Prior distribution (multivariate normal for now)
    theta ~ MvNormal(zeros(npar), 5)

    for i in 2:nobs
        # Latent variables
        if ninter > 0
            Xlat[i-1, 1] ~ Normal(Xt[i-1] + mu(Xt[i-1], theta)*dtl,
                                  sigma(Xt[i-1], theta)*sqrt(dtl))
            for j in 2:ninter
                Xlat[i-1,j] ~ Normal(Xlat[i-1,j-1] + mu(Xlat[i-1,j-1], theta)*dtl,
                                     sigma(Xlat[i-1,j-1], theta)*sqrt(dtl))
            end
        end

        # observed variables
        Xt[i] ~ Normal(Xlat[i-1,ninter] + mu(Xlat[i-1,ninter], theta)*dtl,
                                 sigma(Xlat[i-1,ninter], theta)*sqrt(dtl))
    end
end


"""
Define custom distribution such that log weight for each particle is calculated:

	log weight = log p(Y(t+1) | X(t+1), θ) + log p(X(t+1) | X(t), θ)
				 - log p(X(t+1) | X(t), Y(t+1), θ)

p(X(t+1) | X(t), Y(t+1), θ) is approximated by Gaussian densities as described in
Golightly & Wilkinson (2011).
"""
struct CustomDist <: DiscreteUnivariateDistribution
	X
	Sig_y
	logp_X_path
	logp_X_path_aux
end
Distributions.logpdf(d::CustomDist, Y::Vector) = logpdf(MvNormal(d.X, d.Sig_y), Y) + d.logp_X_path - d.logp_X_path_aux

"""
Particle filter with diffusion bridge as described in Golightly & Wilkinson (2011).

As Turing.jl assigns particle weight by pulling logpdf of the observation model, we can
define our own proposal with CustomDist as defined above.
"""

@model function PFBridge(c, Y, StoichMatrix, HazardFun, Sig_y, delta_t, t_init, t_final)
	# Time discretization.
	nblock = Int64((t_final - t_init) / delta_t)
	X = tzeros((2, nblock+1))

	logp_X_path = 0.0
	logp_X_path_aux = 0.0

	X[:,1] ~ MvNormal([100.0, 100.0], 1.0)
	Y[:,1] ~ MvNormal(X[:,1], 1.0)

	for i in 1:nblock
		# Generate parameters to sample from p(X(t+(i+1)Δt) | X(t), Y(t+1), θ)
		mu_j, Sig_j, mu_aux_j, Sig_aux_j = BridgeParam(StoichMatrix,
													   HazardFun,
													   c,
													   X[:,i],
													   Y[:,2],
													   Sig_y,
													   delta_t,
													   (t_final - t_init) - delta_t*(i-1))

		# Draw from multivariate normal with parameters defined above.
		X[:, i+1] ~ MvNormal(mu_aux_j, Sig_aux_j)

		# Add log pdf incrementally.
		logp_X_path += logpdf(MvNormal(mu_j, Sig_j), X[:, i+1]) # log p(X(t+1) | X(t), θ)
		logp_X_path_aux += logpdf(MvNormal(mu_aux_j, Sig_aux_j), X[:, i+1]) # log p(X(t+1) | X(t), Y(t+1), θ)
	end

	# Assign weight as a distribution.
	WeightDist = CustomDist(X[:,nblock+1], Sig_y, logp_X_path, logp_X_path_aux)
	Y[:,2] ~ WeightDist

end

function BridgeParam(StoichMatrix::Array,
                     HazardFun::Function,
                     c,
                     X,
                     Y,
                     Sig_y,
                     delta_t,
                     delta_j)
    Hazard = HazardFun(c, X)
    diagHazard = Diagonal(Hazard)

    alpha_j = StoichMatrix * Hazard
    beta_j = StoichMatrix * diagHazard * StoichMatrix'
    beta_delta_Sig_j = inv(beta_j * delta_j + Sig_y)

    a_j = alpha_j + beta_j * beta_delta_Sig_j * (Y - (X + alpha_j * delta_j))
    b_j = beta_j - beta_j * beta_delta_Sig_j * beta_j * delta_t

	mu_j = X + alpha_j * delta_t
	if isposdef(beta_j)
		Sig_j = beta_j * delta_t
	else
		Sig_j = Diagonal(beta_j) * delta_t
	end

	mu_aux_j = X + a_j * delta_t
	if isposdef(b_j)
		Sig_aux_j = b_j * delta_t
	else
		Sig_aux_j = Diagonal(b_j) * delta_t
	end

    return(mu_j, Sig_j, mu_aux_j, Sig_aux_j)
end


# Bootstrap
@model function PFBoot(c, Y, StoichMatrix, HazardFun, Sig_y, delta_t, t_init, t_final)
	nblock = Int64((t_final - t_init) / delta_t)
	X = tzeros((2, nblock+1))

	X[:,1] ~ MvNormal([100.0, 100.0], 1.0)
	Y[:,1] ~ MvNormal(X[:,1], 1.0)

	for i in 1:nblock
		mu_j, Sig_j, mu_aux_j, Sig_aux_j = BridgeParam(StoichMatrix, HazardFun, c,
		X[:,i],
		Y[:,2],
		Sig_y,
		delta_t,
		(t_final - t_init) - delta_t*(i-1))
		X[:, i+1] ~ MvNormal(mu_j, Sig_j)
	end

	Y[:,2] ~ MvNormal(X[:,nblock+1], Sig_y)
end

# Exact SSA
@model PFSSA(c, Y, StoichMatrix, HazardFun, Sig_y, t_init, t_final) = begin
	X = tzeros((2, 1))
    tau = tzeros(0)
    j = tzeros(Int64,0)

	X[:,1] ~ MvNormal([100.0, 100.0], 1.0)
	Y[:,1] ~ MvNormal(X[:,1], 1.0)

	t_curr = copy(t_init)
    X_curr = copy(X[:,1])
	i = 1

	while t_curr < t_final
        Hazard = HazardFun(c, X[:,i])
        Hazard0 = sum(Hazard)

        push!(tau, 0.0)
        push!(j, 0.0)

        tau[i] ~ Exponential(1/Hazard0)
        j[i] ~ Categorical(Hazard/Hazard0)

        t_curr += tau[i]
        X_curr += StoichMatrix[:,j[i]]

		X = hcat(X, X_curr)

		i += 1
	end

	Y[:,2] ~ MvNormal(X[:,i-1], Sig_y)

end

function tau_j_to_path(X0, t_init, j, tau)
	tau = collect(skipmissing(tau))
	j = collect(skipmissing(j))

	N = length(tau)

	t_curr = copy(t_init)
	X_curr = copy(X0)

	t = [t_curr]
	X = X_curr

	for i in 1:N
		t_curr += tau[i]
		X_curr += StoichMatrix[:,Int64(j[i])]
		push!(t, t_curr)
		X = hcat(X, X_curr)
	end

	return(t, X)
end
