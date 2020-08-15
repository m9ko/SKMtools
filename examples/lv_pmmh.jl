# Contents of lv_pmmh.jl

"""
Please run the lotka_volterra.jl script before continuing!
"""

# Initialize parameters and initial state of the Lotka-Volterra model.
c = [0.5, 0.0025, 0.3]
X0 = [100.0, 100.0]

# Make the dataset
t_path_gill, X_path_gill = Gillespie(c, X0, HazardFun, StoichMatrix, 0.0, 40.0)
plot(t_path_gill, X_path_gill[1,:], xaxis = "Time", yaxis = "Population",
     title = "Lotka-Volterra using Gillespie algorithm", label = "Prey")
plot!(t_path_gill, X_path_gill[2,:], label = "Predator")

t_LV, Y_LV = DiscretizePath(t_path_gill, X_path_gill, 1, 30)
plot(t_LV, Y_LV[1,:], xaxis = "Time", yaxis = "Population",
     title = "Lotka-Volterra using Gillespie algorithm", label = "Prey")
plot!(t_LV, Y_LV[2,:], label = "Predator")

# Add Gaussian noise.
Y_LV += randn(size(Y_LV))

"""
Turing model for particle filtering for the marginal loglikelihood (or referred to
as the logevidence by Turing).

c is a (nreacts x 1) vector of kinetic rate parameters.
Y is a (nspecies x nobs) array of observations of the species population.
t is a (nobs x 1) vector of time of the observations.
sig_y is a number or a (nspecies x nspecies) array for the standard deviation
	  or covariance matrix for the noise in the measurement model.

The rest of the inputs are the inputs for the adaptive tau-leaping algorithm.
"""
@model function PFmodel(c, Y, t, sig_y, nspecies, nreacts, HazardFun,
						StoichMatrix, ReactPairs, RelEpsilon, n_crit,
						N_stiff, H_mult, epsilon, delta)

	# Take the first observation as a starting point.
    X_curr ~ MvNormal(Y[:,1], sig_y)

	X_track = tzeros(size(Y))
	X_track[:,1] ~ MvNormal(X_curr, 0.0)


	for i in 2:size(Y)[2]
		# Run the adaptive tau-leaping algorithm.
		t_path, X_path = AdaptiveTau(c, X_track[:,i-1], HazardFun, HazardXFuns,
									 StoichMatrix, ReactPairs, RelEpsilon, n_crit,
									 N_stiff, H_mult, epsilon, delta, t[i-1], t[i])
		# This does no effect on the marignal log likelihood, as we are using the bootstrap filter.
		X_track[:,i] ~ MvNormal(X_path[:,end-1], 0.0)
		Y[:,i] ~ MvNormal(X_path[:,end-1], sig_y)
	end
end

"""
A crude particle marginal Metropolis-Hastings with adaptive random walk step-size,
using the bootstrap-tau particle filter for the estimation of the marginal loglikelihood.
"""
# Initial set-up
iter = 500
nParts = 100
c_array = zeros((3, iter)) # parameter storage.
le_array = zeros(iter) # log evidence storage.
sig_array = ones(iter) * 0.5

# Draw the prior log(c) ~ Uniform(-7, 2).
c_array[:,1] = rand(Uniform(-7,2), 3)
# The log evidence of the parameter drawn from the prior.
le_array[1] = sample(PFmodel(exp.(c_array[:,1]), Y_LV, t_LV, 1.0, nspecies, nreacts,
							 HazardFun, StoichMatrix, ReactPairs, RelEpsilon,
							 n_crit, N_stiff, H_mult, epsilon, delta),
				     SMC(), nParts).logevidence

# Iterate through until iter.
@time for i in 2:iter
	# Draw log(c) by random walk, and obtain loglikelihood.
    c_array[:,i] = rand(MvNormal(c_array[:,i-1], sig_array[i-1]))
    le_array[i] = sample(PFmodel(exp.(c_array[:,i]), Y_LV, t_LV, 1.0, nspecies, nreacts,
								 HazardFun, StoichMatrix, ReactPairs, RelEpsilon,
								 n_crit, N_stiff, H_mult, epsilon, delta),
					     SMC(), nParts).logevidence

	# Since we use the Uniform prior and the Normal distribution for proposal,
	# all terms cancel except the loglikelihood.
	# If rejected, keep the current parameter and the loglikelihood, and decrease
	# the random walk step size.
    if le_array[i] - le_array[i-1] < log(rand(Uniform()))
        c_array[:,i] = copy(c_array[:,i-1])
        le_array[i] = copy(le_array[i-1])
		sig_array[i] = exp(log(copy(sig_array[i-1])) - 0.1/i)
	# Otherwise, take the proposed parameter and decrease the random walk step size.
	else
		sig_array[i] = exp(log(copy(sig_array[i-1])) + 0.1/i)
    end
	println(i)
end

@save "successful.jld2" le_array, c_array
plot(le_array, )
plot(exp.(c_array[3,:]))

density(exp.(c_array[1,:]))
plot!([0.5], seriestype="vline")
