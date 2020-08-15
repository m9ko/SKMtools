# Contents of lv_testmll.jl

"""
Please run the lotka_volterra.jl script before continuing!
"""

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

# The Lotka-Volterra data from time 0.0 to 8.0, with highly informative observations.
t = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
Y = [ 100.189  119.879  126.068  170.367  195.857  224.034  246.064  204.448  150.034 ;
 	  100.779  105.975  115.171  107.17   138.02   148.782  200.814  275.395  339.027 ]

# Thresholds for adaptive tau-leaping algorithm.
n_crit = 10
N_stiff = 100
H_mult = 5.0
epsilon = 0.05
delta = 0.05

# The exact parameters.
c = [0.5, 0.0025, 0.3]

# PF model with exact parameters.
exact_param = PFmodel(c, Y, t, 1.0, 2, 3, HazardFun,
					  StoichMatrix, ReactPairs, RelEpsilon,
					  n_crit, N_stiff, H_mult, epsilon, delta)

# Arbitrary parameters.
c_arb = exp.(rand(Uniform(-7,2), 3))

# PF model with exact parameters.
arb_param = PFmodel(c_arb, Y, t, 1.0, 2, 3, HazardFun,
					StoichMatrix, ReactPairs, RelEpsilon,
					n_crit, N_stiff, H_mult, epsilon, delta)

iter = 500 # number of iterations
nParts = 100 # number of particles
le = zeros(iter) # store the log evidence

# Extract log evidence from the chain, and measure time.
@time for i in 1:iter
	le[i] = sample(exact_param, SMC(), nParts).logevidence
end
