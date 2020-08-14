# Contents of tauselection.jl

"""
	TauSelection(X::Vector, Hazard::Vector, StoichMatrix::Array,
	RelEpsilon::Function, indices::Vector, epsilon::Float64) -> Float64

The time-leap candidate, τ, is computed according to the formula in Cao,
Gillespie & Petzold (2006). If the given indices are all the noncritical
reactions, the proposed τ is for an explicit leap. If the given indices are
the intersection of noncritical and nonequilibrium reactions, it is for an
implicit leap.
"""
function TauSelection(
	X::Vector, 			  # a (nspecies x 1) vector of species population.
	Hazard::Vector, 	  # a (nreacts x 1) vector of reaction propensity.
	StoichMatrix::Array,  # a (nspecies x nreacts) stoichiometric matrix.
	indices::Vector, 	  # a vector of reaction indices involved in the leap.
	epsilon::Float64,	  # the bound constant, usually set at 0.05.
	RelEpsilon			  # the relative bound function.
)
	# If the set of indices is empty, τ = ∞.
	if length(indices) == 0
		return(Inf64)
	# Calculate the maximum value of τ that meets the leap conditions, given by
	# the μ and σ2 constraints. More details in Cao, Gillespie & Petzold (2006).
	else
		# The mean and variance of the Poisson r.v. of the reaction numbers.
		μ_X = StoichMatrix[:,indices] * Hazard[indices]
		σ2_X = abs2.(StoichMatrix[:,indices]) * Hazard[indices]

		# The relative bound on the reaction propensity.
		ϵ_rel = RelEpsilon(epsilon, X)

		# The τ constraints given by the relative change in species population.
		μ_constraint = max(maximum(ϵ_rel .* X),1) ./ abs.(μ_X)
		σ2_constraint = max(maximum(ϵ_rel .* X),1)^2 ./ σ2_X

		# Return the minimum of the calculated constraints.
		return(minimum(vcat(μ_constraint, σ2_constraint)))
	end
end
