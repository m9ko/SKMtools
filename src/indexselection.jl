# Contents of indexselection.jl

"""
	CritNonCrit(X::Vector, Hazard::Vector, StoichMatrix::Array,
		    	n_crit::Int64, nreacts = size(StoichMatrix)[2]) -> Vector, Vector

A function that separates the indices of reactions into two groups: critical and
noncritical, as discussed in Cao, Gillespie & Petzold (2007). If a reaction can
exhaust any reactant within n_crit (usually set at 10), it is considered critical.
"""
function CritNonCrit(
	X::Vector, 				 # a (nspecies x 1) vector of species population.
	Hazard::Vector, 		 # a (nreacts x 1) vector of reaction propensity.
	StoichMatrix::Array, 	 # a (nspecies x nreacts) stoichiometric matrix.
	n_crit,    				 # the critical reaction threshold.
	nreacts = length(Hazard) # the number of reactions.
)
	# Initialize output.
	crit = Vector{Int64}()
    noncrit = Vector{Int64}()

	# Calculate the critical number of firing of reaction to exhaust any reactant.
	L = X ./ StoichMatrix
	@. L[L > 0] = Inf64
	L_min = minimum(abs.(L), dims = 1)

	# If the critical number is below the threshold and propensity > 0, classify
	# as critical. Otherwise, classify as noncritical.
	for j in 1:nreacts
		if (L_min[j] < n_crit) & (Hazard[j] > 0)
			push!(crit, j)
		else
			push!(noncrit, j)
		end
	end

	# Return the vectors of critical and noncritical reaction indices.
	return(crit, noncrit)
end

"""
	EquilNonEquil(Hazard::Vector, ReactPairs::Tuple
		    	  delta::Float64, nreacts = length(Hazard)) -> Vector, Vector

A function that separates the indices of reactions into two groups: partial
equilibrium and nonequilibrium, as discussed in Cao, Gillespie & Petzold (2007).
For a pair of reversible reactions, if the relative difference in propensity is
less than delta (usually set at 0.05), it is considered in partial equilibrium.
"""
function EquilNonEquil(
	Hazard::Vector, 		 # a (nreacts x 1) vector of reaction propensity.
	ReactPairs::Tuple, 		 # a tuple of tupled pair of reversible reactions.
	delta::Float64, 		 # the bound constant, usually set at 0.05.
	nreacts = length(Hazard) # the number of reactions.
)
	# Initialize output.
	equil = Vector{Int64}()

	# Iterate over the tuple of pairs.
	for pair in ReactPairs
		# Compute whether the difference in propensity is relatively small.
		H_plus = Hazard[pair[1]]
		H_minus = Hazard[pair[2]]
		isEquilibrium = (abs(H_plus - H_minus) <= delta * min(H_plus, H_minus))

		# If so, push the equilibrium reactions vector.
		if isEquilibrium
			push!(equil, pair[1], pair[2])
		end
	end

	# Classify whichever reactions that are not in equilibrium as nonequilibrium.
	nonequil = filter(x -> !(x in equil), 1:nreacts)

	# Return the vectors of equilibrium and nonequilibrium reaction indices.
    return(equil, nonequil)
end
