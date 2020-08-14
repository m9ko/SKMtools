# Contents of initialization.jl

"""
Data structure for chemical reactions, with reactant and product species
represented as vectors of indices.

e.g. S1 + S2 -> S3 translates into `Reaction([1,2], [3])`
	 S1 + S1 -> S2 + S3 translated into `Reaction([1,1], [2,3])`
"""
struct Reaction
    reactant::Vector
    product::Vector
end

"""
	KineticModelFun(nspecies::Int64, Rs::Reaction...) -> Array{Int64}, Function,
														 Tuple{Function}, Function

An initializing function to produce the following.
	- StoichMatrix: a stoichiometry matrix given a system of reactions.
	- HazardFun: a hazard function to calculate the propensity of each reaction.
	- HazardXFuns: another (relatively slow) version of the hazard function, only
	  for feeding the Jacobian function for implicit tau-leaping.
	- RelEpsilon: a function for calculating the bound of the relative change in
	  the propensity of reactions, as described in Cao, Gillespie & Petzold (2006).

e.g. R1 = Reaction([1,2], [3])
	 R2 = Reaction([3], [2,4])
	 R3 = Reaction([4], [])

	 KineticModelFun(4, R1, R2, R3)
"""
function KineticModelFun(
	nspecies::Int64, # a number of species involved in the system of reactions.
	Rs::Reaction... # any number of chemical reactions of type `Reaction`.
)
    nreacts = length(Rs)

	# Pre-allocate outputs.
	StoichMatrix = zeros(Int64, nspecies, nreacts)
	HazardXFuns = Vector{Function}(undef, nreacts)

	# Pre-allocate intermediate values for `HazardFun`.
    SpeciesMatrix = zeros(Float64, nreacts, nspecies + 1)
    SpeciesIndex = zeros(Int64, nreacts)
    MatrixIndex = zeros(Int64, nreacts)

    m = ones(Float64, nreacts)
    b = zeros(Float64, nreacts)

	# Pre-allocate intermediate values for `HazardXFuns`.
	gFuns = Vector{Function}(undef, nspecies)

	# Pre-allocate intermediate values for `RelEpsilon`.
	HOR = zeros(Float64, nspecies)

	# Iterate through the system of chemical reactions.
    for j in 1:nreacts
        Reac = Rs[j].reactant # reactants of reaction j.
        Prod = Rs[j].product  # products of reaction j.

		# Number of reactants and products.
        LReac = length(Reac)
        LProd = length(Prod)
		# Unique set of reactants and products.
        UReac = unique(Reac)
        UProd = unique(Prod)

		# Initialize.
        k = 1 # index for 'SpeciesIndex'.
        l = 1 # index for `MatrixIndex`.

		# If the reaction is zeroth-order, the hazard function is a constant.
		if LReac == 0
			f = (x -> 1)

		# If the reaction is first-order, the hazard function is linear.
		elseif LReac == 1
            l += Reac[1]
			f = (x -> x[Reac[1]])
			# Update the highest order reaction (HOR) for the reactant.
			HOR[Reac[1]] = max(HOR[Reac[1]], 1.0)

		# If the reaction is second-order, the hazard function is of degree 2.
        elseif LReac == 2
            k += Reac[2]
            l += Reac[1]

			# If the two required molecules are of the same species, the hazard
			# function is f(x) = 0.5cx(x-1).
            if length(UReac) == 1
                m[j] = 0.5
                b[j] = -0.5
				f = (x -> x[Reac[1]] * (x[Reac[1]] - 1) / 2)
				# Update the highest order reaction (HOR) for the reactant.
				HOR[Reac[1]] = max(HOR[Reac[1]], 2.5)

			# Otherwise, the hazard function is f(x1, x2) = c(x1x2).
			else
				f = (x -> x[Reac[1]] * x[Reac[2]])
				# Update the highest order reaction (HOR) for both reactants.
				HOR[Reac[1]] = max(HOR[Reac[1]], 2.0)
				HOR[Reac[2]] = max(HOR[Reac[2]], 2.0)
            end
		# No reactions of order higher than 2 are supported.
        else
            error("This type of reaction is not supported.")
        end

		# Create the stoichimetric matrix. This method is only possible because
		# the reactions are limited to second-order!
        StoichMatrix[UReac,j] .-= LReac / max(1, length(UReac))
        StoichMatrix[UProd,j] .+= LProd / max(1, length(UProd))

		# Update the indices (their purposes are described momentarily).
        SpeciesIndex[j] = k
        MatrixIndex[j] = (l - 1) * nreacts + j
		HazardXFuns[j] = f
    end
	# Make the array of functions to tuple of functions (it is faster this way).
	HazardXFunsTuple = Tuple(HazardXFuns)

	# Iterate through the species to define the relative bound function.
	for i in 1:nspecies
		if HOR[i] == 1.0
			g = x -> 1.0
		elseif HOR[i] == 2.0
			g = x -> 2.0
		elseif HOR[i] == 2.5
			g = x -> 2.0 + 1.0/(x - 1.0)
		else
			g = x -> 0.0
		end
		gFuns[i] = g
	end
	# Make the array of functions to tuple of functions.
	gFunsTuple = Tuple(gFuns)

	"""
		HazardFun(X::Vector) -> Vector{Float64}

	This function utilizes the indices specified from previous loop and casts
	the propensity calculation as a matrix multiplication. Returns the X-component
	of the propensity, such that c * HazardFun(X) gives the full propensity.
	"""
    function HazardFun(
		X::Vector  # a (nspecies x 1) vector of species population.
	)
        X_copy = copy(X)
		# Put the species population as an ordered list, with 1 in front.
		# e.g. If there are 3 species, SpeciesList = [1, X1, X2, X3].
        SpeciesList = zeros(Float64, nspecies + 1)
        SpeciesList[1] = 1
        SpeciesList[2:(nspecies + 1)] = X_copy

		# Create a vector for matrix multiplication of propensity calculation.
		# e.g. SpeciesVector = [1, X1, X2, X3][2,1,4] = [X1, 1, X3]
        SpeciesVector = SpeciesList[SpeciesIndex]

		# Create a matrix for matrix multiplication of propensity calculation.
		# e.g. SpeciesMatrix = [ X2   0    0	;
		#						 0    1    0	;
		#				   		 0    0  (X3-1)/2]
        SpeciesMatrix[MatrixIndex] = @. m * SpeciesVector + b

		# Returns the propensity for each reaction.
		# e.g. SpeciesMatrix * SpeciesList = [X1X2, 1, X3(X3-1)/2]
        return(SpeciesMatrix * SpeciesList)
    end

	"""
		RelEpsilon(epsilon::Float64, X::Vector) -> Vector{Float64}

	This function calculates the relative bound for change in species population
	for preserving the propensity after the time-leap. Calculating the bound
	depends on the highest order of reaction, as discussed above.
	"""
	function RelEpsilon(
		epsilon::Float64, # the bound constant, usually set at 0.05.
		X::Vector		  # a (nspecies x 1) vector of species population.
	)
		# Iterate through the species to obtain relative bound.
		rel_epsilon = epsilon ./ [gFunsTuple[i](X[i]) for i in 1:nspecies]
		return(rel_epsilon)
	end

	# Return all the outputs.
    return(StoichMatrix, HazardFun, HazardXFunsTuple, RelEpsilon)
end

"""
	ReactionPairsFun(Rs::Reaction...) -> Tuple{Tuple}

In the case there are reversible reactions in the system, this function finds
all the reaction pairs, if not given by user.
"""
function ReactionPairsFun(Rs::Reaction...)
	nreacts = length(Rs)
	ReactionsList = 1:nreacts

	# Initialize output.
	PairsArray = Vector{Tuple}(undef, 0)
	# Find matching pair by examining the reactants and products.
	for j in ReactionsList
		ReactionsList_prime = filter(x -> x != j, ReactionsList)

		# Reactants and products must be the same for both directions.
		for j_prime in ReactionsList_prime
			isRPsame = (Rs[j].reactant == Rs[j_prime].product)
			isPRsame = (Rs[j].product == Rs[j_prime].reactant)

			# If the conditions are met, push the array with the matched pair.
			if isRPsame & isPRsame
				PairTuple = (j,j_prime)
				push!(PairsArray, PairTuple)
				ReactionsList = filter(x -> !(x in PairTuple), ReactionsList)
				break
			end
		end
		# Remove the j-th reaction from the list.
		ReactionsList = filter(x -> x != j, ReactionsList)
	end

	# Return a tuple of paired tuples.
	return(Tuple(PairsArray))
end
