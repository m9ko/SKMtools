
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
			g = x -> Inf
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
    Gillespie(...) -> Vector, Array

The Gillespie algorithm first presented by by Daniel T. Gillespie in 1976. The
implementation is inspired by Frost's Gillespie.jl package.

Gillespie, D.T. (1976). A general method for numerically simulating the stochastic
time evolution of coupled chemical reactions. Journal of Computational Physics.
22(4): 403-434. doi:10.1016/0021-9991(76)90041-3
Frost, S.D.W. (2016). Gillespie.jl: Stochastic Simulation Algorithm in Julia.
Journal of Open Source Software. 1(3). doi:0.21105/joss.00042
"""
function Gillespie(
    c::Vector,           # a (nreacts x 1) vector of reaction parameters.
    X0::Vector,          # a (nspecies x 1) vector of initial species population.
    HazardFun,           # the X-component function of reaction propensity.
    StoichMatrix::Array, # a (nspecies x nreacts) stoichiometric matrix.
    t_init::Float64,     # initial time.
    t_final::Float64,    # final time.
    max_iter=10000::Int64  # number of maximum iteration.
)
    # Initialize current time and species population.
    X_curr = copy(X0)
    t_curr = copy(t_init)

    # Pre-allocate time and species population vectors and set with current info.
    t = Vector{Float64}(undef, max_iter)
    t[1] = t_curr
    nX = length(X_curr)
    X = Vector{Float64}(undef, nX * max_iter)
    X[1:nX] = X_curr

    iter = 1

    # Loop while current time is less than final time and under maximum iteration.
    while (t_curr <= t_final) & (iter < max_iter)
        iter += 1

        # Calculate the reaction propensities.
        Hazard = c .* HazardFun(X_curr)
        Hazard0 = sum(Hazard)
        p = Hazard ./ Hazard0

        # If the sum of propensities is zero, break.
        if Hazard0 <= zero(Hazard0)
            break
        end

        # Draw a time step from the exponential distribution, and a reaction
        # index from the categorical distribution.
        tau = rand(Exponential(1 / Hazard0))
        # Sometimes it encounters numerical problems when p is very skewed,
        # hence the workaround.
        j = isprobvec(p) ? rand(Categorical(p)) : findmax(Hazard)[2]

        # Update the current time and population.
        t_curr += tau
        X_curr += StoichMatrix[:,j]

        # Record the updated time and population.
        t[iter] = t_curr
        X[((iter-1)*nX + 1):iter*nX] = X_curr
    end

    if iter < max_iter
        t = t[1:iter]
        X = X[1:iter*nX]
    end

# Old code with push (less efficient memory).
"""
    if typeof(iter) == Int64
        # Begin loop.
        for i in 1:iter
            # Calculate the reaction propensities.
            Hazard = c .* HazardFun(X_curr)
            Hazard0 = sum(Hazard)
            p = Hazard ./ Hazard0

            # If the sum of propensities is zero, break.
            if Hazard0 <= zero(Hazard0)
                break
            end

            # Draw a time step from the exponential distribution, and a reaction
            # index from the categorical distribution.
            tau = rand(Exponential(1 / Hazard0))
            # Sometimes it encounters numerical problems when p is very skewed,
            # hence the workaround.
            j = isprobvec(p) ? rand(Categorical(p)) : findmax(Hazard)[2]

            # Update the current time and population.
            t_curr += tau
            X_curr += StoichMatrix[:,j]

            # Push the updated time and population.
            push!(t, t_curr)
            for X_curr_s in X_curr
                push!(X, X_curr_s)
            end
        end
    elseif typeof(iter) == Float64
        # Begin loop.
        while t_curr <= iter
            # Calculate the reaction propensities.
            Hazard = c .* HazardFun(X_curr)
            Hazard0 = sum(Hazard)
            p = Hazard ./ Hazard0

            # If the sum of propensities is zero, break.
            if Hazard0 <= zero(Hazard0)
                break
            end

            # Draw a time step from the exponential distribution, and a reaction
            # index from the categorical distribution.
            tau = rand(Exponential(1 / Hazard0))
            # Sometimes it encounters numerical problems when p is very skewed,
            # hence the workaround.
            j = isprobvec(p) ? rand(Categorical(p)) : findmax(Hazard)[2]

            # Update the current time and population.
            t_curr += tau
            X_curr += StoichMatrix[:,j]

            # Push the updated time and population.
            push!(t, t_curr)
            for X_curr_s in X_curr
                push!(X, X_curr_s)
            end
        end
    else
        error("Please give an integer or float.")
    end
"""
    # Return the time and population trajectories.
    return(t, reshape(X, nX, length(t)))
end


# Make the simulated dataset to have an equal step size of delta_t, until end
# of the dataset or specified cutoff.
function DiscretizePath(t_path, X_path, delta_t, cutoff)
    t_orig = Int64(floor((t_path[end] - t_path[1]) / delta_t) + 1)
    t_cutoff = Int64(floor((cutoff - t_path[1]) / delta_t) + 1)

    t_len = min(t_orig, t_cutoff)

    t_path_disc = zeros(t_len)
    X_path_disc = zeros((size(X_path)[1],t_len))

    t_curr = copy(t_path[1])

    t_path_disc[1] = copy(t_path[1])
    X_path_disc[:,1] = copy(X_path[:,1])

    for i in 2:t_len
        t_curr += delta_t
        t_index = findmin(abs.(t_path .- t_curr))[2]
        if t_curr == t_path[t_index]
            X_path_disc[:,i] = X_path[:,i]
        elseif t_index == length(t_path)
            X_path_disc[:,i] = X_path[:,end]
        elseif t_curr < t_path[t_index]
            t_ratio = (t_curr - t_path[t_index-1]) / (t_path[t_index] - t_path[t_index-1])
            X_path_disc[:,i] = X_path[:,t_index-1] * (1-t_ratio) + X_path[:,t_index] * t_ratio
        else
            t_ratio = (t_curr - t_path[t_index]) / (t_path[t_index+1] - t_path[t_index])
            X_path_disc[:,i] = X_path[:,t_index] * (1-t_ratio) + X_path[:,t_index+1] * t_ratio
        end
        t_path_disc[i] = t_curr
    end
    return(t_path_disc,round.(X_path_disc))
end
