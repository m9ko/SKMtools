# Contents of gillespie.jl

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
