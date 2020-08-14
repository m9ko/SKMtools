# Contents of adaptivetau.jl

"""
    AdaptiveTau(...) -> Vector, Array

An explicit-implicit tau-leaping method with option to switch to Gillespie
algorith. The implementation adheres to the algorithm outlined in Cao,
Gillespie and Petzold (2007).

Cao, Y., Gillespie, D.T. & Petzold, L.R. (2007) Adaptive explicit-implicit
tau-leaping method with automatic tau selection. The Journal of Chemical Physics
126(22) doi:10.1063/1.2745299
"""
function AdaptiveTau(
    c::Vector,           # a (nreacts x 1) vector of reaction parameters.
    X0::Vector,          # a (nspecies x 1) vector of initial species population.
    HazardFun,           # the X-component function of reaction propensity.
    HazardXFuns::Tuple,  # a tuple of X-component of hazard functions.
    StoichMatrix::Array, # a (nspecies x nreacts) stoichiometric matrix.
    ReactPairs::Tuple,   # a tuple of tupled pair of reversible reactions.
    RelEpsilon,          # the relative bound function.
    n_crit,              # the critical reaction threshold.
    N_stiff,             # the stiffness threshold.
    H_mult,              # the Gillespie-switchover threshold.
    epsilon::Float64,    # the threshold on change in propensity.
    delta::Float64,      # the threshold on propensity difference of reversible reaction.
    t_init::Float64,     # initial time.
    t_final::Float64     # final time.
)

    # Initialize current time and species population.
    X_curr = copy(X0)
    t_curr = copy(t_init)
    tau = 0.0

    # Initialize output of time and population trajectories.
    t_path = Vector{Float64}()
    push!(t_path, t_curr)
    X_path = copy(Array(X_curr))

    # Start the main loop.
    while t_curr < t_final
        # Calculate the propensity given each species population.
        Hazard = c .* HazardFun(X_curr)
        Hazard0 = sum(Hazard)
        # Break when the sum of propensities is equal to zero.
        if Hazard0 == zero(Hazard0)
            break
        end

        """
        Step 1:
        Identify critical and noncritical reactions, as well as partial
        equilibrium and nonequilibrium reactions.

        """
        crit, noncrit = CritNonCrit(X_curr, Hazard, StoichMatrix, n_crit)
        equil, nonequil = EquilNonEquil(Hazard, ReactPairs, delta)

        I_cr = crit
        I_nc = noncrit
        I_necr = intersect(noncrit, nonequil)

        """
        Step 2:
        Calculate candidate (explicit and implicit) τ's for the time leap.
        """
        tau_ex = TauSelection(X_curr, Hazard, StoichMatrix, I_nc, epsilon, RelEpsilon)
        tau_im = TauSelection(X_curr, Hazard, StoichMatrix, I_necr, epsilon, RelEpsilon)

        """
        Step 3:
        Determine whether the system is stiff by comparing the two candidate
        τ's. If stiff, take τ_(im); otherwise, take τ_(ex).
        """
        if tau_im > N_stiff * tau_ex
            tau_1 = tau_im
            stiff = true
        else
            tau_1 = tau_ex
            stiff = false
        end

        """
        Step 4:
        Choose between tau-leaping method and Gillespie algorithm.
        **Currently the Gillespie is turned off.**
        """

        # Set to a negative value such that the while loop below can start.
        X_next = -1 * copy(X_curr)

        while any(X_next .< 0)
            if false #tau_1 < H_mult / Hazard0
                iter = (stiff ? 10 : 100)

                # Run the Gillespie algorithm.
                t_SSA, X_SSA = Gillespie(c, X_curr, HazardFun, StoichMatrix,
                                         t_curr, iter)

                # Append all but last value, so that it could be done after the
                # while loop.
                t_path = vcat(t_path, t_SSA[2:end-1])
                X_path = hcat(X_path, X_SSA[:,2:end-1])
                t_curr = copy(t_path[end-1])
                X_curr = copy(X_path[:,end-1])

                tau = t_SSA[end] - t_SSA[end-1]
                X_next = X_SSA[:,end]

            else

                """
                Step 5:
                Calculate the propensity of critical reactions and generate
                another candidate τ.
                """
                Hazard_cr = Hazard[I_cr]
                Hazard0_cr = (isempty(I_cr) ? 0 : sum(Hazard_cr))

                tau_2 = rand(Exponential(1/Hazard0_cr))

                """
                Step 6:
                Choose a τ from the candidates and calculate the number of
                reactions during the leap.
                """
                # Propensity and stoichiometric matrix for noncritical reactions.
                Hazard_nc = Hazard[I_nc]
                StoichMatrix_nc = StoichMatrix[:,I_nc]

                # Perform the leap on noncritical reactions only.
                if tau_2 > tau_1
                    tau = tau_1

                    # Poisson random variables for number of noncritical reactions.
                    aX_tau = Hazard_nc * tau
                    Poi_aX_tau = rand.(Poisson.(aX_tau))

                    # If stiff, use the implicit method.
                    if stiff
                        HazardXFuns_nc = HazardXFuns[I_nc]
                        c_nc = c[I_nc]

                        X_next = ImplicitMethod(X_curr, tau, c_nc, aXt_tau,
                                                Poi_aXt_tau, StoichMatrix_nc,
                                                HazardXFuns_nc)

                    # If nonstiff, use the explicit method.
                    else
                        X_next = ExplicitMethod(X_curr, Poi_aX_tau, StoichMatrix_nc)
                    end

                # Fire one critical reaction and leap on all noncritical reactions.
                else
                    tau = tau_2

                    # Poisson random variables for number of noncritical reactions.
                    aX_tau = Hazard_nc * tau
                    Poi_aX_tau = rand.(Poisson.(aX_tau))

                    j_cr = I_cr[rand(Categorical(Hazard_cr ./ Hazard0_cr))]
                    change_cr = StoichMatrix[:,j_cr]

                    # If stiff and tau_2 is greater than explicit tau, compute
                    # by implicit method.
                    if stiff & (tau_2 > tau_ex)
                        X_next = ImplicitMethod(X_curr, tau, c_nc, aXt_tau,
                                                Poi_aXt_tau, StoichMatrix_nc,
                                                HazardXFuns_nc) + change_cr

                    # Otherwise, compute by explicit method.
                    else
                        X_next = ExplicitMethod(X_curr, Poi_aX_tau,
                                                StoichMatrix_nc) + change_cr
                    end
                end
            end
            # Half the tau for the case where X_next contains a negative value.
            tau_1 = 0.5*tau_1
        end

        # Update the time and the population.
        t_curr += tau
        X_curr = X_next

        # Push the updated time and the population.
        push!(t_path, t_curr)
        X_path = hcat(X_path, X_curr)
    end

    # Return the time and population trajectories.
    return(t_path, X_path)
end
