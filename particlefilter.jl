# Contents of particlefilter.jl

using Turing

using StatsPlots

# Define a simple Normal model with unknown mean and variance.
@model gdemo(x) = begin
  s ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
end

#  Run sampler, collect results
chn = sample(gdemo([1.5, 2]), HMC(0.1, 5), 100)

describe(chn)

plot(chn)

"""

Turing model using the Gillespie algorithm for particle propagation.
"""
@model GillespieModel(c, Y_prev, Y, HazardFun, StoichMatrix, sig_y, nspecies, nreacts, t_init, t_final)
	tau = tzeros(0)
    j = tzeros(Int64, 0)

	Y0 ~ MvNormal(Y_prev, sig_y)

	t_curr = copy(t_init)
    X_curr = copy(Y0)

	i = 1
	while t_curr < t_final
        Hazard = c .* HazardFun(X_curr)
        Hazard0 = sum(Hazard)

        push!(tau, 0.0)
        push!(j, 0.0)

        tau[i] ~ Exponential(1 / Hazard0)
        j[i] ~ Categorical(Hazard / Hazard0)

		X_curr += StoichMatrix[:,j[i]]
        t_curr += tau[i]

		i += 1
	end

	Y ~ MvNormal(X_curr, sig_y)

end


# Tau-leap Bootstrap
@model function TauLeap_model(c, Y_prev, Y, HazardFun, StoichMatrix, sig_y,
							  ReactPairs, RelEpsilon, n_crit, N_stiff, H_mult,
							  epsilon, delta, nspecies, nreacts, t_init, t_final)
	X_track = tzeros((nspecies, 1))
	t_track = tzeros(1)
	tau = 0.0

	Y0 ~ MvNormal(Y_prev, sig_y)

	t_curr = copy(t_init)
    X_curr = copy(Y0)

	X_track[:,1] ~ MvNormal(X_curr, 0)
	t_track[1] ~ Normal(t_curr, 0)

	i = 1

	while t_curr < t_final
		Hazard = c .* HazardFun(X_curr)
        Hazard0 = sum(Hazard)

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

        tau_ex = TauSelection(X_curr, Hazard, StoichMatrix, RelEpsilon, I_nc, epsilon)
        tau_im = TauSelection(X_curr, Hazard, StoichMatrix, RelEpsilon, I_necr, epsilon)

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

		X_next = -1 * copy(X_curr)

		while any(X_next .< 0)
            if tau_1 < H_mult / Hazard0
                iter = (stiff ? 10 : 100)
                t_SSA, X_SSA = Gillespie(c, X_curr, HazardFun, StoichMatrix, t_curr, iter)

                t_track = vcat(t_track, t_SSA[2:end-1])
                X_track = hcat(X_track, X_SSA[:,2:end-1])

                t_curr = copy(t_track[end])
                X_curr = copy(X_track[:,end])

                tau = t_SSA[end] - t_SSA[end-1]
                X_next = X_SSA[:,end]

				stiff = false
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

                    if stiff & (tau_2 > tau_ex)
                        X_next = ImplicitMethod(X_curr, tau, c_nc, aXt_tau,
                                                Poi_aXt_tau, StoichMatrix_nc,
                                                HazardXFuns_nc) + change_cr
                    else
                        X_next = ExplicitMethod(X_curr, Poi_aX_tau,
                                                StoichMatrix_nc) + change_cr
                    end
                end
            end
            tau_1 = 0.5*tau_1
        end

		t_curr += tau
        X_curr = X_next

        push!(t_track, t_curr)
        X_track = hcat(X_track, X_curr)

		t_track[i+1] ~ Normal(t_curr, 0)
		X_track[:,i+1] ~ MvNormal(X_curr, 0)

		i += 1
	end

	Y ~ MvNormal(X_track[:,end-1], sig_y)
end
