# Contents of lotka_volterra.jl

"""
The Lotka-Volterra model, initially proposed by Alfred J. Lotka in 1910 and also
proposed by Vito Volterra in 1926. The system is described by the following set
of reactions:

    R1: S1      -> S1 + S1
    R2: S1 + S2 -> S2 + S2
    R3: S2      -> ∅

The parameters of the reactions are given by [0.5, 0.0025, 0.3], as suggested
in Boys, Wilkinson & Kirkwood (2008).

Lotka, A.J. (1910). Contribution to the Theory of Periodic Reaction. J. Phys.
Chem. 14(3):271–274. doi:10.1021/j150111a004
Volterra, V. (1926). Variazioni e fluttuazioni del numero d'individui in
specie animali conviventi. Mem. Acad. Lincei Roma. 2:31–113.
Boys, R.J., Wilkinson, D.J. & Kirkwood, T.B.L. (2008). Bayesian inference for
a discretely observed stochastic kinetic model.
"""
using SKMtools
using Distributions
using StatsPlots

# The set of reactions in `Reaction` structure.
R1 = Reaction([1], [1,1])
R2 = Reaction([1,2], [2,2])
R3 = Reaction([2], [])

# Stoichiometric matrix, hazard function (two versions) and relative epsilon function.
StoichMatrix, HazardFun, HazardXFuns, RelEpsilon = KineticModelFun(2, R1, R2, R3)
# Reversible reactions.
ReactPairs = ReactionPairsFun(R1, R2, R3) # empty tuple, as no reaction is reversible.

# Initialize parameter and initial population.
c = [0.5, 0.0025, 0.3]
X0 = [100.0, 100.0]

# Initialize thresholds.
epsilon = 0.05
delta = 0.05
n_crit = 10
N_stiff = 100
H_mult = 5.0

# Calculate the reaction propensity.
Hazard = c .* HazardFun(X0) # gives [50.0, 25.0, 30.0].

# Create the critical/noncritical reaction indices.
crit, noncrit = CritNonCrit(X0, Hazard, StoichMatrix, n_crit) # no critical reaction.
# Create the equilibrium/nonequilibrium reaction indices.
equil, nonequil = EquilNonEquil(Hazard, ReactPairs, delta) # no equilibrium reaction.

# Alias for noncritical reaction.
I_nc = noncrit
# Obtain indices that are both noncritical and not in equilibrium.
I_necr = intersect(noncrit, nonequil)


# The explicit and implicit tau candidates. In this case, both tau values should
# be the same, and the explicit method is chosen.
tau_ex = TauSelection(X0, Hazard, StoichMatrix, I_nc, epsilon, RelEpsilon)
tau_im = TauSelection(X0, Hazard, StoichMatrix, I_necr, epsilon, RelEpsilon)

# If explicit method is chosen (which is the case), get propensity and stoichiometric
# matrix for only the noncritical reactions (all reactions in this case).
Hazard_nc = Hazard[I_nc]
StoichMatrix_nc = StoichMatrix[:,I_nc]

# Calculate the expected numbers of reactions, and generate Poisson random variables
# based on them.
aX_tau = Hazard_nc * tau_ex
P_aX_tau = rand.(Poisson.(aX_tau))

# The population after time + tau via explicit method.
ExplicitMethod(X0, P_aX_tau, StoichMatrix_nc)

# Although the implicit method is not chosen in this case, it can be performed
# for completeness of this example.

# Propensity, stoichiometric matrix, hazard function (second implementation) and
# parameters of only the noncritical and nonequilibrium reactions (which are all
# reactions in this case).
Hazard_necr = Hazard[I_necr]
StoichMatrix_necr = StoichMatrix[:,I_necr]
HazardXFuns_necr = HazardXFuns[I_necr]
c_necr = c[I_necr]

# Calculate the expected numbers of reactions, and generate Poisson random variables
# based on them.
aX_tau = Hazard_necr * tau_im
P_aX_tau = rand.(Poisson.(aX_tau))

# The population after time + tau via implicit method.
ImplicitMethod(X0, tau_im, c_necr, aX_tau, P_aX_tau, StoichMatrix_necr, HazardXFuns_necr)

"""
Using the above sequence as a loop, the adaptive-tau algorithm simulates the
time and species population trajectories. More details in 'src/adaptivetau.jl'.
"""

# Initialize time interval.
t_init = 0.0
t_final = 50.0

# Simulate the Lotka-Volterra model.
t_path, X_path = AdaptiveTau(c, X0, HazardFun, HazardXFuns, StoichMatrix,
                             ReactPairs, RelEpsilon, n_crit, N_stiff, H_mult,
                             epsilon, delta, t_init, t_final)

# Plot the simulated data.
plot(t_path, X_path[1,:], xaxis = "Time", yaxis = "Population",
     title = "Lotka-Volterra using adaptive-tau algorithm", label = "Prey")
plot!(t_path, X_path[2,:], label = "Predator")

# The Gillespie algorithm-generated data. It is IMPORTANT to note that, if the last
# element is given as a Float64, it will recognize as the final time. If it is
# given as an Int64, it will recognize as the number of iterations.
t_path_gill, X_path_gill = Gillespie(c, X0, HazardFun, StoichMatrix, t_init, t_final)
plot(t_path_gill, X_path_gill[1,:], xaxis = "Time", yaxis = "Population",
     title = "Lotka-Volterra using Gillespie algorithm", label = "Prey")
plot!(t_path_gill, X_path_gill[2,:], label = "Predator")
