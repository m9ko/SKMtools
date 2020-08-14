using SKMtools
using Random
using Distributions
using Test

"""
An arbitrary system of reactions is used for the unit tests:

    R1: ∅       -> S1
    R2: S1      -> S2 + S3
    R3: S2 + S3 -> S1
    R4: S4 + S4 -> ∅
    R5: S2 + S4 -> S3 + S4

The resulting stoichiometric matrix and hazard (propensity) function are:

    M =      [ 1 -1  1  0  0 ;
               0  1 -1  0 -1 ;
               0  1 -1  0  1 ;
               0  0  0 -2  0 ]

    h(c,X) = [c1, c2(X1), c3(X2X3), 0.5*c4(X4(X4 - 1)), c5(X2X4)]

"""

nspecies = 4
nreacts = 5

# Reactions of different types.
R1 = Reaction([], [1])      # zeroth-order
R2 = Reaction([1], [2,3])   # first-order, reversible
R3 = Reaction([2,3], [1])   # second-order, reversible
R4 = Reaction([4,4], [])    # second-order, single molecule
R5 = Reaction([2,4], [3,4]) # second-order, two products

# Random population and parameters.
X0 = rand(Poisson(150), nspecies)
c0 = rand(Uniform(0,5), nreacts)

# Thresholds.
epsilon = rand(Uniform(0.01, 0.05))
delta = rand(Uniform(0.01, 0.05))
n_crit = rand(Poisson(10))

# Create the stoichiometric matrix and functions.
StoichMatrix, HazardFun, HazardXFuns, RelEpsilon =
    KineticModelFun(nspecies, R1, R2, R3, R4, R5)

# Specify the values to be tested.
ReactPairs = ReactionPairsFun(R1, R2, R3, R4, R5)
Hazard = c0 .* HazardFun(X0)
rel_eps = RelEpsilon(epsilon, X0)

@testset "initialization" begin
    # Test the stoichiometric matrix with an explicit form.
    @test StoichMatrix == [ 1 -1  1  0  0 ;
                            0  1 -1  0 -1 ;
                            0  1 -1  0  1 ;
                            0  0  0 -2  0 ]

    # Test the hazard function with an explicit form.
    @test Hazard == c0 .* [1.0,
                           X0[1],
                           X0[2]*X0[3],
                           0.5*X0[4]*(X0[4]-1),
                           X0[2]*X0[4]]

    # Test the relative bound for propensity with an explicit form.
    @test rel_eps == epsilon ./ [1.0,
                                 2.0,
                                 2.0,
                                 2.0 + (1 / (X0[4] - 1))]

    # Test whether the two implementations of the hazard function match.
    @test c0 .* [fun(X0) for fun in HazardXFuns] == Hazard

    # Test the reversible pair of the system.
    @test length(ReactPairs) == 1
    @test ReactPairs[1] == (2,3)

end

# Specify the values to be tested.
crit, noncrit = CritNonCrit(X0, Hazard, StoichMatrix, n_crit)
equil, nonequil = EquilNonEquil(Hazard, ReactPairs, delta)

@testset "indexselection" begin
    # Test whether the two partitions are mutually exclusive, and together add
    # up to the total number of reactions.
    @test intersect(crit, noncrit) == []
    @test intersect(equil, nonequil) == []
    @test length(crit) + length(noncrit) == nreacts
    @test length(equil) + length(nonequil) == nreacts

    # Test whether reactions grouped critical exhaust any reactant.
    for j in crit
        X1 = X0 + StoichMatrix[:,j] * n_crit
        @test any(X1 .<= 0)
    end

    # Test whether reactions grouped noncritical does not exhaust any reactant.
    for j in noncrit
        X1 = X0 + StoichMatrix[:,j] * n_crit
        @test all(X1 .>= 0)
    end

    # Test whether the reactions in parital equilbrium are in equilbrium group.
    for pair in ReactPairs
        i_equil, j_equil = pair
        H_diff = abs(Hazard[i_equil] - Hazard[j_equil])
        if H_diff <= delta * min(Hazard[i_equil], Hazard[j_equil])
            @test (i_equil ∈ equil) & (j_equil ∈ equil)
        else
            @test (i_equil ∈ nonequil) & (j_equil ∈ nonequil)
        end
    end
end

# Specify the indices.
I_nc = noncrit
I_necr = intersect(noncrit, nonequil)

# Generate the explicit and implicit tau.
tau_ex = TauSelection(X0, Hazard, StoichMatrix, I_nc, epsilon, RelEpsilon)
tau_im = TauSelection(X0, Hazard, StoichMatrix, I_necr, epsilon, RelEpsilon)

@testset "tauselection" begin
    # Test whether the two tau's are the same if I_nc = I_necr, and implicit tau
    # is greater otherwise.
    if I_necr == I_nc
        @test tau_ex == tau_im
    else
        @test tau_im > tau_ex
    end
end

# Create propensity vector and stoichiometric matrix for noncritical reactions.
Hazard_nc = Hazard[I_nc]
StoichMatrix_nc = StoichMatrix[:,I_nc]

# Generate a Poisson number of reactions during the tau-leap.
aXt_nc = Hazard_nc * tau_ex
P_aXt_nc = rand.(Poisson.(aXt_nc))

# The updated species population via explicit method.
X1_ex = ExplicitMethod(X0, P_aXt_nc, StoichMatrix_nc)

# Create propensity vector and stoichiometric matrix for noncritical and
# nonequilibrium reactions, as well as for parameters and hazard functions.
Hazard_necr = Hazard[I_necr]
StoichMatrix_necr = StoichMatrix[:,I_necr]
HazardXFuns_necr = HazardXFuns[I_necr]
c_necr = c0[I_necr]

# Generate a Poisson number of reactions during the tau-leap.
aXt_necr = Hazard_necr * tau_im
P_aXt_necr = rand.(Poisson.(aXt_necr))

# The updated species population via implicit method.
X1_im = ImplicitMethod(X0, tau_im, c_necr, aXt_necr, P_aXt_necr,
                       StoichMatrix_necr, HazardXFuns_necr)

@testset "leapmethods" begin
    # Test whether the relative change in population is bounded by epsilon.
    # However in reality, we are dealing with Poisson random variables which
    # may not necessarily adhere to our constraints, and also the change in
    # population is approximated in the first place.
    @test all(@. abs(abs(1 - X1_ex / X0) - epsilon) <= 0.1)
    @test all(@. abs(abs(1 - X1_im / X0) - epsilon) <= 0.1)
end
