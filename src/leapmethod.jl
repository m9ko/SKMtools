# Contents of leapmethod.jl

"""
	ExplicitMethod(X::Vector, P_aXt::Vector, StoichMatrix::Array) -> Vector

Calculate the species population at next time (t + τ) via explicit method.
This is a simple calculation of current popultation + StoichMatrix * reaction #.
The naming of P_aXt comes from that it is a Poisson random variable of a(X)τ,
but it may also be a Gaussian approximation: N(a(X)τ, (a(X)τ)^2).
"""
function ExplicitMethod(
    X::Vector,          # a (nspecies x 1) vector of species population.
    P_aXt::Vector,      # a (nreacts x 1) vector of number of reactions.
    StoichMatrix::Array # a (nspecies x nreacts) stoichiometric matrix.
)
    X_next = X + StoichMatrix * P_aXt
    return(X_next)
end

"""
	ImplicitMethod(X::Vector, tau::Float64, c::Vector, aXt::Vector, P_aXt::Vector,
                   StoichMatrix::Array, HazardXFuns::Tuple{Function}) -> Vector

Calculate the species population at next time (t + τ) via implicit method, used
primarily for stiff systems. Because the involved equation is implicit, it requires
a Newton method, implemented below.
"""
function ImplicitMethod(
    X::Vector,                   # a (nspecies x 1) vector of species population.
    tau::Float64,                # the implicit time-leap.
    c::Vector,                   # a (nreacts x 1) vector of reaction parameters.
    aXt::Vector,                 # a (nreacts x 1) vector of expected number of reactions.
    P_aXt::Vector,               # a (nreacts x 1) vector of number of reactions.
    StoichMatrix::Array,         # a (nspecies x nreacts) stoichiometric matrix.
    HazardXFuns::Tuple           # a tuple of X-component of hazard functions.
)
    # The current species population with Poisson noise.
    Z = X + StoichMatrix * (P_aXt - aXt)

    # The function of the species population after time leap, which is set to 0
    # in order to find the solution for X_next = Z + StoichMatrix * (a(X_next)τ).
    G(X_next) = -X_next + Z +
                StoichMatrix *
                (c .* [fun(X_next) for fun in HazardXFuns] * tau)

    # Run the Newton's algorithm to find root.
    X_next = Newton(X, G)

    # Return the rounded species population.
    return(round.(X_next))
end


"""
	Newton(X0::Vector, G::Function, tol = 1e-5, maxIter = 1000) -> Vector

A simple implementation of Newton's algorithm for finding roots.
"""
function Newton(X0::Vector, G::Function, tol = 1e-5, maxIter = 1000)
    Xn = copy(X0)
    i = 1
    # Iterate until the absolute error is tolerated or at maximum iteration.
    while any(abs.(G(Xn)) .> tol) & (i < maxIter)
         Xn = Xn - inv(ForwardDiff.jacobian(G, Xn)) * G(Xn)
         i += 1
    end
    return(Xn)
end
