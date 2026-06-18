# utils.jl
# Small helpers shared across the package.

"""
    x_index(nx, t, i)

Linear index for state component `i ∈ 1:nx` at stage `t ∈ 1:T`
when state trajectory is stored as a flat vector of length `T*nx`.
"""
@inline x_index(nx::Int, t, i) = (t - 1) * nx + i

"""
    u_index(nu, t, i)

Linear index for control component `i ∈ 1:nu` at stage `t ∈ 1:(T-1)`
when controls are stored as a flat vector of length `(T-1)*nu`.
"""
@inline u_index(nu::Int, t, i) = (t - 1) * nu + i

"""
    w_index(nw, t, i)

Linear index for disturbance component `i ∈ 1:nw` at stage `t ∈ 1:(T-1)`
when disturbances are stored as a flat vector of length `(T-1)*nw`.
"""
@inline w_index(nw::Int, t, i) = (t - 1) * nw + i
