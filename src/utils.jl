# utils.jl
# Small helpers shared across the package.

"""
    x_index(nx, t, i) -> Int

Return the flat-vector index for state component `i` at stage `t`.

# Arguments
- `nx::Int`: number of state components per stage.
- `t`: one-based stage index.
- `i`: one-based state-component index.

# Returns
- `Int`: index into a stage-major state trajectory of length `T * nx`.
"""
@inline x_index(nx::Int, t, i) = (t - 1) * nx + i

"""
    u_index(nu, t, i) -> Int

Return the flat-vector index for control component `i` at stage `t`.

# Arguments
- `nu::Int`: number of control components per stage.
- `t`: one-based stage index.
- `i`: one-based control-component index.

# Returns
- `Int`: index into a stage-major control trajectory of length `(T - 1) * nu`.
"""
@inline u_index(nu::Int, t, i) = (t - 1) * nu + i

"""
    w_index(nw, t, i) -> Int

Return the flat-vector index for uncertainty component `i` at stage `t`.

# Arguments
- `nw::Int`: number of uncertainty components per stage.
- `t`: one-based stage index.
- `i`: one-based uncertainty-component index.

# Returns
- `Int`: index into a stage-major uncertainty trajectory of length
  `(T - 1) * nw`.
"""
@inline w_index(nw::Int, t, i) = (t - 1) * nw + i
