# policy.jl
#
# Policy types for TS-DDR training.
#
# Two styles:
#   MLPPolicy             — stateless; called once per scenario with vcat(x0, w_flat)
#   StateConditionedPolicy — stateful LSTM; called once per stage with vcat(w_t, prev)

# ── MLPPolicy ─────────────────────────────────────────────────────────────────

"""
    MLPPolicy(model, output_dim)

Stateless MLP policy: one call with `vcat(x0, w_flat)` returns the full target
trajectory `x̂` as a flat vector of length `T*nx`.
"""
struct MLPPolicy{M}
    model::M
    output_dim::Int
end

Flux.@layer MLPPolicy

function (π::MLPPolicy)(input)
    y = π.model(input)
    return vec(y)[1:π.output_dim]
end

"""
    MLPPolicy(input_dim, output_dim; hidden=(64,64), act=tanh)
"""
function MLPPolicy(input_dim::Int, output_dim::Int;
    hidden = (64, 64),
    act = tanh,
)
    layers = Any[]
    in_dim = input_dim
    for h in hidden
        push!(layers, Flux.Dense(in_dim, h, act))
        in_dim = h
    end
    push!(layers, Flux.Dense(in_dim, output_dim))
    return MLPPolicy(Flux.Chain(layers...), output_dim)
end

# ── StateConditionedPolicy ────────────────────────────────────────────────────

"""
    StateConditionedPolicy{E,C}

Stateful LSTM policy for sequential rollout:

    x̂_t = policy(vcat(w_t, x̂_{t-1}))

- `encoder`: LSTM chain operating on the uncertainty slice `w_t`
- `combiner`: Dense layer combining encoder output with previous state

Call `Flux.reset!(policy)` before each episode.

# Flux 0.16 note
LSTM requires ≥2D input.  The forward pass reshapes the 1D `w_t` slice to
`(n_uncertainty, 1)` before encoding and squeezes back with `vec`.
"""
struct StateConditionedPolicy{E,C}
    encoder::E
    combiner::C
    n_uncertainty::Int
    n_state::Int
end

Flux.@layer StateConditionedPolicy

function (m::StateConditionedPolicy)(input)
    w = reshape(input[1:m.n_uncertainty], :, 1)   # (n_unc, 1) for LSTM
    s = input[m.n_uncertainty+1:end]
    h = vec(m.encoder(w))                          # (hidden,)
    return m.combiner(vcat(h, s))
end

Flux.reset!(m::StateConditionedPolicy) = Flux.reset!(m.encoder)

"""
    StateConditionedPolicy(n_uncertainty, n_state, n_out, layers;
                           activation=tanh, encoder_type=Flux.LSTM)

Construct a `StateConditionedPolicy`.

- `layers` : hidden sizes for the LSTM encoder, e.g. `[64, 64]`
- `n_out`  : output dimension (= nx = state dimension)
"""
function StateConditionedPolicy(
    n_uncertainty::Int,
    n_state::Int,
    n_out::Int,
    layers::AbstractVector{Int};
    activation  = tanh,
    encoder_type = Flux.LSTM,
)
    enc_sizes  = vcat(n_uncertainty, layers)
    enc_layers = [encoder_type(enc_sizes[i] => enc_sizes[i+1])
                  for i in 1:length(layers)]
    encoder  = Flux.Chain(enc_layers...)
    combiner = Flux.Dense(layers[end] + n_state => n_out, activation)
    return StateConditionedPolicy(encoder, combiner, n_uncertainty, n_state)
end
