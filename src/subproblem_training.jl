"""
Stage-wise TS-DDR training using MadDiff implicit differentiation.

This mirrors `train_multistage` from DecisionRules.jl (stage-wise decomposition /
single shooting) but uses ExaModels + MadDiff instead of JuMP + DiffOpt.

The policy operates in closed loop: after each stage solve, the realized state
``x_t^*`` (not the predicted target) is fed back as input to the next stage.
Zygote chains the rrule-equipped `_solve_subproblem_diffable` across stages,
producing the full gradient ``\\nabla_\\theta Q`` per Eq. 2.5.

**Important**: Each stage in the rollout uses its own `SubproblemDEProblem` and
`SubproblemSolverState`.  MadDiffSolver holds a reference to the underlying
MadNLP solver (not a copy), so sharing a single solver across stages would
cause later solves to invalidate earlier MadDiffSolver snapshots before
Zygote's backward pass reaches them.
"""

function _subproblem_rollout_loss(
    model,
    initial_state::AbstractVector{F},
    subproblems::AbstractVector{<:SubproblemDEProblem},
    solver_states::AbstractVector{SubproblemSolverState},
    w_flat::AbstractVector,
    T::Int;
    warmstart::Bool = true,
    madnlp_kwargs = NamedTuple(),
) where {F}
    nw = subproblems[1].nw
    Flux.reset!(model)

    total_obj = F(0)
    prev = F.(initial_state)

    for t in 1:T
        wt = F.(w_flat[(t-1)*nw+1 : t*nw])
        xhat_t = model(vcat(wt, prev))

        obj_t, x_real_t, ok = _solve_subproblem_diffable(
            subproblems[t], prev, Float64.(wt), xhat_t, solver_states[t];
            warmstart, madnlp_kwargs,
        )

        if !ok
            return F(NaN)
        end

        total_obj = total_obj + obj_t
        prev = x_real_t
    end

    return total_obj
end

"""
    train_subproblem_tsddr(
        model, initial_state, subproblem_builder, uncertainty_sampler;
        horizon, num_batches, num_train_per_batch, optimizer,
        adjust_hyperparameters, record_loss,
        madnlp_kwargs, warmstart,
    ) -> model

Train a TS-DDR policy using stage-wise decomposition with MadDiff gradients.

At each batch, the policy rolls out `T` stages sequentially, solving one
single-stage NLP per stage. Zygote differentiates through the full rollout
via the MadDiff-backed rrule on each stage solve. The gradient combines
envelope-theorem duals (``\\lambda_t, \\mu_t``) with solution sensitivities
(``\\partial x_t^* / \\partial x_{t-1}, \\partial x_t^* / \\partial \\hat x_t``).

Each stage uses its own subproblem and solver instance to ensure the MadDiff
VJP snapshots remain valid throughout Zygote's backward pass.

# Arguments
- `model`: Flux policy (LSTM or MLP)
- `initial_state`: initial state vector
- `subproblem_input`: either `() -> SubproblemDEProblem` (called `horizon` times)
  or a pre-built `Vector{SubproblemDEProblem}` of length `horizon`
- `uncertainty_sampler`: `() -> w_flat` (length `T * nw`)

# Keywords
- `horizon`: number of stages `T`
- `num_batches`: total gradient steps (default 100)
- `num_train_per_batch`: scenarios averaged per step (default 1)
- `optimizer`: Flux optimizer
- `adjust_hyperparameters`: `(iter, opt_state, subproblems, n) -> n`
- `record_loss`: `(iter, model, loss, tag) -> Bool`; return `true` to stop
- `madnlp_kwargs`: forwarded to MadNLP
- `warmstart`: warm-start MadNLP between solves (default `true`)
"""
function train_subproblem_tsddr(
    model,
    initial_state::AbstractVector,
    subproblem_input,
    uncertainty_sampler;
    horizon::Int,
    num_batches::Int         = 100,
    num_train_per_batch::Int = 1,
    optimizer                = Flux.Optimisers.OptimiserChain(
                                   Flux.Optimisers.ClipGrad(1.0f0),
                                   Flux.Adam(1f-3),
                               ),
    adjust_hyperparameters   = (iter, opt_state, subproblems, n) -> n,
    record_loss              = (iter, model, loss, tag) -> begin
                                   println("$tag  iter=$iter  loss=$(round(loss; digits=4))")
                                   return false
                               end,
    madnlp_kwargs            = NamedTuple(),
    warmstart::Bool          = true,
)
    F = eltype(initial_state)
    T = horizon

    subproblems = if subproblem_input isa AbstractVector
        subproblem_input
    else
        SubproblemDEProblem[subproblem_input() for _ in 1:T]
    end
    solver_states = SubproblemSolverState[
        _make_subproblem_solver(sp.model, madnlp_kwargs) for sp in subproblems
    ]

    opt_state = Flux.setup(optimizer, model)

    for iter in 1:num_batches
        n_samples = adjust_hyperparameters(iter, opt_state, subproblems, num_train_per_batch)

        batch_loss = F(0)
        batch_grads = nothing
        n_ok = 0

        for s in 1:n_samples
            w_flat = uncertainty_sampler()

            gs = Zygote.gradient(model) do m
                _subproblem_rollout_loss(
                    m, initial_state, subproblems, solver_states, w_flat, T;
                    warmstart, madnlp_kwargs,
                )
            end

            g = materialize_tangent(gs[1])
            g === nothing && continue
            !_all_finite_gradient(g) && continue

            sample_loss = _subproblem_rollout_loss(
                model, initial_state, subproblems, solver_states, w_flat, T;
                warmstart, madnlp_kwargs,
            )
            !isfinite(sample_loss) && continue

            n_ok += 1
            batch_loss += sample_loss

            if batch_grads === nothing
                batch_grads = g
            else
                batch_grads = Flux.fmap((a, b) -> a .+ b, batch_grads, g)
            end
        end

        if n_ok > 0 && batch_grads !== nothing
            avg_grads = Flux.fmap(g -> g === nothing ? nothing : g ./ F(n_ok), batch_grads)
            avg_loss = batch_loss / n_ok

            Flux.update!(opt_state, model, avg_grads)

            stop = record_loss(iter, model, avg_loss, "subproblem_tsddr")
            stop && break
        end
    end

    return model
end
