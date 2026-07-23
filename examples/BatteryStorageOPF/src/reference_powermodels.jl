# reference_powermodels.jl
#
# Independent base-ACP reference. PowerModels builds and solves the standard
# ACPPowerModel from the SAME PGLib file, on an entirely separate modeling stack
# (PowerModels/JuMP/Ipopt). With zero batteries (or zero battery power) our
# ExaModels deterministic equivalent must reproduce this objective within a
# declared tolerance — the model-correctness gate.

using PowerModels
using JuMP
using Ipopt

# Silence PowerModels' info logging during solves.
PowerModels.silence()

"""
    reference_ac_opf(case_name; print_level=0) -> NamedTuple

Solve the base ACP OPF of PGLib `case_name` with PowerModels + Ipopt (no
batteries). Returns `(objective, status, termination, data)` where `objective`
is the operating cost in USD and `data` is the parsed PowerModels dict.
"""
function reference_ac_opf(case_name::AbstractString; print_level::Int = 0)
    _, filepath = resolve_pglib_case(case_name)
    data = PowerModels.parse_file(filepath)
    optimizer = JuMP.optimizer_with_attributes(Ipopt.Optimizer,
                                               "print_level" => print_level,
                                               "tol" => 1e-8)
    result = PowerModels.solve_ac_opf(data, optimizer)
    return (objective = Float64(result["objective"]),
            status = result["termination_status"],
            solve_time = get(result, "solve_time", NaN),
            data = data)
end

"""
    exa_base_objective(case_name; float_type=Float64,
                       madnlp_kwargs=(print_level=MadNLP.ERROR, tol=1e-8))
        -> NamedTuple

Build a single-stage, zero-battery ExaModels deterministic equivalent for
`case_name` and solve it on the CPU, returning `(objective, status, prob,
result)`. This is the ExaModels side of the base-ACP parity check.
"""
function exa_base_objective(case_name::AbstractString;
                            float_type::Type{<:AbstractFloat} = Float64,
                            madnlp_kwargs = (print_level = MadNLP.ERROR, tol = 1e-8))
    case = make_battery_case(case_name; number_of_batteries = 0)
    prob = build_battery_de(case, 1; float_type = float_type)
    result = solve_de!(prob; madnlp_kwargs...)
    return (objective = Float64(result.objective),
            status = result.status, prob = prob, result = result)
end

"""
    check_base_acp_parity(case_name; rtol=1e-3, kwargs...) -> NamedTuple

Compare the zero-battery ExaModels objective against the PowerModels/Ipopt
reference for `case_name`. Returns
`(reference, exa, abs_diff, rel_diff, within, rtol, ...)`.

`rtol` default is 1e-3: AC-OPF is nonconvex, so the two independent
solver/model stacks can in principle settle at numerically distinct—but
physically equivalent—KKT points. In practice, with the shunt admittance and
generator costs modeled identically, agreement is far tighter — measured
≈1e-10 on case14_ieee and ≈1e-12 on case300_ieee — so the loose default only
guards against environment-dependent solver noise.
"""
function check_base_acp_parity(case_name::AbstractString; rtol::Real = 1e-3, kwargs...)
    ref = reference_ac_opf(case_name)
    exa = exa_base_objective(case_name; kwargs...)
    absd = abs(exa.objective - ref.objective)
    reld = absd / max(abs(ref.objective), eps())
    return (case_name = case_name,
            reference = ref.objective, exa = exa.objective,
            abs_diff = absd, rel_diff = reld,
            within = reld <= rtol, rtol = Float64(rtol),
            reference_status = ref.status, exa_status = exa.status)
end
