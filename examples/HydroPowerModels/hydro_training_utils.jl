# Shared helpers for HydroPowerModels training entrypoints.

"""
    parse_layers(s::AbstractString) -> Vector{Int}

Parse a comma-separated hidden-layer specification.

Empty or whitespace-only strings return `Int[]`, which lets environment
variables represent "no hidden layers" without a separate flag.

# Arguments
- `s::AbstractString`: comma-separated layer widths, with optional whitespace.

# Returns
- `Vector{Int}`: parsed hidden-layer widths.

# Examples
```julia
parse_layers("128, 64") == [128, 64]
parse_layers("") == Int[]
parse_layers("  ") == Int[]
```
"""
function parse_layers(s::AbstractString)
    return isempty(strip(s)) ?
           Int[] :
           [parse(Int, strip(x)) for x in split(s, ",") if !isempty(strip(x))]
end

function canonical_context_mode(raw_mode::AbstractString)
    mode = lowercase(strip(raw_mode))
    mode in ("", "none", "off", "false") && return ""
    mode in ("phase", "phase+progress") && return mode
    throw(ArgumentError("DR_CONTEXT must be \"\", \"phase\", or \"phase+progress\"; got \"$raw_mode\""))
end

function build_stage_context(mode::AbstractString, horizon::Int, period::Int)
    isempty(mode) && return nothing
    include_progress = mode == "phase+progress"
    return DecisionRulesExa.stage_phase_context(
        horizon;
        period = period,
        include_progress = include_progress,
    )
end

function context_run_tag(mode::AbstractString)
    isempty(mode) && return ""
    return "-ctx" * replace(mode, "+" => "p")
end
