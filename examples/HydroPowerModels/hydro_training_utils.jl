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
