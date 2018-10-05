module Sativus

using LinearAlgebra, LatticeSites, Requires

@static if "USE_CUDA" in keys(ENV)
    const USE_CUDA = ENV["USE_CUDA"]
else
    const USE_CUDA = false
end

using Random

include("AbstractModel.jl")
include("RBM.jl")

include("wave_functions/wave_functions.jl")

@init @require CuArray="3a865a2d-5b23-5a0f-bc46-62713ec82fae" include("cuda.jl")
end # module
