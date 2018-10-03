export RestrictedBM, nvisible, nhidden, weights, hidden_bias, visible_bias

"""
    RBMParameter{T, MatrixType, VectorType}

Containts parameters for a RBM
"""
struct RBMParameter{T<:Number, MatrixType <: AbstractMatrix{T}, VectorType <: AbstractVector{T}}
    weights::MatrixType
    visible::VectorType
    hidden::VectorType

    RBMParameter(weights::MT, visible::VT, hidden::VT) where {T<:Number, MT <: AbstractMatrix{T}, VT <: AbstractVector{T}} =
        new{T, MT, VT}(weights, visible, hidden)
end

@static if USE_CUDA
    cu(ps::RBMParameter) = RBMParameter(cu(ps.weights), cu(ps.visible), cu(ps.hidden))
    cu(ps::RBMParameter{<:Number, <:CuArray, <:CuArray}) = deepcopy(ps)
end

@inline function _apply!(f::Function, params::RBMParameter)
    f(params.weights)
    f(params.visible)
    f(params.hidden)
    params
end

Base.fill!(params::RBMParameter, v) = _apply(x->fill!(x, v), params)
Random.rand!(params::RBMParameter) = _apply!(rand!, params)
Random.randn!(params::RBMParameter) = _apply!(randn!, params)

"""
    RestrictedBM{T, M, N, ParamType}

Restricted Boltzmann Machine with `M` visible units, and `N` hidden units.
"""
struct RestrictedBM{T, M, N, ParamType <: RBMParameter}
    params::ParamType
    grads::ParamType

    # we don't need to initialize gradients if we don't need it at first
    RestrictedBM{M, N}(params::PT) where {T, M, N, PT <: RBMParameter{T}} = new{T, M, N, PT}(params)
    # grads should have the exactly same type with params
    RestrictedBM{M, N}(params::PT, grads::PT) where {T, M, N, PT <: RBMParameter{T}} = new{T, M, N, PT}(params, grads)
end

# Constructors
function RestrictedBM(::Type{T}, visible::Int, hidden::Int;
        device::Symbol=:cpu0) where {T <: Number}

    params = RBMParameter(
        randn(T, hidden, visible), # weights
        randn(T, visible), # visible bias
        randn(T, hidden) # hidden bias
    )

@static if USE_CUDA
    if device === :cuda
        params = cu(params)
    end
end

    RestrictedBM{visible, hidden}(params)
end

# define default precision
RestrictedBM(visible::Int, hidden::Int; device::Symbol=:cpu0) = RestrictedBM(Float64, visible, hidden; device=device)

Base.eltype(::RestrictedBM{T}) where T = T
nvisible(::RestrictedBM{T, M}) where {T, M} = M
nhidden(::RestrictedBM{T, M, N}) where {T, M, N} = N

weights(bm::RestrictedBM) = bm.params.weights
hidden_bias(bm::RestrictedBM) = bm.params.hidden
visible_bias(bm::RestrictedBM) = bm.params.visible

function Base.show(io::IO, bm::RestrictedBM)
    println(io, "Restricted Boltzmann Machine (", eltype(bm), "):")
    println(io, "  visible units: ", nvisible(bm))
    print(io, "  hidden units: ", nhidden(bm))
end

_theta(bm::RestrictedBM, σ) = bm.params.weights * σ .+ bm.params.hidden

# single
function forward(bm::RestrictedBM, σ::AbstractVector)
    θ = _theta(bm, σ)
    μ = dot(bm.params.visible, σ)
    exp(μ) * prod(cosh.(θ))
end

# NOTE: this should use thresholds
softplus(x::Number) = log(exp(x) + 1)

function effective_energy(bm::RestrictedBM, σ::AbstractVector)
    θ = _theta(bm, σ)
    μ = dot(σ, bm.params.visible)
    - (μ + sum(softplus.(θ)))
end

# batched
function forward(bm::RestrictedBM, σ::AbstractMatrix)
    θ = _theta(bm, σ)
    μ = transpose(bm.params.visible) * σ
    @. exp(μ) * $(prod(cosh.(θ), dims=1))
end

function effective_energy(bm::RestrictedBM, σ::AbstractMatrix)
    θ = _theta(bm, σ)
    μ = transpose(bm.params.visible) * σ
    - @.(μ + $(sum(softplus.(θ), dims=1)))
end
