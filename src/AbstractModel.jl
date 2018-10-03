export initialize!, amplitude, gradient

"""
    initialize!(model, [options...])

Initialize the model with given options.
"""
function initialize! end

"""
    amplitude(model, σ)
    amplitude(model)

Return the amplitude for `σ`, or return the
vector of amplitude.
"""
function amplitude end

# amplitude(model, σ::Sites) = forward(bm, data(σ))

# for 1-D chain configuration
function amplitude(model; n::Int, spintype=Bit)
    σ = Sites(spintype, n)
    amps = zeros(eltype(model), 1<<n)
    @inbounds for i = 1:1<<n
        amps[i] = amplitude(model, σ)
        σ << 1
    end
    amps
end

"""
    gradient(model)

Return the gradient of model
"""
function gradient end
