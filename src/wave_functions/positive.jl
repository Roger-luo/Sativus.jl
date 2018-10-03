"""
    Positive{MT}

Positive wave function contains a real-valued ansatz.
"""
struct Positive{MT}
    ansatz::MT
end

Sativus.amplitude(wf::Positive, σ::Sites) = Sativus.forward(wf.ansatz, data(σ))
Base.eltype(wf::Positive) = eltype(wf.ansatz)
