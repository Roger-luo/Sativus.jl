using Test, Sativus, LatticeSites

bm = RestrictedBM(Float64, 100, 200)
Sativus.effective_energy(bm, rand(100))
wf = WaveFunction.Positive(bm)
amplitude(wf; n=3)
