using CUDA
using FastAI
using FastAI: FluxTraining
using MLDataPattern
using Flux
using Flux.Zygote: Params
using Flux.Optimise: update!
using Statistics: mean

##

path = datasetpath("mnist_png")
data = Datasets.loadfolderdata(path,
                               filterfn = isimagefile,
                               loadfn = (loadfile, parentname))
imgs = mapobs(data[1]) do img
    reshape(Float32.(img), :)
end

##

Din = size(getobs(imgs, 1), 1)
Dhidden = 512
Dlatent = 2
encoder = Chain(Dense(Din, Dhidden, relu), # backbone
                Parallel(tuple,
                         Dense(Dhidden, Dlatent), # μ
                         Dense(Dhidden, Dlatent)  # logσ²
                         )) |> gpu
decoder = Chain(Dense(Dlatent, Dhidden, relu), Dense(Dhidden, Din, sigmoid)) |> gpu

sample_latent(μ::AbstractArray{T}, logσ²::AbstractArray{T}) where T =
    μ .+ exp.(logσ² ./ 2) .* randn(T, size(logσ²))

function βELBO(x, x̄, μ, logσ²; β = 1)
    reconstruction_error = sum(@.((x̄ - x)^2))
    # D(N(μ, Σ)||N(0, I)) = 1/2 * (μᵀμ + tr(Σ) - length(μ) - log(|Σ|))
    kl_divergence = mean(sum(@.((μ^2 + exp(logσ²) - 1 - logσ²) / 2); dims = 1))

    return reconstruction_error + β * kl_divergence
end

##

model = (encoder = encoder, decoder = decoder)
opt = Flux.Optimiser(Momentum(1e-4), WeightDecay(1e-4))
learner = Learner(model, (training = eachbatch(imgs, 32),), opt, βELBO)

##

struct VAETrainingPhase <: FluxTraining.AbstractTrainingPhase end

function FluxTraining.step!(learner, phase::VAETrainingPhase, batch)
    FluxTraining.runstep(learner, phase, (x = batch,)) do handle, state
        ps = union(learner.params...)
        x = gpu(reduce(hcat, state.x))
        gs = gradient(ps) do
            # get encode, sample latent space, decode
            μ, logσ² = learner.model.encoder(x)
            z = sample_latent(μ, logσ²)
            x̄ = learner.model.decoder(z)

            handle(FluxTraining.LossBegin())
            state.loss = learner.lossfn(x, x̄, μ, logσ²)

            handle(FluxTraining.BackwardBegin())
            return state.loss
        end
        handle(FluxTraining.BackwardEnd())
        state.grads = (encoder = [gs[p] for p in learner.params.encoder],
                       decoder = [gs[p] for p in learner.params.decoder])
        update!(learner.optimizer, ps, gs)
    end
end

##

for epoch in 1:50
    epoch!(learner, VAETrainingPhase())
end
