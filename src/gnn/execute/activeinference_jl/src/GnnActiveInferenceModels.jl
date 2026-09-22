# Package module for the committed ActiveInference.jl execution project.
#
# The generated ActiveInference.jl simulators (rendered by
# src/gnn/render/activeinference_jl/) are standalone scripts that
# `using ActiveInference` directly; they do not import this module. It
# exists so the committed project (Project.toml with a real name + uuid
# header) is a well-formed Julia package: Pkg precompiles it cleanly at
# `Pkg.instantiate()` time instead of failing with a missing-source-file
# error for the project package.
module GnnActiveInferenceModels

end