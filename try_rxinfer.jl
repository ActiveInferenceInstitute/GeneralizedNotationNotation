
using RxInfer
using Distributions

@model function minimal_model(y)
    T = length(y)
    states = Vector{Any}(undef, T)
    
    states[1] ~ Categorical(fill(1.0/3, 3))
    y[1] ~ Categorical(fill(1.0/3, 3))
    
    for t in 2:T
        states[t] ~ Categorical(fill(1.0/3, 3))
        y[t] ~ Categorical(fill(1.0/3, 3))
    end
end

println("Model defined successfully")

data = (y = [1, 2, 3],)
result = infer(model = minimal_model(), data = data)
println("Result: $result")
