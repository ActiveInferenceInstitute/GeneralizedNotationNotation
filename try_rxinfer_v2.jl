
using RxInfer
using Distributions

@model function minimal_model(y)
    T = length(y)
    states = Vector{Any}(undef, T)
    
    # Unroll first step
    s_curr ~ Categorical(fill(1.0/3, 3))
    states[1] = s_curr
    y[1] ~ Categorical(fill(1.0/3, 3)) # Assuming y[1] works as it is argument
    
    for t in 2:T
        s_prev = states[t-1]
        s_next ~ Categorical(fill(1.0/3, 3))
        states[t] = s_next
        
        y[t] ~ Categorical(fill(1.0/3, 3))
    end
    
    return states
end

println("Model defined successfully")

data = (y = [1, 2, 3],)
result = infer(model = minimal_model(), data = data)
println("Result: $result")
