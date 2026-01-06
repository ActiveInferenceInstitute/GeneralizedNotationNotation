
using RxInfer
using Distributions

A = [0.5 0.5; 0.5 0.5]

@model function hmm_test()
    s = Vector{Any}(undef, 2)
    s1 ~ Categorical([0.5, 0.5])
    s[1] = s1
    
    # Try DiscreteTransition
    # s[1] is a VariableRef
    s2 ~ DiscreteTransition(s[1], A)
    s[2] = s2
end

println("Defining model...")
try
    res = infer(model = hmm_test(), data = NamedTuple(), iterations = 10)
    println("Success: $res")
catch e
    println("Failed: $e")
    showerror(stdout, e, catch_backtrace())
end
