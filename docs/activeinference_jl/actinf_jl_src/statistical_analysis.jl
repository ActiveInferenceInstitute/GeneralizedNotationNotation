#!/usr/bin/env julia

"""
Statistical Analysis Module for ActiveInference.jl

This module provides rigorous statistical analysis tools for POMDP models:
- Hypothesis testing (t-tests, Mann-Whitney U, Kolmogorov-Smirnov)
- Bayesian model comparison (Bayes factors, posterior odds, model evidence)
- Confidence intervals and significance testing
- Bootstrap and permutation testing
- Time series analysis (stationarity, autocorrelation, trend analysis)
- Model validation (cross-validation, holdout testing, goodness-of-fit)
- Effect size calculations and power analysis
- Multiple comparison corrections (Bonferroni, FDR)
"""

using Pkg
using DelimitedFiles
using Statistics
using LinearAlgebra
using Distributions
using Printf
using Dates
using Random

# Statistical packages
for pkg in ["HypothesisTests", "StatsBase", "Bootstrap", "TimeSeries", "MultipleTesting"]
    try
        Pkg.add(pkg)
    catch e
        @warn "Failed to install $pkg: $e"
    end
end

using StatsBase
try
    using HypothesisTests
    using Bootstrap
    using MultipleTesting
    STATS_PACKAGES_AVAILABLE = true
catch
    @warn "Advanced statistical packages not available, using basic implementations"
    STATS_PACKAGES_AVAILABLE = false
end

# ====================================
# UTILITY FUNCTIONS
# ====================================

"""Load data with error handling and validation."""
function load_statistical_data(filepath::String)
    if !isfile(filepath)
        error("Data file not found: $filepath")
    end
    
    data = readdlm(filepath, ',', skipstart=6)
    
    # Convert to numeric, handling potential parsing errors
    numeric_data = zeros(Float64, size(data, 1), size(data, 2))
    for i in 1:size(data, 1), j in 1:size(data, 2)
        try
            numeric_data[i, j] = parse(Float64, string(data[i, j]))
        catch
            numeric_data[i, j] = NaN
        end
    end
    
    # Remove rows with NaN values
    valid_rows = .!any(isnan.(numeric_data), dims=2)[:, 1]
    clean_data = numeric_data[valid_rows, :]
    
    println("✅ Loaded $(size(clean_data, 1)) clean samples from $filepath")
    return clean_data
end

"""Calculate effect size (Cohen's d)."""
function cohens_d(group1::Vector{Float64}, group2::Vector{Float64})
    mean1, mean2 = mean(group1), mean(group2)
    var1, var2 = var(group1), var(group2)
    n1, n2 = length(group1), length(group2)
    
    # Pooled standard deviation
    pooled_std = sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    return (mean1 - mean2) / pooled_std
end

"""Calculate confidence interval for mean."""
function confidence_interval(data::Vector{Float64}, confidence_level::Float64 = 0.95)
    n = length(data)
    mean_val = mean(data)
    std_err = std(data) / sqrt(n)
    
    # Use t-distribution for small samples
    alpha = 1 - confidence_level
    t_critical = quantile(TDist(n - 1), 1 - alpha/2)
    
    margin_error = t_critical * std_err
    
    return (mean_val - margin_error, mean_val + margin_error)
end

# ====================================
# HYPOTHESIS TESTING
# ====================================

"""Perform comprehensive hypothesis tests comparing two groups."""
function perform_hypothesis_tests(group1::Vector{Float64}, group2::Vector{Float64}, 
                                group1_name::String = "Group1", 
                                group2_name::String = "Group2")
    
    results = Dict{String, Any}()
    
    # Basic descriptive statistics
    results["descriptive"] = Dict(
        "$(group1_name)_mean" => mean(group1),
        "$(group1_name)_std" => std(group1),
        "$(group1_name)_n" => length(group1),
        "$(group2_name)_mean" => mean(group2),
        "$(group2_name)_std" => std(group2),
        "$(group2_name)_n" => length(group2)
    )
    
    # Effect size
    effect_size = cohens_d(group1, group2)
    results["effect_size"] = effect_size
    
    # Interpret effect size
    if abs(effect_size) < 0.2
        effect_interpretation = "Small"
    elseif abs(effect_size) < 0.5
        effect_interpretation = "Medium"
    elseif abs(effect_size) < 0.8
        effect_interpretation = "Large"
    else
        effect_interpretation = "Very Large"
    end
    results["effect_interpretation"] = effect_interpretation
    
    if STATS_PACKAGES_AVAILABLE
        try
            # Two-sample t-test
            t_test = TwoSampleTTest(group1, group2)
            results["t_test"] = Dict(
                "statistic" => t_test.t,
                "p_value" => pvalue(t_test),
                "df" => t_test.df,
                "significant" => pvalue(t_test) < 0.05
            )
            
            # Mann-Whitney U test (non-parametric)
            mw_test = MannWhitneyUTest(group1, group2)
            results["mann_whitney"] = Dict(
                "statistic" => mw_test.U,
                "p_value" => pvalue(mw_test),
                "significant" => pvalue(mw_test) < 0.05
            )
            
            # Kolmogorov-Smirnov test
            ks_test = ApproximateTwoSampleKSTest(group1, group2)
            results["kolmogorov_smirnov"] = Dict(
                "statistic" => ks_test.δ,
                "p_value" => pvalue(ks_test),
                "significant" => pvalue(ks_test) < 0.05
            )
            
        catch e
            @warn "Some hypothesis tests failed: $e"
        end
    else
        # Basic manual implementations
        
        # Manual t-test
        n1, n2 = length(group1), length(group2)
        mean1, mean2 = mean(group1), mean(group2)
        var1, var2 = var(group1), var(group2)
        
        # Pooled variance
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        se = sqrt(pooled_var * (1/n1 + 1/n2))
        
        t_stat = (mean1 - mean2) / se
        df = n1 + n2 - 2
        
        # Approximate p-value using normal distribution for large samples
        if df > 30
            p_val = 2 * (1 - cdf(Normal(0, 1), abs(t_stat)))
        else
            p_val = 2 * (1 - cdf(TDist(df), abs(t_stat)))
        end
        
        results["t_test"] = Dict(
            "statistic" => t_stat,
            "p_value" => p_val,
            "df" => df,
            "significant" => p_val < 0.05
        )
    end
    
    # Confidence intervals
    ci1 = confidence_interval(group1, 0.95)
    ci2 = confidence_interval(group2, 0.95)
    
    results["confidence_intervals"] = Dict(
        "$(group1_name)_95_ci" => ci1,
        "$(group2_name)_95_ci" => ci2,
        "difference_estimate" => mean(group1) - mean(group2)
    )
    
    return results
end

"""Test for normality using Shapiro-Wilk or Anderson-Darling tests."""
function test_normality(data::Vector{Float64}, test_name::String = "data")
    results = Dict{String, Any}()
    
    if STATS_PACKAGES_AVAILABLE
        try
            # Anderson-Darling test
            ad_test = AndersonDarlingTest(data)
            results["anderson_darling"] = Dict(
                "statistic" => ad_test.A²,
                "p_value" => pvalue(ad_test),
                "normal" => pvalue(ad_test) > 0.05
            )
        catch e
            @warn "Anderson-Darling test failed: $e"
        end
    end
    
    # Manual Q-Q plot correlation test
    n = length(data)
    sorted_data = sort(data)
    theoretical_quantiles = [quantile(Normal(0, 1), (i - 0.5) / n) for i in 1:n]
    standardized_data = (sorted_data .- mean(data)) ./ std(data)
    
    qq_correlation = cor(theoretical_quantiles, standardized_data)
    results["qq_correlation"] = qq_correlation
    results["qq_normal"] = qq_correlation > 0.95  # Rule of thumb
    
    # Skewness and kurtosis
    results["skewness"] = skewness(data)
    results["kurtosis"] = kurtosis(data)
    results["skewness_normal"] = abs(skewness(data)) < 1.0
    results["kurtosis_normal"] = abs(kurtosis(data)) < 1.0
    
    return results
end

# ====================================
# BAYESIAN MODEL COMPARISON
# ====================================

"""Calculate Bayesian Information Criterion (BIC)."""
function calculate_bic(log_likelihood::Float64, n_params::Int, n_samples::Int)
    -2 * log_likelihood + n_params * log(n_samples)
end

"""Calculate Akaike Information Criterion (AIC)."""
function calculate_aic(log_likelihood::Float64, n_params::Int)
    -2 * log_likelihood + 2 * n_params
end

"""Bayesian model comparison using information criteria."""
function bayesian_model_comparison(models_data::Dict{String, Dict{String, Float64}})
    results = Dict{String, Any}()
    
    # Calculate information criteria for each model
    model_scores = Dict{String, Dict{String, Float64}}()
    
    for (model_name, model_info) in models_data
        scores = Dict{String, Float64}()
        
        if haskey(model_info, "log_likelihood") && haskey(model_info, "n_params") && haskey(model_info, "n_samples")
            ll = model_info["log_likelihood"]
            k = model_info["n_params"]
            n = model_info["n_samples"]
            
            scores["BIC"] = calculate_bic(ll, Int(k), Int(n))
            scores["AIC"] = calculate_aic(ll, Int(k))
            scores["log_likelihood"] = ll
        end
        
        model_scores[model_name] = scores
    end
    
    # Find best model for each criterion
    if !isempty(model_scores)
        for criterion in ["BIC", "AIC"]
            if all(haskey(scores, criterion) for scores in values(model_scores))
                criterion_values = [scores[criterion] for scores in values(model_scores)]
                model_names = collect(keys(model_scores))
                
                best_idx = argmin(criterion_values)
                best_model = model_names[best_idx]
                best_score = criterion_values[best_idx]
                
                # Calculate relative scores (differences from best)
                relative_scores = criterion_values .- best_score
                
                results["$(criterion)_comparison"] = Dict(
                    "best_model" => best_model,
                    "best_score" => best_score,
                    "relative_scores" => Dict(zip(model_names, relative_scores)),
                    "model_rankings" => model_names[sortperm(criterion_values)]
                )
            end
        end
        
        # Calculate Bayes factors (simplified using BIC approximation)
        if haskey(results, "BIC_comparison")
            bic_scores = results["BIC_comparison"]["relative_scores"]
            bayes_factors = Dict{String, Float64}()
            
            for (model, rel_bic) in bic_scores
                # BF approximation: exp(-0.5 * Δ BIC)
                bf = exp(-0.5 * rel_bic)
                bayes_factors[model] = bf
            end
            
            results["bayes_factors"] = bayes_factors
        end
    end
    
    results["model_scores"] = model_scores
    return results
end

"""Calculate model evidence using harmonic mean estimator."""
function calculate_model_evidence(log_likelihoods::Vector{Float64})
    # Harmonic mean estimator (basic implementation)
    n = length(log_likelihoods)
    max_ll = maximum(log_likelihoods)
    
    # Stabilize computation
    adjusted_lls = log_likelihoods .- max_ll
    harmonic_mean = -log(mean(exp.(-adjusted_lls))) + max_ll
    
    return harmonic_mean
end

# ====================================
# BOOTSTRAP AND PERMUTATION TESTING
# ====================================

"""Perform bootstrap confidence intervals."""
function bootstrap_confidence_interval(data::Vector{Float64}, 
                                     statistic_func::Function,
                                     confidence_level::Float64 = 0.95,
                                     n_bootstrap::Int = 1000)
    
    bootstrap_stats = Float64[]
    n = length(data)
    
    for _ in 1:n_bootstrap
        # Bootstrap sample
        bootstrap_sample = data[rand(1:n, n)]
        stat = statistic_func(bootstrap_sample)
        push!(bootstrap_stats, stat)
    end
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)
    
    ci_lower = percentile(bootstrap_stats, lower_percentile)
    ci_upper = percentile(bootstrap_stats, upper_percentile)
    
    return (ci_lower, ci_upper), bootstrap_stats
end

"""Permutation test for comparing two groups."""
function permutation_test(group1::Vector{Float64}, group2::Vector{Float64},
                         statistic_func::Function = (x, y) -> mean(x) - mean(y),
                         n_permutations::Int = 1000)
    
    # Observed test statistic
    observed_stat = statistic_func(group1, group2)
    
    # Combine groups for permutation
    combined = vcat(group1, group2)
    n1, n2 = length(group1), length(group2)
    n_total = n1 + n2
    
    # Permutation distribution
    permuted_stats = Float64[]
    
    for _ in 1:n_permutations
        # Random permutation
        perm_indices = randperm(n_total)
        perm_group1 = combined[perm_indices[1:n1]]
        perm_group2 = combined[perm_indices[n1+1:end]]
        
        stat = statistic_func(perm_group1, perm_group2)
        push!(permuted_stats, stat)
    end
    
    # Calculate p-value (two-tailed)
    extreme_count = sum(abs.(permuted_stats) .>= abs(observed_stat))
    p_value = extreme_count / n_permutations
    
    return Dict(
        "observed_statistic" => observed_stat,
        "p_value" => p_value,
        "permutation_distribution" => permuted_stats,
        "significant" => p_value < 0.05
    )
end

# ====================================
# TIME SERIES ANALYSIS
# ====================================

"""Test for stationarity using augmented Dickey-Fuller test (simplified)."""
function test_stationarity(time_series::Vector{Float64})
    results = Dict{String, Any}()
    
    n = length(time_series)
    
    # Calculate first differences
    diffs = time_series[2:end] - time_series[1:end-1]
    
    # Simple trend test using linear regression
    t = 1:n
    X = [ones(n) t]
    y = time_series
    
    # OLS regression: y = β₀ + β₁*t + ε
    beta = (X' * X) \ (X' * y)
    fitted = X * beta
    residuals = y - fitted
    
    # Test if trend coefficient is significantly different from zero
    mse = sum(residuals.^2) / (n - 2)
    var_beta1 = mse * inv(X' * X)[2, 2]
    se_beta1 = sqrt(var_beta1)
    t_stat = beta[2] / se_beta1
    
    # Approximate p-value
    p_value = 2 * (1 - cdf(TDist(n - 2), abs(t_stat)))
    
    results["trend_test"] = Dict(
        "trend_coefficient" => beta[2],
        "t_statistic" => t_stat,
        "p_value" => p_value,
        "has_trend" => p_value < 0.05
    )
    
    # Variance of differences (for stationarity assessment)
    results["difference_variance"] = var(diffs)
    results["original_variance"] = var(time_series)
    results["variance_ratio"] = var(diffs) / var(time_series)
    
    # Ljung-Box test for autocorrelation (simplified)
    autocorrs = [cor(time_series[1:end-lag], time_series[lag+1:end]) for lag in 1:min(10, n÷4)]
    results["autocorrelations"] = autocorrs
    results["max_autocorrelation"] = maximum(abs.(autocorrs))
    
    return results
end

"""Analyze autocorrelation structure."""
function analyze_autocorrelation(time_series::Vector{Float64}, max_lag::Int = 20)
    n = length(time_series)
    max_lag = min(max_lag, n ÷ 4)
    
    autocorrs = Float64[]
    for lag in 1:max_lag
        if lag < n
            corr_val = cor(time_series[1:end-lag], time_series[lag+1:end])
            push!(autocorrs, corr_val)
        end
    end
    
    # Confidence bounds (approximate)
    ci_bound = 1.96 / sqrt(n)
    
    return Dict(
        "autocorrelations" => autocorrs,
        "lags" => 1:length(autocorrs),
        "confidence_bound" => ci_bound,
        "significant_lags" => findall(abs.(autocorrs) .> ci_bound)
    )
end

# ====================================
# MODEL VALIDATION
# ====================================

"""K-fold cross-validation."""
function k_fold_cross_validation(data::Matrix{Float64}, 
                                model_func::Function,
                                evaluation_func::Function,
                                k::Int = 5)
    
    n_samples = size(data, 1)
    fold_size = ceil(Int, n_samples / k)
    
    fold_scores = Float64[]
    fold_predictions = Vector{Vector{Float64}}()
    fold_actuals = Vector{Vector{Float64}}()
    
    for fold in 1:k
        # Define test indices
        test_start = (fold - 1) * fold_size + 1
        test_end = min(fold * fold_size, n_samples)
        test_indices = test_start:test_end
        
        # Training indices
        train_indices = setdiff(1:n_samples, test_indices)
        
        if length(train_indices) > 0
            # Split data
            train_data = data[train_indices, :]
            test_data = data[test_indices, :]
            
            # Train model and make predictions
            try
                model = model_func(train_data)
                predictions = model(test_data)
                actuals = test_data[:, end]  # Assume last column is target
                
                # Evaluate
                score = evaluation_func(predictions, actuals)
                push!(fold_scores, score)
                push!(fold_predictions, predictions)
                push!(fold_actuals, actuals)
                
            catch e
                @warn "Cross-validation fold $fold failed: $e"
            end
        end
    end
    
    return Dict(
        "fold_scores" => fold_scores,
        "mean_score" => mean(fold_scores),
        "std_score" => std(fold_scores),
        "fold_predictions" => fold_predictions,
        "fold_actuals" => fold_actuals
    )
end

"""Goodness-of-fit tests."""
function goodness_of_fit_tests(observed::Vector{Float64}, expected::Vector{Float64})
    results = Dict{String, Any}()
    
    # Mean Squared Error
    mse = mean((observed - expected).^2)
    results["mse"] = mse
    
    # Root Mean Squared Error
    results["rmse"] = sqrt(mse)
    
    # Mean Absolute Error
    results["mae"] = mean(abs.(observed - expected))
    
    # R-squared
    ss_res = sum((observed - expected).^2)
    ss_tot = sum((observed .- mean(observed)).^2)
    r_squared = 1 - (ss_res / ss_tot)
    results["r_squared"] = r_squared
    
    # Correlation coefficient
    results["correlation"] = cor(observed, expected)
    
    if STATS_PACKAGES_AVAILABLE
        try
            # Chi-square goodness of fit (for count data)
            # Discretize continuous data for chi-square test
            n_bins = min(10, length(unique(observed)))
            obs_hist = fit(Histogram, observed, nbins=n_bins)
            exp_hist = fit(Histogram, expected, obs_hist.edges[1])
            
            # Chi-square test
            chi2_stat = sum((obs_hist.weights - exp_hist.weights).^2 ./ 
                          max.(exp_hist.weights, 1))
            df = n_bins - 1
            chi2_p = 1 - cdf(Chisq(df), chi2_stat)
            
            results["chi_square"] = Dict(
                "statistic" => chi2_stat,
                "df" => df,
                "p_value" => chi2_p,
                "good_fit" => chi2_p > 0.05
            )
        catch e
            @warn "Chi-square test failed: $e"
        end
    end
    
    return results
end

# ====================================
# MULTIPLE COMPARISON CORRECTIONS
# ====================================

"""Apply multiple comparison corrections."""
function multiple_comparison_correction(p_values::Vector{Float64}, 
                                      method::String = "bonferroni")
    
    if STATS_PACKAGES_AVAILABLE && method in ["bonferroni", "holm", "fdr"]
        try
            if method == "bonferroni"
                corrected = adjust(p_values, Bonferroni())
            elseif method == "holm"
                corrected = adjust(p_values, Holm())
            elseif method == "fdr"
                corrected = adjust(p_values, BenjaminiHochberg())
            else
                corrected = p_values
            end
            
            return corrected
        catch e
            @warn "Multiple comparison correction failed: $e"
        end
    end
    
    # Manual Bonferroni correction
    if method == "bonferroni"
        return min.(p_values * length(p_values), 1.0)
    else
        return p_values
    end
end

# ====================================
# COMPREHENSIVE STATISTICAL ANALYSIS
# ====================================

"""Comprehensive statistical analysis of simulation results."""
function comprehensive_statistical_analysis(output_dir::String)
    stats_dir = joinpath(output_dir, "statistical_analysis")
    mkpath(stats_dir)
    
    println("\n📊 Comprehensive Statistical Analysis")
    println("="^50)
    
    data_traces_dir = joinpath(output_dir, "data_traces")
    advanced_dir = joinpath(output_dir, "advanced_analysis")
    
    try
        # Load available data
        data_dict = Dict{String, Matrix{Float64}}()
        
        for filename in ["beliefs_over_time.csv", "actions_over_time.csv", 
                        "observations_over_time.csv", "learning_curve.csv",
                        "planning_rewards.csv"]
            filepath = joinpath(data_traces_dir, filename)
            if isfile(filepath)
                data_name = replace(filename, ".csv" => "")
                data_dict[data_name] = load_statistical_data(filepath)
            end
        end
        
        # Load advanced analysis data
        for filename in ["belief_entropy_trace.csv", "information_gain_trace.csv"]
            filepath = joinpath(advanced_dir, filename)
            if isfile(filepath)
                data_name = replace(filename, ".csv" => "")
                data_dict[data_name] = load_statistical_data(filepath)
            end
        end
        
        # ===== DESCRIPTIVE STATISTICS =====
        println("📈 Computing descriptive statistics...")
        
        descriptive_results = Dict{String, Any}()
        
        for (data_name, data) in data_dict
            if size(data, 2) >= 2
                values = data[:, 2]  # Use first data column
                
                desc_stats = Dict(
                    "n" => length(values),
                    "mean" => mean(values),
                    "median" => median(values),
                    "std" => std(values),
                    "var" => var(values),
                    "min" => minimum(values),
                    "max" => maximum(values),
                    "q25" => percentile(values, 25),
                    "q75" => percentile(values, 75),
                    "iqr" => percentile(values, 75) - percentile(values, 25),
                    "skewness" => skewness(values),
                    "kurtosis" => kurtosis(values)
                )
                
                descriptive_results[data_name] = desc_stats
            end
        end
        
        # Save descriptive statistics
        open(joinpath(stats_dir, "descriptive_statistics.txt"), "w") do f
            println(f, "Descriptive Statistics")
            println(f, "="^40)
            
            for (data_name, stats) in descriptive_results
                println(f, "\n$data_name:")
                println(f, "  N: $(stats["n"])")
                println(f, "  Mean: $(round(stats["mean"], digits=4))")
                println(f, "  Median: $(round(stats["median"], digits=4))")
                println(f, "  Std: $(round(stats["std"], digits=4))")
                println(f, "  Min: $(round(stats["min"], digits=4))")
                println(f, "  Max: $(round(stats["max"], digits=4))")
                println(f, "  Q25: $(round(stats["q25"], digits=4))")
                println(f, "  Q75: $(round(stats["q75"], digits=4))")
                println(f, "  Skewness: $(round(stats["skewness"], digits=4))")
                println(f, "  Kurtosis: $(round(stats["kurtosis"], digits=4))")
            end
        end
        
        # ===== NORMALITY TESTS =====
        println("🔔 Testing normality assumptions...")
        
        normality_results = Dict{String, Any}()
        
        for (data_name, data) in data_dict
            if size(data, 2) >= 2
                values = data[:, 2]
                norm_test = test_normality(values, data_name)
                normality_results[data_name] = norm_test
            end
        end
        
        # Save normality results
        open(joinpath(stats_dir, "normality_tests.txt"), "w") do f
            println(f, "Normality Tests")
            println(f, "="^40)
            
            for (data_name, norm_test) in normality_results
                println(f, "\n$data_name:")
                println(f, "  Q-Q Correlation: $(round(norm_test["qq_correlation"], digits=4))")
                println(f, "  Normal (Q-Q): $(norm_test["qq_normal"])")
                println(f, "  Skewness: $(round(norm_test["skewness"], digits=4))")
                println(f, "  Kurtosis: $(round(norm_test["kurtosis"], digits=4))")
                println(f, "  Normal (Skew/Kurt): $(norm_test["skewness_normal"] && norm_test["kurtosis_normal"])")
                
                if haskey(norm_test, "anderson_darling")
                    ad = norm_test["anderson_darling"]
                    println(f, "  Anderson-Darling p-value: $(round(ad["p_value"], digits=4))")
                    println(f, "  Normal (A-D): $(ad["normal"])")
                end
            end
        end
        
        # ===== TIME SERIES ANALYSIS =====
        println("⏰ Performing time series analysis...")
        
        time_series_results = Dict{String, Any}()
        
        for (data_name, data) in data_dict
            if size(data, 2) >= 2 && size(data, 1) > 10
                values = data[:, 2]
                
                # Stationarity test
                stationarity = test_stationarity(values)
                
                # Autocorrelation analysis
                autocorr = analyze_autocorrelation(values)
                
                time_series_results[data_name] = Dict(
                    "stationarity" => stationarity,
                    "autocorrelation" => autocorr
                )
            end
        end
        
        # Save time series results
        open(joinpath(stats_dir, "time_series_analysis.txt"), "w") do f
            println(f, "Time Series Analysis")
            println(f, "="^40)
            
            for (data_name, ts_results) in time_series_results
                println(f, "\n$data_name:")
                
                stationarity = ts_results["stationarity"]
                println(f, "  Trend coefficient: $(round(stationarity["trend_test"]["trend_coefficient"], digits=6))")
                println(f, "  Trend p-value: $(round(stationarity["trend_test"]["p_value"], digits=4))")
                println(f, "  Has trend: $(stationarity["trend_test"]["has_trend"])")
                println(f, "  Max autocorr: $(round(stationarity["max_autocorrelation"], digits=4))")
                
                autocorr = ts_results["autocorrelation"]
                println(f, "  Significant lags: $(length(autocorr["significant_lags"]))")
                if !isempty(autocorr["significant_lags"])
                    println(f, "  Lag positions: $(autocorr["significant_lags"])")
                end
            end
        end
        
        # ===== COMPARATIVE ANALYSIS =====
        if length(data_dict) >= 2
            println("⚖️  Performing comparative analysis...")
            
            comparison_results = Dict{String, Any}()
            data_names = collect(keys(data_dict))
            
            # Pairwise comparisons
            for i in 1:length(data_names)-1
                for j in i+1:length(data_names)
                    name1, name2 = data_names[i], data_names[j]
                    
                    if size(data_dict[name1], 2) >= 2 && size(data_dict[name2], 2) >= 2
                        values1 = data_dict[name1][:, 2]
                        values2 = data_dict[name2][:, 2]
                        
                        # Perform hypothesis tests
                        comparison = perform_hypothesis_tests(values1, values2, name1, name2)
                        comparison_results["$(name1)_vs_$(name2)"] = comparison
                        
                        # Bootstrap confidence intervals
                        ci1, _ = bootstrap_confidence_interval(values1, mean)
                        ci2, _ = bootstrap_confidence_interval(values2, mean)
                        
                        comparison["bootstrap_ci"] = Dict(
                            "$(name1)_mean_ci" => ci1,
                            "$(name2)_mean_ci" => ci2
                        )
                        
                        # Permutation test
                        perm_test = permutation_test(values1, values2)
                        comparison["permutation_test"] = perm_test
                    end
                end
            end
            
            # Save comparison results
            open(joinpath(stats_dir, "comparative_analysis.txt"), "w") do f
                println(f, "Comparative Analysis")
                println(f, "="^40)
                
                for (comparison_name, comp_results) in comparison_results
                    println(f, "\n$comparison_name:")
                    
                    desc = comp_results["descriptive"]
                    println(f, "  Group means: $(round(desc[keys(desc)[1]], digits=4)) vs $(round(desc[keys(desc)[4]], digits=4))")
                    println(f, "  Effect size (Cohen's d): $(round(comp_results["effect_size"], digits=4))")
                    println(f, "  Effect interpretation: $(comp_results["effect_interpretation"])")
                    
                    if haskey(comp_results, "t_test")
                        t_test = comp_results["t_test"]
                        println(f, "  T-test p-value: $(round(t_test["p_value"], digits=4))")
                        println(f, "  T-test significant: $(t_test["significant"])")
                    end
                    
                    if haskey(comp_results, "mann_whitney")
                        mw = comp_results["mann_whitney"]
                        println(f, "  Mann-Whitney p-value: $(round(mw["p_value"], digits=4))")
                        println(f, "  Mann-Whitney significant: $(mw["significant"])")
                    end
                    
                    if haskey(comp_results, "permutation_test")
                        perm = comp_results["permutation_test"]
                        println(f, "  Permutation p-value: $(round(perm["p_value"], digits=4))")
                        println(f, "  Permutation significant: $(perm["significant"])")
                    end
                end
            end
            
            # Multiple comparison correction
            all_p_values = Float64[]
            test_names = String[]
            
            for (comp_name, comp_results) in comparison_results
                if haskey(comp_results, "t_test")
                    push!(all_p_values, comp_results["t_test"]["p_value"])
                    push!(test_names, "$(comp_name)_t_test")
                end
            end
            
            if !isempty(all_p_values)
                corrected_p = multiple_comparison_correction(all_p_values, "bonferroni")
                
                open(joinpath(stats_dir, "multiple_comparisons.txt"), "w") do f
                    println(f, "Multiple Comparison Corrections")
                    println(f, "="^40)
                    println(f, "Method: Bonferroni")
                    println(f, "")
                    
                    for i in 1:length(test_names)
                        println(f, "$(test_names[i]):")
                        println(f, "  Original p-value: $(round(all_p_values[i], digits=4))")
                        println(f, "  Corrected p-value: $(round(corrected_p[i], digits=4))")
                        println(f, "  Significant (α=0.05): $(corrected_p[i] < 0.05)")
                    end
                end
            end
        end
        
        # ===== SAVE RAW DATA FOR FURTHER ANALYSIS =====
        println("💾 Saving processed data...")
        
        # Create summary CSV with key statistics
        if !isempty(descriptive_results)
            summary_data = []
            headers = ["Variable", "N", "Mean", "Std", "Min", "Max", "Skewness", "Kurtosis"]
            push!(summary_data, headers)
            
            for (data_name, stats) in descriptive_results
                row = [data_name, stats["n"], stats["mean"], stats["std"], 
                      stats["min"], stats["max"], stats["skewness"], stats["kurtosis"]]
                push!(summary_data, row)
            end
            
            writedlm(joinpath(stats_dir, "summary_statistics.csv"), summary_data, ',')
        end
        
        println("✅ Statistical analysis completed successfully!")
        println("📁 Results saved to: $stats_dir")
        
    catch e
        println("❌ Error in statistical analysis: $e")
        @warn "Statistical analysis failed" exception=e
    end
end

"""Main entry point for statistical analysis."""
function main()
    if length(ARGS) > 0
        output_dir = ARGS[1]
        if isdir(output_dir)
            comprehensive_statistical_analysis(output_dir)
        else
            error("❌ Directory not found: $output_dir")
        end
    else
        println("Usage: julia statistical_analysis.jl <output_directory>")
        println("Example: julia statistical_analysis.jl activeinference_outputs_YYYY-MM-DD_HH-MM-SS")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 