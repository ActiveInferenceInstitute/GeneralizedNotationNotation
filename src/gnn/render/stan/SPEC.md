# Stan Renderer — Specification

## Entry point

`render_gnn_to_stan(gnn_spec: dict, output_path: Path, options: dict | None)
-> (success: bool, message: str, artifacts: list[str])`

`artifacts = [driver.py, program.stan]`, both under `output_path.parent`.

## Discrete program (HMM / POMDP)

```
data      T, S, O, U, o[T] (1-based), u[T-1] (1-based), B[U] matrix[S,S], alpha_A[S] vector[O], D simplex[S]
params    A_est[S] simplex[O]
tparams   filtered_state[T] vector[S], log_marginal  (scaled vectorised forward recursion)
model     A_est[s] ~ dirichlet(alpha_A[s]);  target += log_marginal
```

## Continuous program (LGSSM)

```
data      T, n, m, F, H, Q, R, prior_mean, prior_cov, y[T] vector[m], u[T] vector[n]
params    obs_noise_scale > 0
tparams   mu[T], P[T], log_lik  (Kalman predict/update with Joseph-form covariance)
model     obs_noise_scale ~ lognormal(0, 0.5); target += log_lik
```

## Driver (`*_stan.py`)

Simulates data with `numpy.random.default_rng(random_seed)`, writes
`<stem>_data.json` to `$STAN_OUTPUT_DIR`, compiles with
`cmdstanpy.CmdStanModel`; discrete models sample NUTS (`chains=1,
iter_warmup=300, iter_sampling=300`) when `S·O ≤ ModelParameters.
stan_nuts_param_budget` (default 1024) and otherwise run L-BFGS MAP
(`jacobian=True`, validation key `map_converged` instead of `rhat_ok`);
continuous models always sample NUTS. Writes `simulation_results.json`; exit 0 iff
`validation.all_valid`. Missing `cmdstanpy` → exit 1 with an install hint
(Step 12 skips before reaching this via `utils.framework_availability`).
