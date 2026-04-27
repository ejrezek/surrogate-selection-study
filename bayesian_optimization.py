"""
bayesian_optimization.py

Experiment 2: Bayesian optimization loop comparing four surrogate strategies.

Kriging and kNN use Expected Improvement (EI) as the acquisition function,
since both provide uncertainty estimates. RBF and NN use pure exploitation
(lowest predicted value) because they do not.

Reads surrogate_results.csv implicitly via the model choices made in the
benchmarking study. Outputs bo_results.csv.

Design notes:
  - Inputs are normalized to [0,1]^d using function bounds before fitting.
  - kNN uncertainty is the std of the k nearest neighbors' y values.
  - kNN metric is manhattan (best performer from benchmarking).
  - RBF uses multiquadric basis (best performer from benchmarking).
  - NN handles its own output scaling internally.
  - Set DEBUG = True for a quick sanity check before running the full grid.
"""

import numpy as np
import time
import pandas as pd
from scipy.stats import norm
from smt.surrogate_models import KRG, RBF
from smt.sampling_methods import LHS
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


# ── Debug toggle ───────────────────────────────────────────────────────────────
# Set to True to run dim=2, Rosenbrock only, 1 seed, 20 total evaluations.
DEBUG = False
# ──────────────────────────────────────────────────────────────────────────────


# ── Benchmark functions ────────────────────────────────────────────────────────

def rosenbrock(X):
    return np.sum(100 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (1 - X[:, :-1]) ** 2, axis=1)

def rastrigin(X):
    d = X.shape[1]
    return 10 * d + np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X), axis=1)

def ackley(X):
    d = X.shape[1]
    a, b, c = 20, 0.2, 2 * np.pi
    return (-a * np.exp(-b * np.sqrt(np.sum(X ** 2, axis=1) / d))
            - np.exp(np.sum(np.cos(c * X), axis=1) / d)
            + a + np.e)

def styblinski_tang(X):
    return 0.5 * np.sum(X ** 4 - 16 * X ** 2 + 5 * X, axis=1)

benchmarks = {
    "Rosenbrock":      (rosenbrock,      [-2,    2]),
    "Rastrigin":       (rastrigin,       [-5.12, 5.12]),
    "Ackley":          (ackley,          [-5,    5]),
    "Styblinski-Tang": (styblinski_tang, [-5,    5]),
}


# ── Acquisition strategy per surrogate ────────────────────────────────────────

ACQUISITION = {
    "Kriging":          "EI",
    "kNN_manhattan":    "EI",
    "RBF_multiquadric": "exploitation",
    "NN_relu":          "exploitation",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_knn(n_neighbors, metric, p_val):
    """Build a KNeighborsRegressor, omitting p for metrics that do not use it."""
    kwargs = {"n_neighbors": n_neighbors, "metric": metric}
    if p_val is not None:
        kwargs["p"] = p_val
    return KNeighborsRegressor(**kwargs)


def select_best_k(X_train, y_train, metric, p_val,
                  k_candidates=(1, 2, 3, 5, 7, 10, 15)):
    """Pick k via cross-validation; returns the k with lowest mean RMSE."""
    n_train = len(y_train)
    best_k, best_score = 1, float("inf")

    for k in k_candidates:
        if k >= n_train:
            continue
        n_folds   = max(2, min(5, n_train // k))
        fold_size = n_train - (n_train // n_folds)
        if k >= fold_size:
            continue
        scores = -cross_val_score(
            make_knn(k, metric, p_val),
            X_train, y_train,
            cv=n_folds,
            scoring="neg_root_mean_squared_error",
        )
        if scores.mean() < best_score:
            best_score = scores.mean()
            best_k     = k

    return best_k


def expected_improvement(mu, sigma, f_best):
    """
    EI(x) = (f_best - mu) * Phi(Z) + sigma * phi(Z), Z = (f_best - mu) / sigma.
    Points with sigma == 0 receive EI = 0.
    """
    ei   = np.zeros(len(mu))
    mask = sigma > 1e-10
    Z    = (f_best - mu[mask]) / sigma[mask]
    ei[mask] = (f_best - mu[mask]) * norm.cdf(Z) + sigma[mask] * norm.pdf(Z)
    return ei


# ── Surrogate wrappers ─────────────────────────────────────────────────────────

def kriging_predict(X_train, y_train, X_cand):
    kg = KRG(print_global=False)
    kg.set_training_values(X_train, y_train.reshape(-1, 1))
    kg.train()
    mu    = kg.predict_values(X_cand).flatten()
    sigma = np.sqrt(np.maximum(kg.predict_variances(X_cand).flatten(), 0.0))
    return mu, sigma


def knn_predict(X_train, y_train, X_cand, metric="minkowski", p_val=1):
    best_k = select_best_k(X_train, y_train, metric, p_val)
    knn    = make_knn(best_k, metric, p_val)
    knn.fit(X_train, y_train)
    mu               = knn.predict(X_cand)
    neighbor_indices = knn.kneighbors(X_cand, return_distance=False)
    sigma            = np.array([np.std(y_train[idx]) for idx in neighbor_indices])
    return mu, sigma


def rbf_predict(X_train, y_train, X_cand):
    rb = RBF(print_global=False)
    rb.set_training_values(X_train, y_train.reshape(-1, 1))
    rb.train()
    return rb.predict_values(X_cand).flatten()


def nn_predict(X_train, y_train, X_cand):
    """Scales outputs internally; returns predictions in original y units."""
    y_mean   = y_train.mean()
    y_std    = y_train.std(ddof=1) + 1e-8
    y_scaled = (y_train - y_mean) / y_std

    x_scaler       = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_cand_scaled  = x_scaler.transform(X_cand)

    nn = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=5000,
        random_state=42,
        early_stopping=False,
        tol=1e-6,
        n_iter_no_change=50,
    )
    nn.fit(X_train_scaled, y_scaled)
    return nn.predict(X_cand_scaled) * y_std + y_mean


# ── Single BO run ──────────────────────────────────────────────────────────────

def run_bo(func, bounds, dim, surrogate_name, seed,
           n_init=10, budget=50, n_candidates=500):
    """
    Run one full BO trial. Returns a list of dicts with one row per evaluation.
    Wall-clock time is cumulative from the start of the run.
    """
    acquisition = ACQUISITION[surrogate_name]
    rng     = np.random.default_rng(seed)
    lo, hi  = bounds[0], bounds[1]
    xlimits = np.array([bounds] * dim, dtype=float)

    np.random.seed(int(seed))
    X_init_raw = LHS(xlimits=xlimits)(n_init)
    X_init     = (X_init_raw - lo) / (hi - lo)
    y_init     = func(X_init_raw)

    X_obs = X_init.copy()
    y_obs = y_init.copy()

    rows        = []
    t_start     = time.time()
    best_so_far = y_obs.min()

    for i in range(n_init):
        best_so_far = min(best_so_far, y_obs[i])
        rows.append({
            "model":                   surrogate_name,
            "acquisition":             acquisition,
            "evaluation_number":       i + 1,
            "best_value_found":        best_so_far,
            "wall_clock_time_seconds": time.time() - t_start,
        })

    for ev in range(n_init + 1, budget + 1):
        y_mean   = y_obs.mean()
        y_std    = y_obs.std(ddof=1) + 1e-8
        y_scaled = (y_obs - y_mean) / y_std

        X_cand = rng.uniform(0, 1, size=(n_candidates, dim))

        try:
            if acquisition == "EI":
                f_best_s = (best_so_far - y_mean) / y_std

                if surrogate_name == "Kriging":
                    mu_s, sigma_s = kriging_predict(X_obs, y_scaled, X_cand)
                else:
                    mu_s, sigma_s = knn_predict(X_obs, y_scaled, X_cand)

                mu       = mu_s * y_std + y_mean
                sigma    = sigma_s * y_std
                scores   = expected_improvement(mu, sigma, best_so_far)
                best_idx = np.argmax(scores)

            else:
                if surrogate_name == "RBF_multiquadric":
                    mu_s     = rbf_predict(X_obs, y_scaled, X_cand)
                    mu       = mu_s * y_std + y_mean
                else:
                    mu       = nn_predict(X_obs, y_obs, X_cand)

                best_idx = np.argmin(mu)

        except Exception as e:
            print(f"  [warn] {surrogate_name} failed at ev={ev}: {e}. Falling back to random.")
            best_idx = rng.integers(n_candidates)

        x_new_norm = X_cand[[best_idx]]
        x_new_raw  = x_new_norm * (hi - lo) + lo
        y_new      = func(x_new_raw)[0]

        X_obs = np.vstack([X_obs, x_new_norm])
        y_obs = np.append(y_obs, y_new)

        best_so_far = min(best_so_far, y_new)
        rows.append({
            "model":                   surrogate_name,
            "acquisition":             acquisition,
            "evaluation_number":       ev,
            "best_value_found":        best_so_far,
            "wall_clock_time_seconds": time.time() - t_start,
        })

    return rows


# ── Experiment configuration ───────────────────────────────────────────────────

if DEBUG:
    dims_to_test  = [2]
    funcs_to_test = {"Rosenbrock": benchmarks["Rosenbrock"]}
    seeds         = [0]
    budget        = 20
    n_init        = 10
    print("DEBUG MODE: dim=2, Rosenbrock, 1 seed, 20 evaluations")
else:
    dims_to_test  = [2, 5, 10]
    funcs_to_test = benchmarks
    seeds         = list(range(5))
    budget        = 50
    n_init        = 10

surrogates = ["Kriging", "kNN_manhattan", "RBF_multiquadric", "NN_relu"]


# ── Main loop ──────────────────────────────────────────────────────────────────

all_rows = []

for dim in dims_to_test:
    for func_name, (func, bounds) in funcs_to_test.items():
        for surrogate_name in surrogates:
            for seed in seeds:
                print(f"Running: {surrogate_name:15s} | {func_name:15s} | dim={dim} | seed={seed}")
                rows = run_bo(
                    func           = func,
                    bounds         = bounds,
                    dim            = dim,
                    surrogate_name = surrogate_name,
                    seed           = seed,
                    n_init         = n_init,
                    budget         = budget,
                    n_candidates   = 500,
                )
                for r in rows:
                    r["function"] = func_name
                    r["dim"]      = dim
                    r["seed"]     = seed
                all_rows.extend(rows)


# ── Save results ───────────────────────────────────────────────────────────────

df = pd.DataFrame(all_rows)[
    ["model", "acquisition", "function", "dim", "seed",
     "evaluation_number", "best_value_found", "wall_clock_time_seconds"]
]

df.to_csv("bo_results.csv", index=False)
print("\nSaved bo_results.csv")
print(df.head(20).to_string(index=False))
