"""
surrogate_benchmarking.py

Experiment 1: Accuracy vs. training set size and problem dimensionality.

Fits Kriging, RBF (4 basis variants), NN (3 activation variants), and kNN
(4 distance metrics) on Latin Hypercube samples of increasing size, then
evaluates each on a fixed 300-point held-out test set.

Output: surrogate_results.csv
Columns: model, function, dim, n_train, rmse, train_time, [best_k for kNN]
"""

import numpy as np
import time
import pandas as pd
from smt.surrogate_models import KRG, RBF
from smt.sampling_methods import LHS
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


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


# ── Surrogate variant definitions ──────────────────────────────────────────────

# SMT RBF basis options: thin_plate_spline, gaussian, multiquadric, cubic
RBF_VARIANTS = {
    "RBF_thin_plate_spline": "thin_plate_spline",
    "RBF_gaussian":          "gaussian",
    "RBF_multiquadric":      "multiquadric",
    "RBF_cubic":             "cubic",
}

# sklearn MLPRegressor activation strings (sigmoid is called 'logistic' in sklearn)
NN_VARIANTS = {
    "NN_relu":    "relu",
    "NN_tanh":    "tanh",
    "NN_sigmoid": "logistic",
}

# (metric, p_val) — p_val is only passed for Minkowski variants
KNN_VARIANTS = {
    "kNN_euclidean": ("minkowski", 2),
    "kNN_manhattan": ("minkowski", 1),
    "kNN_chebyshev": ("chebyshev", None),
    "kNN_canberra":  ("canberra",  None),
}


# ── Experiment configuration ───────────────────────────────────────────────────
# To run a faster subset, swap in the Med or Tiny blocks below.

# Full
dims_to_test  = [2, 5, 10]
n_train_sizes = [20, 50, 100, 200, 500]
n_test        = 300

# Med (uncomment to use)
# dims_to_test  = [2, 5]
# n_train_sizes = [20, 50, 100]
# n_test        = 75

# Tiny (uncomment to use)
# dims_to_test  = [2]
# n_train_sizes = [20]
# n_test        = 50


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_knn(n_neighbors, metric, p_val):
    """Build a KNeighborsRegressor, omitting p for metrics that do not use it."""
    kwargs = {"n_neighbors": n_neighbors, "metric": metric}
    if p_val is not None:
        kwargs["p"] = p_val
    return KNeighborsRegressor(**kwargs)


def select_best_k(X_train, y_train, metric, p_val,
                  k_candidates=(1, 2, 3, 5, 7, 10, 15)):
    """Pick k via 5-fold cross-validation; returns the k with lowest mean RMSE."""
    n_train = len(y_train)
    n_folds = min(5, n_train // 2)
    best_k, best_score = 1, float("inf")

    for k in k_candidates:
        if k >= n_train:
            continue
        fold_size = n_train - (n_train // n_folds)
        if k >= fold_size:
            continue
        scores = -cross_val_score(
            make_knn(k, metric, p_val),
            X_train, y_train,
            cv=max(2, n_folds),
            scoring="neg_root_mean_squared_error",
        )
        if scores.mean() < best_score:
            best_score = scores.mean()
            best_k     = k

    return best_k


# ── Main loop ──────────────────────────────────────────────────────────────────

results = []

for dim in dims_to_test:
    for func_name, (func, bounds) in benchmarks.items():
        lo, hi  = bounds[0], bounds[1]
        xlimits = np.array([bounds] * dim, dtype=float)

        # Fixed test set — same for all training sizes within this (func, dim) cell
        X_test_raw = LHS(xlimits=xlimits)(n_test)
        y_test     = func(X_test_raw)
        X_test     = (X_test_raw - lo) / (hi - lo)

        for n_train in n_train_sizes:
            X_train_raw = LHS(xlimits=xlimits)(n_train)
            y_train     = func(X_train_raw)
            X_train     = (X_train_raw - lo) / (hi - lo)

            # ── Kriging ───────────────────────────────────────────────────────
            try:
                t0 = time.time()
                kg = KRG(print_global=False)
                kg.set_training_values(X_train, y_train.reshape(-1, 1))
                kg.train()
                t_train = time.time() - t0
                y_pred  = kg.predict_values(X_test).flatten()
                rmse    = np.sqrt(np.mean((y_pred - y_test) ** 2))
                results.append({"model": "Kriging", "function": func_name,
                                 "dim": dim, "n_train": n_train,
                                 "rmse": rmse, "train_time": t_train})
            except Exception as e:
                print(f"Kriging failed: {func_name} {dim}D n={n_train}: {e}")

            # ── RBF variants ──────────────────────────────────────────────────
            for model_label, basis in RBF_VARIANTS.items():
                try:
                    t0 = time.time()
                    try:
                        rb = RBF(print_global=False, function=basis)
                    except TypeError:
                        rb = RBF(print_global=False)
                        rb.options["function"] = basis
                    rb.set_training_values(X_train, y_train.reshape(-1, 1))
                    rb.train()
                    t_train = time.time() - t0
                    y_pred  = rb.predict_values(X_test).flatten()
                    rmse    = np.sqrt(np.mean((y_pred - y_test) ** 2))
                    results.append({"model": model_label, "function": func_name,
                                     "dim": dim, "n_train": n_train,
                                     "rmse": rmse, "train_time": t_train})
                except Exception as e:
                    print(f"{model_label} failed: {func_name} {dim}D n={n_train}: {e}")

            # ── Neural network variants ────────────────────────────────────────
            # Outputs are z-scored before fitting for numerical stability on
            # Rosenbrock and Rastrigin, then unscaled before computing RMSE.
            y_mean      = y_train.mean()
            y_std       = y_train.std(ddof=1) + 1e-8
            y_tr_scaled = (y_train - y_mean) / y_std

            x_scaler       = StandardScaler()
            X_train_scaled = x_scaler.fit_transform(X_train)
            X_test_scaled  = x_scaler.transform(X_test)

            for model_label, activation in NN_VARIANTS.items():
                try:
                    t0 = time.time()
                    nn = MLPRegressor(
                        hidden_layer_sizes=(64, 64),
                        activation=activation,
                        solver="adam",
                        learning_rate_init=1e-3,
                        max_iter=5000,
                        random_state=42,
                        early_stopping=False,
                        tol=1e-6,
                        n_iter_no_change=50,
                    )
                    nn.fit(X_train_scaled, y_tr_scaled)
                    t_train       = time.time() - t0
                    y_pred_scaled = nn.predict(X_test_scaled)
                    y_pred        = y_pred_scaled * y_std + y_mean
                    rmse          = np.sqrt(np.mean((y_pred - y_test) ** 2))
                    results.append({"model": model_label, "function": func_name,
                                     "dim": dim, "n_train": n_train,
                                     "rmse": rmse, "train_time": t_train})
                except Exception as e:
                    print(f"{model_label} failed: {func_name} {dim}D n={n_train}: {e}")

            # ── kNN variants ──────────────────────────────────────────────────
            for model_label, (metric, p_val) in KNN_VARIANTS.items():
                try:
                    best_k  = select_best_k(X_train, y_train, metric, p_val)
                    t0      = time.time()
                    knn     = make_knn(best_k, metric, p_val)
                    knn.fit(X_train, y_train)
                    t_train = time.time() - t0
                    y_pred  = knn.predict(X_test)
                    rmse    = np.sqrt(np.mean((y_pred - y_test) ** 2))
                    results.append({"model": model_label, "function": func_name,
                                     "dim": dim, "n_train": n_train,
                                     "rmse": rmse, "train_time": t_train,
                                     "best_k": best_k})
                except Exception as e:
                    print(f"{model_label} failed: {func_name} {dim}D n={n_train}: {e}")

            print(f"Done: {func_name} | {dim}D | n={n_train}")


# ── Save results ───────────────────────────────────────────────────────────────

df = pd.DataFrame(results)
df.to_csv("surrogate_results.csv", index=False)
print("Saved surrogate_results.csv")

summary = df.pivot_table(
    index=["function", "dim", "n_train"],
    columns="model",
    values="rmse",
).round(4)
print(summary)
