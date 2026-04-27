# surrogate-selection-study
Benchmarking Kriging, RBF, neural network, and kNN surrogates on accuracy, data requirements, and computational cost across standard optimization test functions.

# Surrogate Model Selection Under Budget Constraints

Repository for **"Surrogate Model Selection Under Budget Constraints: Accuracy, Data, and Computational Cost Trade-offs"**  
Group 14 — Elliot Rezek & Zack Brooks, NC State University

---

## Summary

This repository contains all code and results needed to reproduce the experiments in our paper. We evaluate four surrogate model families — Kriging (Gaussian process), RBF, neural network, and kNN regression — across four benchmark functions, three problem dimensionalities, and five training set sizes. The goal is to map the trade-off between accuracy, data requirements, and computational cost precisely enough to give practitioners better guidance than existing rules of thumb provide.

A key contribution is the first systematic evaluation of kNN regression as a primary surrogate in expensive black-box optimization, benchmarked against the three canonical alternatives.

---

## Experimental Setup

| Setting | Value |
|---|---|
| Benchmark functions | Rosenbrock, Rastrigin, Ackley, Styblinski-Tang |
| Dimensionalities | 2, 5, 10 |
| Training set sizes | 20, 50, 100, 200, 500 |
| Test set size | 300 (fixed LHS sample) |
| BO budget | 50 evaluations (10 initial LHS + 40 guided) |
| BO seeds | 5 independent seeds |
| Sampling method | Latin Hypercube Sampling (LHS) |

### Surrogates

| Model | Variants tested | Acquisition (BO) |
|---|---|---|
| Kriging | 1 (SMT default) | Expected Improvement |
| RBF | thin plate spline, gaussian, multiquadric, cubic | Pure exploitation |
| Neural Network | relu, tanh, sigmoid | Pure exploitation |
| kNN | euclidean, manhattan, chebyshev, canberra | Expected Improvement (manhattan only) |

---

## Reproducing the Results

All experiments were run on Google Colab (paid tier) using Python 3. Runtime for the full benchmarking grid is several hours; see the configuration options in each script to run a smaller subset first.

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Run Experiment 1 (Figures 4, 5, 6, 7)

Fits all surrogate variants across all (function, dimensionality, training size) combinations and saves RMSE and training time to `surrogate_results.csv`.

```bash
python surrogate_benchmarking.py
```

To run a faster subset before committing to the full grid, open `surrogate_benchmarking.py` and swap in the `Med` or `Tiny` configuration block near the top of the file.

### Step 3 — Run Experiment 2 (Figure 8)

Runs the Bayesian optimization loop for the four selected surrogate variants across all functions, dimensionalities, and seeds. Saves convergence data to `bo_results.csv`.

```bash
python bayesian_optimization.py
```

To run a quick sanity check first, set `DEBUG = True` at the top of `bayesian_optimization.py`. This runs dim=2, Rosenbrock only, 1 seed, 20 total evaluations.

---

## Output Files

| File | Description |
|---|---|
| `surrogate_results.csv` | RMSE and training time for all surrogate variants (Experiment 1) |
| `bo_results.csv` | Best value found and wall-clock time per evaluation (Experiment 2) |

---

## Figures

| Figure | What it shows |
|---|---|
| `plot1_learning_curves.pdf` | RMSE vs. training set size across all models, functions, and dimensionalities |
| `plot2_dim_scaling.pdf` | RMSE vs. problem dimensionality at fixed training sizes |
| `plot3_accuracy_cost.pdf` | Accuracy vs. training time scatter — the cost-accuracy frontier |
| `plot4_low_data.pdf` | Model comparison in the data-scarce regime (n = 20 and n = 50) |
| `bo_convergence_raster.pdf` | Bayesian optimization convergence curves, median across 5 seeds |

---

## Notes

- All inputs are normalized to [0, 1]^d using function domain bounds before fitting. Skipping this step causes numerical failures on Rosenbrock and Rastrigin at higher dimensions.
- kNN k is selected per condition via 5-fold cross-validation over k in {1, 2, 3, 5, 7, 10, 15}.
- NN outputs are z-scored before fitting and unscaled before evaluation.
- The scripts were originally developed in Google Colab. The `!pip install smt` magic command has been removed; install via `requirements.txt` instead.
