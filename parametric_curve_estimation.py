import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution


# ============================================================
# 1. Load Dataset
# ============================================================

file_path = "xy_data.csv"

df = pd.read_csv(file_path)

x_data = df["x"].to_numpy(dtype=float)
y_data = df["y"].to_numpy(dtype=float)

print("Dataset loaded successfully.")
print("Number of observations:", len(df))
print("Columns:", df.columns.tolist())


# ============================================================
# 2. Mathematical Transformation
# ============================================================

def transformed_values(params, x, y):
    """
    Transform observed (x, y) coordinates into latent
    t and A values for a candidate parameter vector.

    Parameters:
        params = [theta_deg, M, X]
    """

    theta_deg, M, X = params
    theta = np.deg2rad(theta_deg)

    # Since 6 < t < 60, t is positive and |t| = t.
    t = (
        (x - X) * np.cos(theta)
        + (y - 42) * np.sin(theta)
    )

    A = (
        -(x - X) * np.sin(theta)
        + (y - 42) * np.cos(theta)
    )

    A_model = (
        np.exp(M * t)
        * np.sin(0.3 * t)
    )

    return t, A, A_model


# ============================================================
# 3. Objective Function
# ============================================================

def objective(params):
    """
    Objective function used for parameter estimation.

    The fitting error is based on the difference between
    transformed A values and the model
    e^(Mt) sin(0.3t).

    A penalty is applied when recovered t values
    fall outside 6 < t < 60.
    """

    t, A, A_model = transformed_values(
        params,
        x_data,
        y_data
    )

    residual = A - A_model

    data_error = np.mean(residual ** 2)

    lower_violation = np.maximum(6 - t, 0)
    upper_violation = np.maximum(t - 60, 0)

    penalty = np.mean(
        lower_violation ** 2
        + upper_violation ** 2
    )

    total_error = data_error + 1000 * penalty

    return total_error


# ============================================================
# 4. Parameter Constraints
# ============================================================

bounds = [
    (0.001, 49.999),       # theta
    (-0.049999, 0.049999), # M
    (0.001, 99.999)        # X
]


# ============================================================
# 5. Differential Evolution Optimization
# ============================================================

print("\nStarting optimization...")

result = differential_evolution(
    objective,
    bounds=bounds,
    seed=42,
    popsize=20,
    maxiter=1000,
    tol=1e-10,
    polish=True
)

theta_final = result.x[0]
M_final = result.x[1]
X_final = result.x[2]

print("Optimization completed.")


# ============================================================
# 6. Estimated Parameters
# ============================================================

print("\nEstimated Parameters")
print("--------------------")
print(f"theta = {theta_final:.10f} degrees")
print(f"M     = {M_final:.12f}")
print(f"X     = {X_final:.10f}")
print(f"Objective value = {result.fun:.12e}")


# ============================================================
# 7. Constraint Validation
# ============================================================

theta_valid = 0 < theta_final < 50
M_valid = -0.05 < M_final < 0.05
X_valid = 0 < X_final < 100

print("\nConstraint Validation")
print("---------------------")
print("0 < theta < 50     :", theta_valid)
print("-0.05 < M < 0.05   :", M_valid)
print("0 < X < 100        :", X_valid)

all_parameters_valid = (
    theta_valid
    and M_valid
    and X_valid
)

print(
    "All parameter constraints satisfied:",
    all_parameters_valid
)


# ============================================================
# 8. Recover Latent t Values
# ============================================================

t_recovered, A_recovered, A_model = transformed_values(
    [theta_final, M_final, X_final],
    x_data,
    y_data
)

print("\nRecovered t-Value Range")
print("-----------------------")
print(f"Minimum t = {t_recovered.min():.6f}")
print(f"Maximum t = {t_recovered.max():.6f}")

t_valid = np.all(
    (t_recovered > 6)
    & (t_recovered < 60)
)

print("All recovered t values satisfy 6 < t < 60:", t_valid)


# ============================================================
# 9. Original Parametric Curve
# ============================================================

def parametric_curve(t, theta_deg, M, X):
    """
    Generate x(t) and y(t) from the original
    parametric curve equations.
    """

    theta = np.deg2rad(theta_deg)

    x = (
        t * np.cos(theta)
        - np.exp(M * np.abs(t))
        * np.sin(0.3 * t)
        * np.sin(theta)
        + X
    )

    y = (
        42
        + t * np.sin(theta)
        + np.exp(M * np.abs(t))
        * np.sin(0.3 * t)
        * np.cos(theta)
    )

    return x, y


# ============================================================
# 10. Generate Reconstructed Curve
# ============================================================

t_smooth = np.linspace(
    6.000001,
    59.999999,
    5000
)

x_pred, y_pred = parametric_curve(
    t_smooth,
    theta_final,
    M_final,
    X_final
)


# ============================================================
# 11. Plot Observed Data and Reconstructed Curve
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    x_data,
    y_data,
    s=8,
    alpha=0.6,
    label="Observed Data"
)

plt.plot(
    x_pred,
    y_pred,
    linewidth=2,
    label="Reconstructed Curve"
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Observed Data vs Reconstructed Parametric Curve")
plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 12. Sort Observations by Recovered t
# ============================================================

sort_idx = np.argsort(t_recovered)

t_sorted = t_recovered[sort_idx]
x_sorted = x_data[sort_idx]
y_sorted = y_data[sort_idx]


# ============================================================
# 13. L1 Distance Evaluation
# ============================================================

t_eval_start = t_sorted.min()
t_eval_end = t_sorted.max()

N_final = 10000

t_final = np.linspace(
    t_eval_start,
    t_eval_end,
    N_final
)

x_final_pred, y_final_pred = parametric_curve(
    t_final,
    theta_final,
    M_final,
    X_final
)

# Interpolate observed data onto the same uniform t-grid.
x_final_expected = np.interp(
    t_final,
    t_sorted,
    x_sorted
)

y_final_expected = np.interp(
    t_final,
    t_sorted,
    y_sorted
)

# Pointwise L1 distance.
l1_distance = (
    np.abs(
        x_final_expected - x_final_pred
    )
    +
    np.abs(
        y_final_expected - y_final_pred
    )
)

total_l1 = np.sum(l1_distance)
mean_l1 = np.mean(l1_distance)
max_l1 = np.max(l1_distance)


# ============================================================
# 14. Final Results
# ============================================================

print("\nFINAL RESULTS")
print("=============")

print(f"Theta (degrees) : {theta_final:.10f}")
print(f"M               : {M_final:.12f}")
print(f"X               : {X_final:.10f}")

print("\nL1 Evaluation")
print("-------------")
print(f"Number of samples       : {N_final}")
print(
    f"Evaluation t-range      : "
    f"{t_eval_start:.6f} to {t_eval_end:.6f}"
)
print(f"Total L1 distance       : {total_l1:.12e}")
print(f"Mean L1 distance        : {mean_l1:.12e}")
print(f"Maximum pointwise L1    : {max_l1:.12e}")


# ============================================================
# 15. Pointwise L1 Error Plot
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    t_final,
    l1_distance,
    linewidth=1
)

plt.xlabel("t")
plt.ylabel("Pointwise L1 Error")
plt.title("Pointwise L1 Error Across the Parameter Range")
plt.grid(True)

plt.show()


# ============================================================
# 16. Final Parametric Equations
# ============================================================

print("\nFinal Parametric Curve")
print("----------------------")

print(
    "x(t) = t*cos(30°) "
    "- exp(0.03*|t|)*sin(0.3t)*sin(30°) + 55"
)

print(
    "y(t) = 42 + t*sin(30°) "
    "+ exp(0.03*|t|)*sin(0.3t)*cos(30°)"
)

print("Valid range: 6 < t < 60")
