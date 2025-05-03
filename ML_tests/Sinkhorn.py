import numpy as np


def sinkhorn_primal(r, c, C, epsilon, num_iters=100, tol=1e-9, verbose=False):
    """
    Solves entropy-regularized OT using the Primal Sinkhorn algorithm.

    Args:
        r (np.ndarray): Source distribution (m,). Must sum to 1.
        c (np.ndarray): Target distribution (n,). Must sum to 1.
        C (np.ndarray): Cost matrix (m, n).
        epsilon (float): Regularization strength.
        num_iters (int): Maximum number of iterations.
        tol (float): Tolerance for convergence check.
        verbose (bool): Print convergence progress.

    Returns:
        tuple: (P, u, v, cost)
            P (np.ndarray): Optimal transport plan (m, n).
            u (np.ndarray): Primal scaling vector (m,).
            v (np.ndarray): Primal scaling vector (n,).
            cost (float): Regularized OT cost sum(P * C).
    """
    m, n = C.shape
    if not np.isclose(r.sum(), 1.0):
        print(f"Warning: Source distribution r sums to {r.sum()}, normalizing.")
        r = r / r.sum()
    if not np.isclose(c.sum(), 1.0):
        print(f"Warning: Target distribution c sums to {c.sum()}, normalizing.")
        c = c / c.sum()
    if r.shape[0] != m or c.shape[0] != n:
        raise ValueError("Shape mismatch between r, c, and C.")

    # Add small constant to prevent division by zero if r or c have zeros
    r_eps = r + 1e-100
    c_eps = c + 1e-100

    # Initialize scaling vector v
    v = np.ones(n)
    u = np.ones(m)  # Will be updated first

    # Calculate Gibbs Kernel K
    K = np.exp(-C / epsilon)
    if np.any(K == 0):
        print("Warning: K contains zeros. May lead to division by zero.")
        # Add small epsilon where K is zero to prevent immediate failure
        K[K == 0] = 1e-100

    if verbose:
        print(f"{'Iter':<5} | {'Change in v':<15}")
        print("-" * 25)

    for i in range(num_iters):
        v_prev = v.copy()

        # Update u
        Kv = K @ v
        # Add small epsilon to denominator for numerical stability
        u = r_eps / (Kv + 1e-100)

        # Update v
        KTu = K.T @ u
        # Add small epsilon to denominator for numerical stability
        v = c_eps / (KTu + 1e-100)

        # --- Convergence Check ---
        # Check change in v (or u). More robust checks involve marginal errors.
        change = np.linalg.norm(v - v_prev)
        if verbose and (i % 10 == 0 or i == num_iters - 1):
            print(f"{i:<5} | {change:<15.4e}")

        if change < tol:
            if verbose:
                print(f"\nConverged after {i+1} iterations.")
            break
    else:  # If loop finishes without break
        if verbose:
            print(f"\nReached max iterations ({num_iters}) without converging.")

    # Calculate final transport plan P
    # P = np.diag(u) @ K @ np.diag(v) # Equivalent but less efficient
    P = u[:, None] * K * v[None, :]

    # Calculate regularized OT cost
    cost = np.sum(P * C)

    return P, u, v, cost


# |%%--%%| <2N8OHWJLTf|7yOljOrKU8>

import numpy as np
from scipy.special import logsumexp  # Numerically stable log-sum-exp


def sinkhorn_dual_log(r, c, C, epsilon, num_iters=100, tol=1e-9, verbose=False):
    """
    Solves entropy-regularized OT using the Dual Sinkhorn algorithm
    in log-space for numerical stability.

    Args:
        r (np.ndarray): Source distribution (m,). Must sum to 1.
        c (np.ndarray): Target distribution (n,). Must sum to 1.
        C (np.ndarray): Cost matrix (m, n).
        epsilon (float): Regularization strength.
        num_iters (int): Maximum number of iterations.
        tol (float): Tolerance for convergence check based on dual potentials.
        verbose (bool): Print convergence progress.

    Returns:
        tuple: (P, f, g, cost, dual_obj)
            P (np.ndarray): Optimal transport plan (m, n).
            f (np.ndarray): Dual potential vector (m,).
            g (np.ndarray): Dual potential vector (n,).
            cost (float): Regularized OT cost sum(P * C).
            dual_obj (float): Value of the dual objective sum(f*r) + sum(g*c).
    """
    m, n = C.shape
    if not np.isclose(r.sum(), 1.0):
        print(f"Warning: Source distribution r sums to {r.sum()}, normalizing.")
        r = r / r.sum()
    if not np.isclose(c.sum(), 1.0):
        print(f"Warning: Target distribution c sums to {c.sum()}, normalizing.")
        c = c / c.sum()
    if r.shape[0] != m or c.shape[0] != n:
        raise ValueError("Shape mismatch between r, c, and C.")

    # Initialize dual potentials
    f = np.zeros(m)
    g = np.zeros(n)

    # Precompute for efficiency
    C_eps = -C / epsilon
    log_r = np.log(r + 1e-100)  # Add epsilon for stability if r has zeros
    log_c = np.log(c + 1e-100)  # Add epsilon for stability if c has zeros

    if verbose:
        print(f"{'Iter':<5} | {'Change in f':<15} | {'Change in g':<15}")
        print("-" * 40)

    for i in range(num_iters):
        f_prev = f.copy()
        g_prev = g.copy()

        # Update g (using previous f)
        # logsumexp_arg_g = (f[:, None] - C) / epsilon # Equivalent to below
        logsumexp_arg_g = f[:, None] / epsilon + C_eps
        g = -epsilon * logsumexp(logsumexp_arg_g, axis=0) + epsilon * log_c

        # Update f (using updated g)
        # logsumexp_arg_f = (g[None, :] - C) / epsilon # Equivalent to below
        logsumexp_arg_f = g[None, :] / epsilon + C_eps
        f = -epsilon * logsumexp(logsumexp_arg_f, axis=1) + epsilon * log_r

        # --- Convergence Check ---
        # Check change in dual potentials f and g
        change_f = np.linalg.norm(f - f_prev)
        change_g = np.linalg.norm(g - g_prev)
        total_change = change_f + change_g

        if verbose and (i % 10 == 0 or i == num_iters - 1):
            print(f"{i:<5} | {change_f:<15.4e} | {change_g:<15.4e}")

        if total_change < tol:
            if verbose:
                print(f"\nConverged after {i+1} iterations.")
            break
    else:  # If loop finishes without break
        if verbose:
            print(f"\nReached max iterations ({num_iters}) without converging.")

    # Calculate final transport plan P
    # P = np.exp((f[:, None] + g[None, :] - C) / epsilon) # Equivalent to below
    log_P = (f[:, None] + g[None, :]) / epsilon + C_eps
    P = np.exp(log_P)

    # Calculate regularized OT cost
    cost = np.sum(P * C)

    # Calculate dual objective value
    dual_obj = np.sum(f * r) + np.sum(g * c)

    return P, f, g, cost, dual_obj


# |%%--%%| <7yOljOrKU8|bahDMVIFwI>

# --- Example Data ---
m = 3
n = 4
r = np.array([0.2, 0.5, 0.3])  # Source distribution (sums to 1)
c = np.array([0.1, 0.4, 0.3, 0.2])  # Target distribution (sums to 1)

# Create a cost matrix (e.g., squared Euclidean distance if points were on a line)
source_pos = np.linspace(0, 1, m)
target_pos = np.linspace(0, 1, n)
C = (source_pos[:, None] - target_pos[None, :]) ** 2  # Cost matrix

epsilon = 0.05  # Regularization strength
num_iters = 200  # Max iterations
tolerance = 1e-7

# --- Run Primal Sinkhorn ---
print("--- Running Primal Sinkhorn ---")
P_primal, u_primal, v_primal, cost_primal = sinkhorn_primal(
    r, c, C, epsilon, num_iters, tol=tolerance, verbose=True
)

print("\nPrimal Results:")
print("Optimal Transport Plan P (Primal):")
print(np.round(P_primal, 4))
print(f"\nRow sums: {np.round(P_primal.sum(axis=1), 4)} (Target: {r})")
print(f"Col sums: {np.round(P_primal.sum(axis=0), 4)} (Target: {c})")
print(f"\nRegularized OT Cost (Primal): {cost_primal:.6f}")

# --- Run Dual (Log-Stabilized) Sinkhorn ---
print("\n\n--- Running Dual (Log-Stabilized) Sinkhorn ---")
P_dual, f_dual, g_dual, cost_dual, dual_obj = sinkhorn_dual_log(
    r, c, C, epsilon, num_iters, tol=tolerance, verbose=True
)

print("\nDual (Log) Results:")
print("Optimal Transport Plan P (Dual/Log):")
print(np.round(P_dual, 4))
print(f"\nRow sums: {np.round(P_dual.sum(axis=1), 4)} (Target: {r})")
print(f"Col sums: {np.round(P_dual.sum(axis=0), 4)} (Target: {c})")
print(f"\nRegularized OT Cost (Dual/Log): {cost_dual:.6f}")
print(f"Dual Objective Value:           {dual_obj:.6f}")


# --- Verification ---
print("\n\n--- Verification ---")
# Check if primal and dual results are close
print(f"Difference in P matrices (norm): {np.linalg.norm(P_primal - P_dual):.4e}")
print(f"Difference in Costs: {abs(cost_primal - cost_dual):.4e}")
print(
    f"Difference between Primal Cost and Dual Objective: {abs(cost_dual - dual_obj):.4e}"
)

# Check relationship between primal (u, v) and dual (f, g)
f_from_primal = epsilon * np.log(u_primal + 1e-100)
g_from_primal = epsilon * np.log(v_primal + 1e-100)

# Note: f and g are defined up to additive constants (f + const, g - const)
# We compare them after centering
f_diff = f_dual - f_from_primal
g_diff = g_dual - g_from_primal
print(
    f"Norm of (f_dual - eps*log(u_primal)) after centering: {np.linalg.norm(f_diff - f_diff.mean()):.4e}"
)
print(
    f"Norm of (g_dual - eps*log(v_primal)) after centering: {np.linalg.norm(g_diff - g_diff.mean()):.4e}"
)
