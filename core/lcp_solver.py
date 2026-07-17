from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LCPSolution:
    lambda_n: np.ndarray
    gap_g: np.ndarray
    active_indices: list[int]
    inactive_indices: list[int]
    residuals: dict[str, float]
    iteration_count: int
    convergence_status: str
    active_set_trace: list[dict]


def solve_lcp_active_set(q: np.ndarray, W: np.ndarray, eps: float = 1e-9, max_iter: int = 100) -> LCPSolution:
    """Solve 0 <= lambda ⟂ g = q + W lambda >= 0 using a small active-set method.

    This implementation is intentionally compact and transparent for thesis/software verification.
    It assumes W is symmetric positive semidefinite on the active set.
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    W = np.asarray(W, dtype=float)
    if W.shape != (q.size, q.size):
        raise ValueError(f'W shape {W.shape} does not match q length {q.size}')

    n = q.size
    active = set(np.where(q < -eps)[0].tolist())
    trace: list[dict] = []
    lam = np.zeros(n)

    for it in range(max_iter):
        lam = np.zeros(n)
        if active:
            idx = np.array(sorted(active), dtype=int)
            Wcc = W[np.ix_(idx, idx)]
            rhs = -q[idx]
            try:
                lam_c = np.linalg.solve(Wcc + 1e-12 * np.eye(len(idx)), rhs)
            except np.linalg.LinAlgError:
                lam_c = np.linalg.lstsq(Wcc + 1e-9 * np.eye(len(idx)), rhs, rcond=None)[0]
            lam[idx] = lam_c
            if lam_c.min(initial=0.0) < -eps:
                worst = int(idx[int(np.argmin(lam_c))])
                active.remove(worst)
                trace.append({'iteration': it, 'action': 'remove_negative_force', 'index': worst, 'active_set': sorted(active)})
                continue

        gap = q + W @ lam
        inactive = [i for i in range(n) if i not in active]
        if inactive:
            gD = gap[inactive]
            if gD.min(initial=0.0) < -eps:
                worst = int(inactive[int(np.argmin(gD))])
                active.add(worst)
                trace.append({'iteration': it, 'action': 'add_negative_gap', 'index': worst, 'active_set': sorted(active)})
                continue

        residuals = _compute_residuals(q, W, lam, gap)
        return LCPSolution(
            lambda_n=lam,
            gap_g=gap,
            active_indices=sorted(active),
            inactive_indices=[i for i in range(n) if i not in active],
            residuals=residuals,
            iteration_count=it + 1,
            convergence_status='CONVERGED',
            active_set_trace=trace,
        )

    gap = q + W @ lam
    return LCPSolution(
        lambda_n=lam,
        gap_g=gap,
        active_indices=sorted(active),
        inactive_indices=[i for i in range(n) if i not in active],
        residuals=_compute_residuals(q, W, lam, gap),
        iteration_count=max_iter,
        convergence_status='MAX_ITER',
        active_set_trace=trace,
    )


def _compute_residuals(q: np.ndarray, W: np.ndarray, lam: np.ndarray, gap: np.ndarray) -> dict[str, float]:
    return {
        'gap_violation': float(max(0.0, -np.min(gap))),
        'force_violation': float(max(0.0, -np.min(lam))),
        'complementarity_residual': float(np.max(np.abs(gap * lam))) if gap.size else 0.0,
        'equilibrium_residual': float(np.linalg.norm(gap - (q + W @ lam))),
    }
