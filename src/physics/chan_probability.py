"""
Chan-style collision probability calculation in the B-plane.

Projects the 3D relative state and covariance into the 2D plane perpendicular
to the relative velocity vector (B-plane), then numerically integrates the
2D Gaussian probability density over a circular disk of given collision radius.

This implementation uses vectorized numpy operations (meshgrid + trapz)
instead of nested Python loops.
"""

import numpy as np
from typing import Tuple


def _plane_basis_from_vector(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a unit vector u (3,), compute two orthonormal vectors e1, e2
    that span the plane orthogonal to u.

    Used to define the B-plane coordinate system.
    """
    u = np.asarray(u, dtype=float)
    norm_u = np.linalg.norm(u)
    if norm_u < 1e-12:
        raise ValueError("Cannot define plane basis from (near-)zero vector")

    u = u / norm_u

    # Choose an arbitrary vector not (too) parallel to u
    arb = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(arb, u)) > 0.95:
        arb = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(u, arb)
    e1 /= np.linalg.norm(e1)

    e2 = np.cross(u, e1)
    e2 /= np.linalg.norm(e2)

    return e1, e2


def chan_collision_probability(
    rel_pos: np.ndarray,
    rel_vel: np.ndarray,
    cov_rel: np.ndarray,
    collision_radius: float,
    n_theta: int = 64,
    n_r: int = 120
) -> float:
    """
    Approximate collision probability using Chan's method (B-plane projection
    + numerical integration of 2D Gaussian over a disk).

    Parameters
    ----------
    rel_pos : array-like, shape (3,)
        Relative position vector at TCA (m) in ECI
    rel_vel : array-like, shape (3,)
        Relative velocity vector at TCA (m/s) in ECI
    cov_rel : array-like, shape (3,3)
        Relative position covariance matrix at TCA (m²)
    collision_radius : float
        Sum of object radii (hard-body radius) [m]
    n_theta : int, optional
        Number of angular points (default: 64)
    n_r : int, optional
        Number of radial points (default: 120)

    Returns
    -------
    float
        Estimated collision probability (clipped to [0,1])
    """
    rel_pos = np.asarray(rel_pos, dtype=float)
    rel_vel = np.asarray(rel_vel, dtype=float)
    cov_rel = np.asarray(cov_rel, dtype=float)

    if rel_pos.shape != (3,) or rel_vel.shape != (3,) or cov_rel.shape != (3, 3):
        raise ValueError("rel_pos and rel_vel must be (3,), cov_rel must be (3,3)")

    vnorm = np.linalg.norm(rel_vel)
    if vnorm < 1e-6:
        # Very low relative velocity → fallback to isotropic approximation
        sigma = np.sqrt(np.trace(cov_rel) / 3.0)
        sigma = max(sigma, 10.0)  # avoid division by zero or tiny values
        d = np.linalg.norm(rel_pos)
        p = np.exp(-0.5 * (d / sigma) ** 2)
        return float(np.clip(p, 0.0, 1.0))

    # Unit vector along relative velocity (defines normal to B-plane)
    u = rel_vel / vnorm

    # Orthonormal basis in B-plane
    e1, e2 = _plane_basis_from_vector(u)

    # Mean position in 2D B-plane coordinates
    mu_x = np.dot(rel_pos, e1)
    mu_y = np.dot(rel_pos, e2)
    mu2 = np.array([mu_x, mu_y])

    # Project 3×3 covariance into 2×2 B-plane covariance
    C11 = e1 @ cov_rel @ e1
    C12 = e1 @ cov_rel @ e2
    C22 = e2 @ cov_rel @ e2
    C2 = np.array([[C11, C12], [C12, C22]])

    # Regularization (helps when covariance is near-singular)
    trace = np.trace(C2)
    eps = 1e-8 * max(trace, 1.0)
    C2 += eps * np.eye(2)

    detC2 = np.linalg.det(C2)
    if detC2 <= 0 or not np.isfinite(detC2):
        # Fallback — crude isotropic Gaussian
        sigma = np.sqrt(trace / 2.0) if trace > 0 else 100.0
        d = np.linalg.norm(mu2)
        p = np.exp(-0.5 * (d / sigma) ** 2)
        return float(np.clip(p, 0.0, 1.0))

    invC2 = np.linalg.inv(C2)
    a, b, c = invC2[0, 0], invC2[0, 1], invC2[1, 1]

    # Build polar coordinate grid (vectorized)
    # IMPORTANT: include 2π endpoint for trapezoidal integration over [0, 2π]
    # Using endpoint=False under-integrates by ~1/n_theta for periodic integrands.
    thetas = np.linspace(0, 2 * np.pi, int(n_theta) + 1, endpoint=True)
    rs = np.linspace(0, collision_radius, n_r)

    # meshgrid → shapes (n_r, n_theta+1)
    th_grid, r_grid = np.meshgrid(thetas, rs, indexing='xy')

    # Transpose so integration axes are convenient
    ux = np.cos(th_grid).T      # (n_theta+1, n_r)
    uy = np.sin(th_grid).T
    rmat = r_grid.T             # (n_theta+1, n_r)

    # Cartesian coordinates in B-plane (relative to mean)
    x = rmat * ux - mu_x
    y = rmat * uy - mu_y

    # Quadratic form of 2D Gaussian
    exponent = -0.5 * (a * x**2 + 2 * b * x * y + c * y**2)
    density = np.exp(exponent)

    # Integrand for polar coordinates: density * r
    integrand = density * rmat

    # Integrate first along radius (axis=1), then along theta (axis=0)
    int_r = np.trapz(integrand, rs, axis=1)          # shape (n_theta+1,)
    integral = np.trapz(int_r, thetas)               # scalar

    # Normalization constant of 2D Gaussian
    norm = 1.0 / (2.0 * np.pi * np.sqrt(detC2))

    prob = norm * integral

    # Final safety clipping
    if not np.isfinite(prob):
        prob = 0.0
    prob = max(0.0, min(1.0, prob))

    return float(prob)
