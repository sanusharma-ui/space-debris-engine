# src/physics/chan_probability.py  niche wala fully working hai 
import numpy as np
from typing import Tuple

def _plane_basis_from_vector(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given unit vector u (3,), find two orthonormal basis vectors e1,e2 spanning plane orthogonal to u.
    """
    u = np.array(u, dtype=float)
    if np.linalg.norm(u) == 0:
        raise ValueError("Zero vector for plane normal")
    u = u / np.linalg.norm(u)
    # pick arbitrary vector not parallel to u
    arb = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(arb, u)) > 0.9:
        arb = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u, arb)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    e2 = e2 / np.linalg.norm(e2)
    return e1, e2

def chan_collision_probability(
    rel_pos: np.ndarray,
    rel_vel: np.ndarray,
    cov_rel: np.ndarray,
    collision_radius: float,
    n_theta: int = 48,
    n_r: int = 96
) -> float:
    """
    Compute approximate Chan-style collision probability by projecting relative position & covariance
    onto the plane orthogonal to relative velocity (b-plane) and numerically integrating the 2D Gaussian
    over the disk of radius = collision_radius.

    rel_pos: full 3D relative position vector (ECI) at tca (m)
    rel_vel: full 3D relative velocity vector (ECI) at tca (m/s)
    cov_rel: 3x3 relative position covariance matrix (m^2)
    collision_radius: scalar radius (m)
    """
    rel_pos = np.array(rel_pos, dtype=float)
    rel_vel = np.array(rel_vel, dtype=float)
    cov_rel = np.array(cov_rel, dtype=float)

    vnorm = np.linalg.norm(rel_vel)
    if vnorm < 1e-8:
        # fallback: use isotropic projection
        # project onto plane orthogonal to arbitrary axis
        rel_vel = np.array([1.0, 0.0, 0.0])
        vnorm = 1.0

    u = rel_vel / vnorm
    e1, e2 = _plane_basis_from_vector(u)

    # project mean into plane coordinates
    mu2 = np.array([np.dot(rel_pos, e1), np.dot(rel_pos, e2)], dtype=float)

    # project covariance into plane: 2x2
    C11 = float(e1.dot(cov_rel.dot(e1)))
    C12 = float(e1.dot(cov_rel.dot(e2)))
    C22 = float(e2.dot(cov_rel.dot(e2)))
    C2 = np.array([[C11, C12], [C12, C22]], dtype=float)

    # regularize if necessary
    eps = 1e-6 * max(np.trace(C2), 1.0)
    C2 += np.eye(2) * eps

    detC2 = np.linalg.det(C2)
    if detC2 <= 0 or not np.isfinite(detC2):
        # fallback to isotropic sigma from trace
        sigma = np.sqrt(np.trace(cov_rel) / 3.0)
        if sigma <= 0:
            sigma = 100.0
        # 2D isotropic density integral over disk with center offset mu2
        # approximate via simple formula using radial Gaussian (crude fallback)
        d = np.linalg.norm(mu2)
        return float(np.exp(-(d ** 2) / (2.0 * sigma ** 2)))

    invC2 = np.linalg.inv(C2)
    norm_prefactor = 1.0 / (2.0 * np.pi * np.sqrt(detC2))

    # polar integration grids
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    rs = np.linspace(0.0, collision_radius, n_r)

    # precompute r weights for trapezoid (simple)
    # perform integration: integral = ∫_0^{2π} ∫_0^R r * f(r,theta) dr dtheta
    total = 0.0
    for theta in thetas:
        ux = np.cos(theta)
        uy = np.sin(theta)
        # vector of unit direction in 2D plane basis
        # radial integration values
        vals = []
        for r in rs:
            point = np.array([r * ux, r * uy])  # coordinates in plane
            dvec = point - mu2
            exponent = -0.5 * float(dvec.dot(invC2.dot(dvec)))
            vals.append(np.exp(exponent) * r)
        # trapezoidal integrate over r
        vals = np.array(vals, dtype=float)
        integral_r = 0.0
        if n_r > 1:
            integral_r = np.trapz(vals, rs)
        else:
            integral_r = vals[0] * collision_radius
        total += integral_r

    integral = (2.0 * np.pi / float(len(thetas))) * total
    prob = norm_prefactor * integral
    # safety clipping
    if not np.isfinite(prob):
        prob = 0.0
    prob = max(0.0, min(1.0, prob))
    return float(prob)


# # numba  isme issue hai ... windows error koi theek kar sakta hai toh kar lo
# # src/physics/chan_probability.py
# import numpy as np
# from typing import Tuple

# # Safe numba decorator (fallback if numba is not available)
# try:
#     from numba import njit
# except ImportError:
#     def njit(**kwargs):
#         def decorator(func):
#             return func
#         return decorator


# def _plane_basis_from_vector(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Given unit vector u (3,), find two orthonormal basis vectors e1,e2 
#     spanning the plane orthogonal to u.
#     """
#     u = np.asarray(u, dtype=float)
#     norm_u = np.linalg.norm(u)
#     if norm_u < 1e-12:
#         raise ValueError("Zero vector provided as plane normal")
    
#     u = u / norm_u
    
#     # Choose arbitrary vector not (almost) parallel to u
#     arb = np.array([1.0, 0.0, 0.0])
#     if abs(np.dot(arb, u)) > 0.95:  # a bit stricter threshold
#         arb = np.array([0.0, 1.0, 0.0])
        
#     e1 = np.cross(u, arb)
#     e1 /= np.linalg.norm(e1)
    
#     e2 = np.cross(u, e1)
#     e2 /= np.linalg.norm(e2)
    
#     return e1, e2


# @njit(cache=True, fastmath=True)
# def _polar_integral(
#     n_theta: int,
#     n_r: int,
#     R: float,           # collision_radius
#     mu_x: float,
#     mu_y: float,
#     invC00: float,
#     invC01: float,
#     invC10: float,
#     invC11: float,
#     norm_prefactor: float
# ) -> float:
#     """
#     Fast Numba-compiled polar integration over the collision disk.
#     Uses simple trapezoidal rule in r direction.
#     """
#     total = 0.0
#     two_pi = 2.0 * 3.141592653589793
    
#     for ti in range(n_theta):
#         theta = two_pi * ti / n_theta
#         ux = np.cos(theta)
#         uy = np.sin(theta)
        
#         integral_r = 0.0
#         prev_val = 0.0
#         prev_r = 0.0
        
#         for ri in range(n_r):
#             # linear spacing in r
#             r = R * ri / (n_r - 1) if n_r > 1 else R
            
#             x = r * ux
#             y = r * uy
            
#             dx = x - mu_x
#             dy = y - mu_y
            
#             # quadratic form: dx*invC*dx + 2*dx*invC*dy + dy*invC*dy
#             quad = (dx * (invC00 * dx + invC01 * dy) +
#                     dy * (invC10 * dx + invC11 * dy))
            
#             val = np.exp(-0.5 * quad) * r
            
#             if ri > 0:
#                 integral_r += 0.5 * (prev_val + val) * (r - prev_r)
                
#             prev_val = val
#             prev_r = r
        
#         total += integral_r
    
#     integral = (two_pi / n_theta) * total
#     prob = norm_prefactor * integral
    
#     # Safe clamping
#     if not np.isfinite(prob) or prob < 0.0:
#         prob = 0.0
#     if prob > 1.0:
#         prob = 1.0
        
#     return prob


# def chan_collision_probability(
#     rel_pos: np.ndarray,
#     rel_vel: np.ndarray,
#     cov_rel: np.ndarray,
#     collision_radius: float,
#     n_theta: int = 48,
#     n_r: int = 96
# ) -> float:
#     """
#     Approximate Chan-style collision probability using 2D Gaussian integration
#     over the b-plane (plane perpendicular to relative velocity at TCA).

#     Parameters
#     ----------
#     rel_pos : array-like (3,)
#         Relative position vector at TCA (m)
#     rel_vel : array-like (3,)
#         Relative velocity vector at TCA (m/s)
#     cov_rel : array-like (3,3)
#         Relative position covariance matrix (m²)
#     collision_radius : float
#         Sum of object radii (hard-body radius) [m]
#     n_theta, n_r : int, optional
#         Angular and radial integration points (controls accuracy vs speed)

#     Returns
#     -------
#     float
#         Estimated collision probability (between 0 and 1)
#     """
#     rel_pos = np.asarray(rel_pos, dtype=float)
#     rel_vel = np.asarray(rel_vel, dtype=float)
#     cov_rel = np.asarray(cov_rel, dtype=float)

#     if rel_pos.shape != (3,) or rel_vel.shape != (3,) or cov_rel.shape != (3, 3):
#         raise ValueError("Input arrays must have correct shapes: (3,), (3,), (3,3)")

#     # Avoid division by zero / very slow encounters
#     vnorm = np.linalg.norm(rel_vel)
#     if vnorm < 1e-8:
#         rel_vel = np.array([1.0, 0.0, 0.0], dtype=float)
#         vnorm = 1.0

#     u = rel_vel / vnorm

#     e1, e2 = _plane_basis_from_vector(u)

#     # Project mean position onto b-plane
#     mu2 = np.array([
#         np.dot(rel_pos, e1),
#         np.dot(rel_pos, e2)
#     ], dtype=float)

#     # Project covariance onto b-plane (2×2)
#     C11 = e1 @ cov_rel @ e1
#     C12 = e1 @ cov_rel @ e2
#     C22 = e2 @ cov_rel @ e2

#     C2 = np.array([[C11, C12], [C12, C22]], dtype=float)

#     # Small regularization to avoid numerical issues
#     eps = 1e-6 * max(np.trace(C2), 1.0)
#     C2 += np.eye(2) * eps

#     detC2 = np.linalg.det(C2)

#     # Fallback for degenerate / invalid covariance
#     if detC2 <= 0 or not np.isfinite(detC2):
#         sigma = np.sqrt(np.trace(cov_rel) / 3.0)
#         if sigma <= 0:
#             sigma = 100.0  # very large default uncertainty
#         d = np.linalg.norm(mu2)
#         return float(np.exp(-0.5 * (d / sigma) ** 2))

#     invC2 = np.linalg.inv(C2)
#     norm_prefactor = 1.0 / (2.0 * np.pi * np.sqrt(detC2))

#     # Call fast numba integration
#     prob = _polar_integral(
#         n_theta,
#         n_r,
#         float(collision_radius),
#         mu2[0], mu2[1],
#         invC2[0, 0], invC2[0, 1],
#         invC2[1, 0], invC2[1, 1],
#         norm_prefactor
#     )

#     return float(prob)

