"""
geomagpro.li_oldenburg
======================
Li-Oldenburg Iteratively Reweighted Least Squares (IRLS) inversion
for 2D density contrast (gravity) and susceptibility contrast
(magnetic) along a single profile.

Theoretical framework
---------------------
The inversion minimises the regularised objective function:

    φ(m) = ||Gm − d||² + λ (φ_s + φ_x)

where φ_s and φ_x are depth-weighted smallness and smoothness model
norms, respectively.  Depth weighting follows

    w(z) = 1 / (z + ε)^(β/2)

with β = 1.5 for gravity (Li and Oldenburg 1996) and β = 3.0 for
magnetic data (Li and Oldenburg 1998), reflecting the different spatial
decay rates of the two fields.

The IRLS reweighting scheme (Li and Oldenburg 1996) iteratively
updates the model norm matrices at each iteration so that compact,
focused density or susceptibility distributions emerge without the
diffuse smearing typical of standard least-squares regularisation.

Academic references
-------------------
Li Y, Oldenburg DW (1996) 3-D inversion of magnetic data.
  Geophysics 61:394-408. https://doi.org/10.1190/1.1443968

Li Y, Oldenburg DW (1998) 3-D inversion of gravity data.
  Geophysics 63:109-119. https://doi.org/10.1190/1.1444302

Author
------
Muhammet Ali Aygün
Istanbul University, Institute of Marine Sciences, Dep.of Marine Geology and Geophysics.
maygun@ogr.iu.edu.tr

Version : 1.0.1
Date    : 2024-03
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LiOldenburgParams:
    """Configuration for Li-Oldenburg IRLS inversion.

    Attributes
    ----------
    n_iterations : int
        Number of IRLS iterations.
    regularisation_lambda : float
        Trade-off parameter λ between data fit and model norm.
    depth_weight_beta : float
        Exponent for depth weighting w(z) = 1/(z + ε)^(β/2).
        Li and Oldenburg (1996) recommend 1.5 for gravity, 3.0 for
        magnetic.
    depth_weight_eps : float
        Small offset ε to prevent singularity at the surface (km).
    param_clip : float
        Hard bound applied to the recovered model values.
    rms_tol : float
        Convergence tolerance (same units as input data).
    """
    n_iterations          : int   = 100
    regularisation_lambda : float = 0.50
    depth_weight_beta     : float = 1.50   # use 3.0 for magnetic
    depth_weight_eps      : float = 0.01
    param_clip            : float = 0.80
    rms_tol               : float = 1e-4


@dataclass
class LiOldenburgResult:
    """Container for a Li-Oldenburg inversion result.

    Attributes
    ----------
    lon_profile : np.ndarray, shape (n_obs,)
        Longitude values along the profile.
    obs : np.ndarray, shape (n_obs,)
        Zero-mean observed anomaly (mGal or nT).
    pred : np.ndarray, shape (n_obs,)
        Final predicted anomaly.
    model : np.ndarray, shape (n_layers, n_obs)
        Recovered parameter field.
    rms : float
        Final RMS misfit.
    convergence : list[float]
        Per-iteration RMS.
    n_layers : int
    layer_tops_km : list[float]
    layer_bots_km : list[float]
    data_type : str
        'gravity' or 'magnetic'.
    """
    lon_profile   : np.ndarray
    obs           : np.ndarray
    pred          : np.ndarray
    model         : np.ndarray
    rms           : float
    convergence   : list
    n_layers      : int
    layer_tops_km : list
    layer_bots_km : list
    data_type     : str = 'gravity'


# ─────────────────────────────────────────────────────────────────────────────
# Forward kernels (same as PSO module, reproduced here for independence)
# ─────────────────────────────────────────────────────────────────────────────

def _grav_kernel_element(x_obs: float, z_top: float, z_bot: float,
                          x_src: float, dx: float) -> float:
    """Vertical gravity of a 2D prism at a single observation point (m/s²
    per kg/m³).  Multiply by 1e5 × 1000 for mGal per g/cm³."""
    G  = 6.674e-11
    rp = 1000.0

    x1 = (x_obs - x_src - dx / 2) * 1e3
    x2 = (x_obs - x_src + dx / 2) * 1e3
    z1 = z_top * 1e3
    z2 = z_bot * 1e3

    def F(x, z):
        return (x * np.arctan2(z, abs(x) + 1e-12)
                - z / 2 * np.log(x ** 2 + z ** 2 + 1e-9))

    return 2 * G * rp * (F(x2, z2) - F(x2, z1) - F(x1, z2) + F(x1, z1)) * 1e5


def _mag_kernel_element(x_obs: float, z_top: float, z_bot: float,
                         x_src: float, dx: float,
                         scale: float = 100.0) -> float:
    """Vertical magnetic anomaly (RTP) of a 2D prism at a single point."""
    x1 = (x_obs - x_src - dx / 2) * 1e3
    x2 = (x_obs - x_src + dx / 2) * 1e3
    z1 = z_top * 1e3
    z2 = z_bot * 1e3

    def F(x, z):
        return (z * np.arctan2(x, z)
                - x / 2 * np.log(x ** 2 + z ** 2 + 1e-9))

    return scale * (F(x2, z2) - F(x2, z1) - F(x1, z2) + F(x1, z1))


def _build_sensitivity_matrix(x_obs_km: np.ndarray,
                               layer_tops: list,
                               layer_dzs: list,
                               dx_km: float,
                               data_type: str = 'gravity') -> np.ndarray:
    """Assemble the full sensitivity matrix G.

    Parameters
    ----------
    x_obs_km : np.ndarray, shape (n_obs,)
        Horizontal positions of observations (km).
    layer_tops, layer_dzs : list
        Layer geometry.
    dx_km : float
        Cell width (km).
    data_type : {'gravity', 'magnetic'}

    Returns
    -------
    G : np.ndarray, shape (n_obs, n_obs * n_layers)
    """
    n_obs    = len(x_obs_km)
    n_layers = len(layer_tops)
    kern     = _grav_kernel_element if data_type == 'gravity' else _mag_kernel_element
    G        = np.zeros((n_obs, n_obs * n_layers))

    for il, (z, dz) in enumerate(zip(layer_tops, layer_dzs)):
        for js in range(n_obs):
            for io in range(n_obs):
                G[io, il * n_obs + js] = kern(
                    x_obs_km[io], z, z + dz, x_obs_km[js], dx_km)
    return G


# ─────────────────────────────────────────────────────────────────────────────
# Depth weighting
# ─────────────────────────────────────────────────────────────────────────────

def _depth_weight_matrix(layer_tops: list, layer_dzs: list,
                          n_obs: int, beta: float, eps: float) -> np.ndarray:
    """Diagonal depth weighting matrix W_z.

    w_j = 1 / (z_j + ε)^(β/2)  where z_j is the midpoint depth of
    the cell at layer l, position j.

    Parameters
    ----------
    layer_tops : list[float]
        Layer top depths (km).
    layer_dzs : list[float]
        Layer thicknesses (km).
    n_obs : int
        Number of observation points.
    beta, eps : float
        Depth weighting exponent and offset.

    Returns
    -------
    W : np.ndarray, shape (n_obs * n_layers,)
        Diagonal weights.
    """
    weights = []
    for z, dz in zip(layer_tops, layer_dzs):
        z_mid = z + dz / 2.0
        w     = 1.0 / (z_mid + eps) ** (beta / 2.0)
        weights.extend([w] * n_obs)
    return np.array(weights)


# ─────────────────────────────────────────────────────────────────────────────
# IRLS core
# ─────────────────────────────────────────────────────────────────────────────

def _irls_loop(G: np.ndarray,
               d_obs: np.ndarray,
               W_depth: np.ndarray,
               lam: float,
               n_iter: int,
               param_clip: float,
               rms_tol: float) -> tuple[np.ndarray, np.ndarray, list, float]:
    """Iteratively Reweighted Least Squares loop.

    At each iteration solves the normal equations:

        (G^T G + λ W_m^T W_m) m = G^T d

    where W_m combines depth weighting and the current IRLS weights.

    Parameters
    ----------
    G : np.ndarray, shape (n_obs, n_model)
        Sensitivity matrix.
    d_obs : np.ndarray, shape (n_obs,)
        Observed anomaly.
    W_depth : np.ndarray, shape (n_model,)
        Depth weight vector.
    lam : float
        Regularisation parameter λ.
    n_iter : int
        Maximum iterations.
    param_clip, rms_tol : float

    Returns
    -------
    m : np.ndarray, shape (n_model,)
        Final model.
    pred : np.ndarray, shape (n_obs,)
        Final predicted data.
    history : list[float]
        Per-iteration RMS.
    rms_final : float
    """
    n_m = G.shape[1]
    m   = np.zeros(n_m)
    eps = 1e-6
    history = []

    # Initial IRLS weights = depth weights
    irls_w = W_depth.copy()

    GtG = G.T @ G

    for it in range(n_iter):
        # Regularisation matrix
        Wm = irls_w
        A  = GtG + lam * np.diag(Wm ** 2)
        b  = G.T @ d_obs

        # Solve normal equations
        try:
            m = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            m = np.linalg.lstsq(A, b, rcond=None)[0]

        m = np.clip(m, -param_clip, param_clip)

        # Update IRLS weights
        irls_w = W_depth / (np.abs(m) + eps) ** 0.5

        pred  = G @ m
        rms   = float(np.sqrt(np.mean((d_obs - pred) ** 2)))
        history.append(rms)

        if rms < rms_tol:
            break

    return m, G @ m, history, history[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_li_oldenburg(anomaly_profile: np.ndarray,
                     lon_profile: np.ndarray,
                     layer_tops_km: list,
                     layer_dzs_km: list,
                     dx_grid_km: float,
                     data_type: str = 'gravity',
                     params: Optional[LiOldenburgParams] = None,
                     verbose: bool = True) -> LiOldenburgResult:
    """Li-Oldenburg IRLS inversion along a single E-W profile.

    The profile is typically extracted at a fixed latitude from the
    gridded anomaly field and subsampled to a manageable number of
    points before computing the sensitivity matrix.

    Parameters
    ----------
    anomaly_profile : np.ndarray, shape (n_obs,)
        Zero-mean observed Bouguer anomaly (mGal) or RTP anomaly (nT).
    lon_profile : np.ndarray, shape (n_obs,)
        Longitude values corresponding to the profile.
    layer_tops_km : list[float]
        Top depths of inversion layers (km).
    layer_dzs_km : list[float]
        Layer thicknesses (km).
    dx_grid_km : float
        Cell spacing used to construct the sensitivity matrix (km).
    data_type : {'gravity', 'magnetic'}
    params : LiOldenburgParams, optional
    verbose : bool

    Returns
    -------
    LiOldenburgResult
    """
    if params is None:
        beta = 3.0 if data_type == 'magnetic' else 1.5
        params = LiOldenburgParams(depth_weight_beta=beta)

    # Convert lon to km (E-W distance from profile start)
    x_obs_km = (lon_profile - lon_profile.min()) * 111.0 * np.cos(
        np.radians(40.46))   # approximate, valid for Marmara latitude

    if verbose:
        print(f'  Li-Oldenburg ({data_type}): '
              f'{len(x_obs_km)} obs, {len(layer_tops_km)} layers')

    G = _build_sensitivity_matrix(
        x_obs_km, layer_tops_km, layer_dzs_km, dx_grid_km, data_type)

    if verbose:
        print(f'  Sensitivity matrix: {G.shape}')

    W_depth = _depth_weight_matrix(
        layer_tops_km, layer_dzs_km, len(x_obs_km),
        params.depth_weight_beta, params.depth_weight_eps)

    # Normalise G by its Frobenius norm to help conditioning
    G_scale = np.max(np.abs(G)) + 1e-30
    G_n     = G / G_scale

    m, pred_norm, history, rms_norm = _irls_loop(
        G_n, anomaly_profile, W_depth,
        params.regularisation_lambda,
        params.n_iterations,
        params.param_clip,
        params.rms_tol)

    # Rescale back
    pred    = pred_norm
    rms     = rms_norm

    if verbose:
        print(f'  Final RMS: {rms:.4f}  '
              f'model range {m.min():.4f}–{m.max():.4f}')

    n_layers = len(layer_tops_km)
    n_obs    = len(x_obs_km)

    return LiOldenburgResult(
        lon_profile   = lon_profile,
        obs           = anomaly_profile,
        pred          = pred,
        model         = m.reshape(n_layers, n_obs),
        rms           = rms,
        convergence   = history,
        n_layers      = n_layers,
        layer_tops_km = layer_tops_km,
        layer_bots_km = [t + d for t, d in zip(layer_tops_km, layer_dzs_km)],
        data_type     = data_type,
    )
