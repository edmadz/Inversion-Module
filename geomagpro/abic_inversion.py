"""
geomagpro.abic_inversion
========================
ABIC (Akaike's Bayesian Information Criterion) 3D spectral inversion
for gravity density contrast and aeromagnetic susceptibility contrast.

Theoretical framework
---------------------
ABIC solves the fundamental regularisation problem in geophysical
inversion: determining the optimal weight between data fit and model
smoothness without subjective parameter choice.  The method maximises
the marginal likelihood p(d|ω) over the regularisation parameter ω,
which is equivalent to minimising the ABIC score:

    ABIC(ω) = N · log σ²(ω) + 2 · log₁₀(ω) + const

where N is the number of valid observations and σ²(ω) is the data
variance after inversion with regularisation weight ω
(Akaike 1980; Murata 1993).

The spectral forward operators connect model (density or susceptibility
contrast) to predicted field anomaly in the wavenumber domain:

  Gravity:   Ĝ_pred(k) = 2πG · Δρ̂(k) · exp(−2π|k|z) · dz
  Magnetic:  Ĝ_pred(k) =  G  · Δκ̂(k) · exp(−2π|k|z) · dz

where k = √(kx² + ky²) is the radial wavenumber and z is layer depth.

Inversion is performed by spectral Tikhonov regularisation in the
wavenumber domain:

    Δρ̂(k) = H*(k) · Ĝ_obs(k) / (|H(k)|² + ω · |k|⁴)

The optimal ω is found by minimising ABIC over a range of candidate
values using scipy.optimize.minimize_scalar.

Academic references
-------------------
Akaike H (1980) Likelihood and the Bayes procedure. In: Bernardo JM
  et al. (eds) Bayesian Statistics. University Press, Valencia,
  pp 143-166.

Murata Y (1993) Estimation of optimum average surficial density from
  gravity data: an objective Bayesian approach. J Geophys Res Solid
  Earth 98:12097-12109. https://doi.org/10.1029/93JB00571

Fukuda J, Johnson KM (2008) A fully Bayesian inversion for spatial
  distribution of fault slip. Bull Seismol Soc Am 98:1128-1146.
  https://doi.org/10.1785/0120070194

Yabuki T, Matsu'ura M (1992) Geodetic data inversion using a Bayesian
  information criterion. Geophys J Int 109:363-375.
  https://doi.org/10.1111/j.1365-246X.1992.tb00102.x

Author
------
Muhammet Ali Aygün
Istanbul Tecnical University&Istanbul University, Institute of Marine Sciences and Management
maygun@ogr.iu.edu.tr

Version : 1.0.2
Date    : 2025-05
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.optimize import minimize_scalar


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ABICParams:
    """Configuration for ABIC spectral inversion.

    Attributes
    ----------
    log_omega_min : float
        Lower bound of log₁₀(ω) search range.
    log_omega_max : float
        Upper bound of log₁₀(ω) search range.
    n_grid_search : int
        Number of candidate ω values in the initial grid search.
    param_clip : float
        Physical bound for clipping recovered model values.
        Applied symmetrically: model ∈ [−clip, +clip].
    """
    log_omega_min  : float = 1.0
    log_omega_max  : float = 8.0
    n_grid_search  : int   = 30
    param_clip     : float = 0.15   # g/cm³ for gravity; SI for magnetic


@dataclass
class ABICResult:
    """Container for a complete ABIC 3D inversion result.

    Attributes
    ----------
    model_layers : list[np.ndarray]
        Recovered parameter field for each depth layer; each array has
        shape (nys, nxs) — the subsampled grid dimensions.
    pred_layers : list[np.ndarray]
        Predicted anomaly field for each layer (same shape).
    rms_layers : list[float]
        RMS misfit for each layer (mGal or nT).
    opt_omega : float
        Optimal regularisation parameter selected by ABIC.
    opt_log_omega : float
        log₁₀(opt_omega).
    abic_grid_log_omegas : np.ndarray
        log₁₀(ω) values tested in the grid search.
    abic_grid_scores : list[float]
        ABIC scores at each grid search point.
    lo_sub, la_sub : np.ndarray
        Longitude / latitude coordinate arrays of the subsampled grid.
    depths_km : list[float]
        Top depth of each layer (km).
    dzs_km : list[float]
        Thickness of each layer (km).
    data_type : str
        'gravity' or 'magnetic'.
    """
    model_layers         : list
    pred_layers          : list
    rms_layers           : list
    opt_omega            : float
    opt_log_omega        : float
    abic_grid_log_omegas : np.ndarray
    abic_grid_scores     : list
    lo_sub               : np.ndarray
    la_sub               : np.ndarray
    depths_km            : list
    dzs_km               : list
    data_type            : str = 'gravity'


# ─────────────────────────────────────────────────────────────────────────────
# Spectral forward operators
# ─────────────────────────────────────────────────────────────────────────────

def _gravity_spectral_kernel(ny: int, nx: int, dx_km: float,
                              z_km: float, dz_km: float) -> np.ndarray:
    """2D FFT gravity forward operator for a thin horizontal slab.

    H(k) = 2π G_nT · dz · exp(−2π |k| z)

    where G_nT = 2π × 6.674×10⁻¹¹ × 10³ × 10⁵ = ~4.19×10⁻⁵ mGal·km/g·cm⁻³.

    Parameters
    ----------
    ny, nx : int
        Grid dimensions.
    dx_km : float
        Cell spacing (km) — assumed isotropic.
    z_km : float
        Layer centre depth (km).
    dz_km : float
        Layer thickness (km).

    Returns
    -------
    H : np.ndarray, shape (ny, nx), complex-valued
        Wavenumber-domain operator.
    """
    G_SI  = 6.674e-11   # m³/(kg·s²)
    rho   = 1000.0      # 1 g/cm³ in kg/m³
    mGal  = 1e5
    km    = 1000.0

    ky = np.fft.fftfreq(ny, d=dx_km)
    kx = np.fft.fftfreq(nx, d=dx_km)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)
    K[K == 0] = 1e-12

    return 2 * np.pi * G_SI * rho * (dz_km * km) * np.exp(-2 * np.pi * K * z_km) * mGal


def _magnetic_spectral_kernel(ny: int, nx: int, dx_km: float,
                               z_km: float, dz_km: float,
                               scale: float = 1e5) -> np.ndarray:
    """2D FFT magnetic forward operator for RTP data.

    H(k) = scale · dz · exp(−2π |k| z)

    The scale factor (~10⁵ nT·km/SI) is chosen to give anomaly
    amplitudes in nT for susceptibility contrasts in SI units.

    Parameters
    ----------
    ny, nx : int
        Grid dimensions.
    dx_km : float
        Cell spacing (km).
    z_km : float
        Layer depth (km).
    dz_km : float
        Layer thickness (km).
    scale : float
        Physical scaling constant.

    Returns
    -------
    H : np.ndarray, shape (ny, nx)
        Wavenumber-domain operator.
    """
    ky = np.fft.fftfreq(ny, d=dx_km)
    kx = np.fft.fftfreq(nx, d=dx_km)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)
    K[K == 0] = 1e-12
    return scale * dz_km * np.exp(-2 * np.pi * K * z_km)


# ─────────────────────────────────────────────────────────────────────────────
# ABIC score and inversion engine
# ─────────────────────────────────────────────────────────────────────────────

def _spectral_invert(H: np.ndarray,
                     G_obs: np.ndarray,
                     K2: np.ndarray,
                     omega: float) -> tuple[np.ndarray, np.ndarray]:
    """Spectral Tikhonov inversion for one layer.

    m̂(k) = H*(k) G_obs(k) / (|H(k)|² + ω |k|⁴)

    Parameters
    ----------
    H : np.ndarray
        Forward operator (wavenumber domain).
    G_obs : np.ndarray
        FFT of the zero-mean observed anomaly.
    K2 : np.ndarray
        |k|² = kx² + ky².
    omega : float
        Regularisation parameter.

    Returns
    -------
    model : np.ndarray
        Recovered model in the spatial domain.
    pred : np.ndarray
        Predicted anomaly in the spatial domain.
    """
    denom = np.abs(H) ** 2 + omega * K2 ** 2
    denom[denom == 0] = 1e-30
    model_hat = np.conj(H) * G_obs / denom
    model     = np.real(np.fft.ifft2(model_hat))
    pred      = np.real(np.fft.ifft2(H * model_hat))
    return model, pred


def _abic_score(log_omega: float,
                H: np.ndarray,
                G_obs: np.ndarray,
                K2: np.ndarray,
                mask: np.ndarray,
                anom: np.ndarray) -> float:
    """Evaluate ABIC(ω) = N·log σ²(ω) + 2·log₁₀(ω).

    Parameters
    ----------
    log_omega : float
        log₁₀ of the regularisation parameter.
    H, G_obs, K2 : np.ndarray
        Forward operator, observed FFT, wavenumber squared.
    mask : np.ndarray, bool
        Valid observation mask.
    anom : np.ndarray
        Zero-mean anomaly (spatial domain).

    Returns
    -------
    float
        ABIC score (lower is better).
    """
    omega = 10 ** log_omega
    _, pred = _spectral_invert(H, G_obs, K2, omega)
    sigma2 = np.mean((anom[mask] - pred[mask]) ** 2)
    if sigma2 <= 0:
        return np.inf
    return int(mask.sum()) * np.log(sigma2) + 2.0 * log_omega


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_abic_gravity(grid: np.ndarray,
                     lons: np.ndarray,
                     lats: np.ndarray,
                     dx_deg: float,
                     depths_km: list[float],
                     dzs_km: list[float],
                     subsample_step: int = 4,
                     params: Optional[ABICParams] = None,
                     verbose: bool = True) -> ABICResult:
    """ABIC 3D spectral gravity inversion.

    Inverts the Bouguer gravity grid for density contrast at each
    RAPS-guided depth layer.  The optimal regularisation ω is selected
    by minimising the ABIC score using a two-stage strategy: coarse
    grid search followed by bounded scalar optimisation.

    Parameters
    ----------
    grid : np.ndarray, shape (ny, nx)
        Bouguer anomaly grid (mGal).
    lons : np.ndarray, shape (nx,)
        Longitude array.
    lats : np.ndarray, shape (ny,)
        Latitude array.
    dx_deg : float
        Grid cell size in degrees (used to compute dx_km).
    depths_km : list[float]
        Top depths of inversion layers (km).
    dzs_km : list[float]
        Layer thicknesses (km).
    subsample_step : int
        Spatial subsampling factor (1 = full resolution, 4 = every 4th
        cell).  Increase for speed; decrease for resolution.
    params : ABICParams, optional
    verbose : bool

    Returns
    -------
    ABICResult
    """
    return _run_abic_core(
        grid=grid, lons=lons, lats=lats, dx_deg=dx_deg,
        depths_km=depths_km, dzs_km=dzs_km,
        subsample_step=subsample_step, params=params,
        data_type='gravity', verbose=verbose)


def run_abic_magnetic(grid: np.ndarray,
                      lons: np.ndarray,
                      lats: np.ndarray,
                      dx_deg: float,
                      depths_km: list[float],
                      dzs_km: list[float],
                      subsample_step: int = 4,
                      params: Optional[ABICParams] = None,
                      verbose: bool = True) -> ABICResult:
    """ABIC 3D spectral magnetic inversion.

    Identical workflow to :func:`run_abic_gravity` but uses the
    magnetic spectral kernel appropriate for RTP-transformed data.

    Parameters
    ----------
    grid : np.ndarray, shape (ny, nx)
        RTP total-field anomaly grid (nT).
    See :func:`run_abic_gravity` for remaining parameters.

    Returns
    -------
    ABICResult
        ``data_type = 'magnetic'``; model values are susceptibility
        contrast in SI units.
    """
    return _run_abic_core(
        grid=grid, lons=lons, lats=lats, dx_deg=dx_deg,
        depths_km=depths_km, dzs_km=dzs_km,
        subsample_step=subsample_step, params=params,
        data_type='magnetic', verbose=verbose)


def _run_abic_core(grid, lons, lats, dx_deg, depths_km, dzs_km,
                   subsample_step, params, data_type, verbose):
    """Shared implementation for gravity and magnetic ABIC inversion."""
    if params is None:
        params = ABICParams()

    step   = subsample_step
    g_sub  = grid[::step, ::step]
    lo_sub = lons[::step]
    la_sub = lats[::step]
    nys, nxs = g_sub.shape
    dx_km    = dx_deg * step * 111.0

    g_mean = float(np.nanmean(g_sub))
    mask   = ~np.isnan(g_sub)
    anom   = np.nan_to_num(g_sub - g_mean, nan=0.0)

    G_obs = np.fft.fft2(anom)
    ky    = np.fft.fftfreq(nys, d=dx_km)
    kx    = np.fft.fftfreq(nxs, d=dx_km)
    KX, KY = np.meshgrid(kx, ky)
    K2    = KX ** 2 + KY ** 2

    # Use the shallowest layer for ω selection (best-conditioned)
    kernel_fn = (_gravity_spectral_kernel if data_type == 'gravity'
                 else _magnetic_spectral_kernel)
    H_ref = kernel_fn(nys, nxs, dx_km, depths_km[0], dzs_km[0])

    def score(log_omega):
        return _abic_score(log_omega, H_ref, G_obs, K2, mask, anom)

    # Grid search
    log_omegas  = np.linspace(params.log_omega_min, params.log_omega_max,
                              params.n_grid_search)
    grid_scores = [score(lo) for lo in log_omegas]

    if verbose:
        print(f'  ABIC grid search ({data_type}):')
        for lo, sc in zip(log_omegas, grid_scores):
            print(f'    log10(ω)={lo:.2f}  ABIC={sc:.2f}')

    best_lo  = log_omegas[int(np.argmin(grid_scores))]
    lo_bound = max(params.log_omega_min, best_lo - 1.0)
    hi_bound = min(params.log_omega_max, best_lo + 1.0)
    res_opt  = minimize_scalar(score, bounds=(lo_bound, hi_bound),
                               method='bounded')
    opt_log  = float(res_opt.x)
    opt_omega = 10 ** opt_log

    if verbose:
        print(f'  Optimal log10(ω)={opt_log:.3f}  ω={opt_omega:.4g}')

    # Invert each layer
    model_layers, pred_layers, rms_layers = [], [], []
    for z, dz in zip(depths_km, dzs_km):
        H = kernel_fn(nys, nxs, dx_km, z, dz)
        m, p = _spectral_invert(H, G_obs, K2, opt_omega)
        m_clip = np.clip(m, -params.param_clip, params.param_clip)
        rms    = float(np.sqrt(np.mean((anom[mask] - p[mask]) ** 2)))
        model_layers.append(m_clip)
        pred_layers.append(p)
        rms_layers.append(rms)
        if verbose:
            print(f'  Layer z={z:.2f} km: RMS={rms:.4f}  '
                  f'range {m_clip.min():.5f}–{m_clip.max():.5f}')

    return ABICResult(
        model_layers         = model_layers,
        pred_layers          = pred_layers,
        rms_layers           = rms_layers,
        opt_omega            = opt_omega,
        opt_log_omega        = opt_log,
        abic_grid_log_omegas = log_omegas,
        abic_grid_scores     = grid_scores,
        lo_sub               = lo_sub,
        la_sub               = la_sub,
        depths_km            = depths_km,
        dzs_km               = dzs_km,
        data_type            = data_type,
    )
