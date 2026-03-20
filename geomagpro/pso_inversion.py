"""
geomagpro.pso_inversion
=======================
Particle Swarm Optimisation (PSO) inversion for 2D gravity and
aeromagnetic profiles.

Theoretical framework
---------------------
PSO is a metaheuristic global optimisation method inspired by the
collective behaviour of bird flocks and fish schools (Kennedy and
Eberhart 1995). Each particle in the swarm represents a candidate
model vector; its velocity is updated every iteration using three
terms: inertia (tendency to keep moving in the same direction),
cognitive attraction toward its own personal-best position, and
social attraction toward the global-best position found by any
particle.

The gravity kernel follows the 2D rectangular-prism formulation of
Telford et al. (1990), evaluated analytically for the vertical
component of gravitational acceleration.  The magnetic kernel follows
the Talwani and Heirtzler (1964) formulation for induced
magnetisation under Reduction-to-Pole (RTP) conditions, where the
vertical-component anomaly simplifies to the same spectral form as
gravity with susceptibility contrast replacing density contrast.

Academic references
-------------------
Kennedy J, Eberhart RC (1995) Particle swarm optimization.
  Proc IEEE Int Conf Neural Netw 4:1942-1948.
  https://doi.org/10.1109/ICNN.1995.488968

Pallero JLG, Fernandez-Martinez JL, Bonvalot S, Fudym O (2015)
  Gravity inversion and uncertainty assessment of basement relief via
  Particle Swarm Optimization. J Appl Geophys 116:180-191.
  https://doi.org/10.1016/j.jappgeo.2015.03.008

Essa KS, Elhussein M (2018) PSO for interpretation of magnetic
  anomalies caused by simple geometrical structures. Pure Appl
  Geophys 175:3539-3553.
  https://doi.org/10.1007/s00024-018-1867-0

Telford WM, Geldart LP, Sheriff RE (1990) Applied Geophysics,
  2nd edn. Cambridge University Press, Cambridge.

Talwani M, Heirtzler JR (1964) Computation of magnetic anomalies
  caused by two-dimensional structures. In: Parks GA (ed)
  Computers in the Mineral Industries. Stanford Univ Publ, pp 464-480.

Author
------
Muhammet Ali Aygün
Istanbul Tecnical University&Istanbul University, Institute of Marine Sciences and Management
maygun@ogr.iu.edu.tr

Version : 1.0.0
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
class PSOParams:
    """Swarm hyper-parameters for PSO inversion.

    Attributes
    ----------
    n_particles : int
        Swarm size.  30–50 is typically sufficient for this problem.
    n_iterations : int
        Maximum number of iterations.
    inertia_w : float
        Inertia weight *w* (Shi and Eberhart 1998 recommend 0.72).
    cognitive_c1 : float
        Personal-best attraction coefficient.
    social_c2 : float
        Global-best attraction coefficient.
    param_min : float
        Lower bound for model parameters (g/cm³ or SI).
    param_max : float
        Upper bound for model parameters.
    rms_tol : float
        Early-stopping RMS threshold (mGal or nT).
    random_seed : int
        Seed for reproducibility; -1 disables seeding.
    """
    n_particles  : int   = 30
    n_iterations : int   = 100
    inertia_w    : float = 0.72
    cognitive_c1 : float = 1.50
    social_c2    : float = 1.50
    param_min    : float = -0.80
    param_max    : float =  0.80
    rms_tol      : float =  0.10
    random_seed  : int   = 42


@dataclass
class PSOResult:
    """Container for a single-profile PSO inversion result.

    Attributes
    ----------
    profile_lon : float
        Longitude of the N-S profile (degrees East).
    lat_obs : np.ndarray, shape (n_obs,)
        Observed latitudes along the profile.
    obs : np.ndarray, shape (n_obs,)
        Zero-mean observed anomaly (mGal or nT).
    pred : np.ndarray, shape (n_obs,)
        Best-fit predicted anomaly.
    model : np.ndarray, shape (n_layers, n_obs)
        Recovered parameter field (density contrast g/cm³ or
        susceptibility contrast SI).
    rms : float
        Final RMS misfit.
    convergence : list[float]
        RMS at each iteration.
    n_layers : int
        Number of depth layers.
    layer_tops_km : list[float]
        Top depth of each layer (km).
    layer_bots_km : list[float]
        Bottom depth of each layer (km).
    data_type : str
        'gravity' or 'magnetic'.
    """
    profile_lon   : float
    lat_obs       : np.ndarray
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
# Forward kernels
# ─────────────────────────────────────────────────────────────────────────────

def _gravity_kernel_prism(x_obs_km: np.ndarray,
                           z_top_km: float,
                           z_bot_km: float,
                           x_src_km: float,
                           dx_prism_km: float) -> np.ndarray:
    """Vertical gravity anomaly of a 2D horizontal prism (mGal per g/cm³).

    Uses the analytical formula from Telford et al. (1990) for a
    2D rectangular prism of infinite along-strike extent.

    Parameters
    ----------
    x_obs_km : array_like
        Observation positions in km (N-S distance from profile origin).
    z_top_km : float
        Depth to prism top (km, positive downward).
    z_bot_km : float
        Depth to prism bottom (km).
    x_src_km : float
        Horizontal position of prism centre (km).
    dx_prism_km : float
        Prism horizontal width (km).

    Returns
    -------
    np.ndarray
        Vertical gravity anomaly in mGal per g/cm³.
    """
    G_SI = 6.674e-11   # m³/(kg·s²)
    rho_SI_per_gcm3 = 1000.0
    mGal = 1e5         # m/s² → mGal

    x1 = (x_obs_km - x_src_km - dx_prism_km / 2.0) * 1e3   # m
    x2 = (x_obs_km - x_src_km + dx_prism_km / 2.0) * 1e3
    z1 = z_top_km * 1e3
    z2 = z_bot_km * 1e3

    def _F(x, z):
        r2 = x ** 2 + z ** 2 + 1e-9
        return x * np.arctan2(z, np.abs(x) + 1e-12) - z / 2.0 * np.log(r2)

    g = (2.0 * G_SI * rho_SI_per_gcm3
         * (_F(x2, z2) - _F(x2, z1) - _F(x1, z2) + _F(x1, z1)))
    return g * mGal


def _magnetic_kernel_prism(x_obs_km: np.ndarray,
                            z_top_km: float,
                            z_bot_km: float,
                            x_src_km: float,
                            dx_km: float,
                            scale: float = 100.0) -> np.ndarray:
    """Vertical magnetic anomaly of a 2D prism under RTP (nT per SI unit).

    For RTP-transformed data the inclination is effectively 90°, which
    simplifies the Talwani-Heirtzler formula to the same corner-function
    expression as gravity (Blakely 1995, eq. 6.17).

    Parameters
    ----------
    x_obs_km : array_like
        Observation positions (km).
    z_top_km, z_bot_km : float
        Layer depth bounds (km).
    x_src_km : float
        Source horizontal position (km).
    dx_km : float
        Source width (km).
    scale : float
        Physical scaling constant (≈100 nT·km/SI for RTP geometry).
        The kernel is further normalised by its maximum value in
        `build_magnetic_kernel`, so the exact value only affects the
        relative amplitude of the sensitivity matrix.

    Returns
    -------
    np.ndarray
        Magnetic anomaly in nT per SI susceptibility unit.
    """
    x1 = (x_obs_km - x_src_km - dx_km / 2.0) * 1e3
    x2 = (x_obs_km - x_src_km + dx_km / 2.0) * 1e3
    z1 = z_top_km * 1e3
    z2 = z_bot_km * 1e3

    def _F(x, z):
        r2 = x ** 2 + z ** 2 + 1e-9
        return z * np.arctan2(x, z) - x / 2.0 * np.log(r2)

    return scale * (_F(x2, z2) - _F(x2, z1) - _F(x1, z2) + _F(x1, z1))


# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity matrix builders
# ─────────────────────────────────────────────────────────────────────────────

def build_gravity_kernel(lat_obs: np.ndarray,
                          layer_tops_km: list[float],
                          layer_dzs_km: list[float],
                          dx_cell_km: float) -> np.ndarray:
    """Build the 2D gravity sensitivity matrix G.

    G has shape (n_obs, n_obs * n_layers).  Column j of layer l
    gives the gravity response at all observation points due to a
    unit-density prism at position j and depth layer l.

    Parameters
    ----------
    lat_obs : np.ndarray, shape (n_obs,)
        Observation latitudes (degrees).  Converted to km internally.
    layer_tops_km : list[float]
        Top depth of each layer (km).
    layer_dzs_km : list[float]
        Thickness of each layer (km).
    dx_cell_km : float
        Cell width (km), typically equal to the grid spacing.

    Returns
    -------
    np.ndarray, shape (n_obs, n_obs * n_layers)
    """
    n_obs = len(lat_obs)
    n_layers = len(layer_tops_km)
    x_obs = (lat_obs - lat_obs.min()) * 111.0   # degrees → km

    K = np.zeros((n_obs, n_obs * n_layers))
    for il, (z_top, dz) in enumerate(zip(layer_tops_km, layer_dzs_km)):
        for js in range(n_obs):
            K[:, il * n_obs + js] = _gravity_kernel_prism(
                x_obs, z_top, z_top + dz, x_obs[js], dx_cell_km)
    return K


def build_magnetic_kernel(lat_obs: np.ndarray,
                           layer_tops_km: list[float],
                           layer_dzs_km: list[float],
                           dx_cell_km: float) -> tuple[np.ndarray, float]:
    """Build the normalised 2D magnetic sensitivity matrix.

    Returns the matrix *and* the normalisation factor (max absolute
    value of the raw kernel) so that the recovered susceptibility can
    be scaled back to physical SI units.

    Parameters
    ----------
    lat_obs : np.ndarray, shape (n_obs,)
        Observation latitudes (degrees).
    layer_tops_km : list[float]
        Top depth of each layer (km).
    layer_dzs_km : list[float]
        Thickness of each layer (km).
    dx_cell_km : float
        Cell width (km).

    Returns
    -------
    K_norm : np.ndarray, shape (n_obs, n_obs * n_layers)
        Row-normalised sensitivity matrix.
    K_max : float
        Normalisation constant (raw kernel maximum).
    """
    n_obs = len(lat_obs)
    n_layers = len(layer_tops_km)
    x_obs = (lat_obs - lat_obs.min()) * 111.0

    K = np.zeros((n_obs, n_obs * n_layers))
    for il, (z_top, dz) in enumerate(zip(layer_tops_km, layer_dzs_km)):
        for js in range(n_obs):
            K[:, il * n_obs + js] = _magnetic_kernel_prism(
                x_obs, z_top, z_top + dz, x_obs[js], dx_cell_km)

    K_max = np.max(np.abs(K)) + 1e-30
    return K / K_max, K_max


# ─────────────────────────────────────────────────────────────────────────────
# Core PSO engine
# ─────────────────────────────────────────────────────────────────────────────

def _run_pso(K: np.ndarray,
             d_obs: np.ndarray,
             params: PSOParams) -> tuple[np.ndarray, np.ndarray, list, float]:
    """Core PSO loop.

    Minimises the mean-squared misfit ||K m - d_obs||² over the
    model vector *m* bounded to [param_min, param_max].

    Parameters
    ----------
    K : np.ndarray, shape (n_obs, n_model)
        Sensitivity matrix.
    d_obs : np.ndarray, shape (n_obs,)
        Observed (zero-mean) data vector.
    params : PSOParams
        Swarm hyper-parameters.

    Returns
    -------
    gbest : np.ndarray, shape (n_model,)
        Global best model vector.
    pred : np.ndarray, shape (n_obs,)
        Predicted data for the global best model.
    history : list[float]
        Per-iteration global-best RMS.
    rms_final : float
        Final RMS misfit.
    """
    rng = (np.random.default_rng(params.random_seed)
           if params.random_seed >= 0 else np.random.default_rng())

    n_m = K.shape[1]
    pos = rng.uniform(params.param_min, params.param_max,
                      (params.n_particles, n_m))
    vel = np.zeros_like(pos)

    pbest      = pos.copy()
    pbest_cost = np.full(params.n_particles, np.inf)
    gbest      = np.zeros(n_m)
    gbest_cost = np.inf
    history    = []

    for _ in range(params.n_iterations):
        # Evaluate cost for each particle
        preds = pos @ K.T                               # (n_p, n_obs)
        costs = np.mean((d_obs - preds) ** 2, axis=1)  # (n_p,)

        # Update personal and global bests
        improved = costs < pbest_cost
        pbest[improved] = pos[improved]
        pbest_cost[improved] = costs[improved]

        idx_best = np.argmin(costs)
        if costs[idx_best] < gbest_cost:
            gbest      = pos[idx_best].copy()
            gbest_cost = costs[idx_best]

        # Velocity and position update (standard PSO)
        r1 = rng.random((params.n_particles, n_m))
        r2 = rng.random((params.n_particles, n_m))
        vel = (params.inertia_w * vel
               + params.cognitive_c1 * r1 * (pbest - pos)
               + params.social_c2   * r2 * (gbest - pos))
        pos = np.clip(pos + vel, params.param_min, params.param_max)

        rms = np.sqrt(gbest_cost)
        history.append(rms)
        if rms < params.rms_tol:
            break

    pred      = K @ gbest
    rms_final = np.sqrt(np.mean((d_obs - pred) ** 2))
    return gbest, pred, history, rms_final


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_pso_gravity_profile(grid_column: np.ndarray,
                             lats: np.ndarray,
                             profile_lon: float,
                             layer_tops_km: list[float],
                             layer_dzs_km: list[float],
                             dx_grid_km: float,
                             n_subsample: int = 60,
                             params: Optional[PSOParams] = None) -> PSOResult:
    """Run PSO gravity inversion on a single N-S profile.

    Extracts a column from the Bouguer gravity grid, subsamples it to
    *n_subsample* evenly spaced observations, builds the 2D prism
    sensitivity matrix, runs PSO, and returns a :class:`PSOResult`.

    Parameters
    ----------
    grid_column : np.ndarray, shape (ny,)
        Bouguer anomaly values along the N-S column (mGal).
    lats : np.ndarray, shape (ny,)
        Latitude values corresponding to the column.
    profile_lon : float
        Longitude of the profile (degrees East).
    layer_tops_km : list[float]
        Top depths of inversion layers (km).
    layer_dzs_km : list[float]
        Thicknesses of inversion layers (km).
    dx_grid_km : float
        Grid cell spacing (km).
    n_subsample : int
        Number of observations to use (subsampling the full column).
    params : PSOParams, optional
        Swarm parameters.  Uses defaults if None.

    Returns
    -------
    PSOResult
    """
    if params is None:
        params = PSOParams()

    valid     = ~np.isnan(grid_column)
    lat_v     = lats[valid]
    grav_v    = grid_column[valid]
    g_mean    = grav_v.mean()
    g_anom    = grav_v - g_mean

    # Subsample
    n_sub = min(len(lat_v), n_subsample)
    idx   = np.round(np.linspace(0, len(lat_v) - 1, n_sub)).astype(int)
    lat_s = lat_v[idx]
    obs_s = g_anom[idx]

    K = build_gravity_kernel(lat_s, layer_tops_km, layer_dzs_km, dx_grid_km)
    gbest, pred, history, rms = _run_pso(K, obs_s, params)

    n_layers  = len(layer_tops_km)
    model_2d  = gbest.reshape(n_layers, n_sub)

    return PSOResult(
        profile_lon   = profile_lon,
        lat_obs       = lat_s,
        obs           = obs_s,
        pred          = pred,
        model         = model_2d,
        rms           = rms,
        convergence   = history,
        n_layers      = n_layers,
        layer_tops_km = layer_tops_km,
        layer_bots_km = [t + d for t, d in zip(layer_tops_km, layer_dzs_km)],
        data_type     = 'gravity',
    )


def run_pso_magnetic_profile(grid_column: np.ndarray,
                              lats: np.ndarray,
                              profile_lon: float,
                              layer_tops_km: list[float],
                              layer_dzs_km: list[float],
                              dx_grid_km: float,
                              n_subsample: int = 60,
                              params: Optional[PSOParams] = None) -> PSOResult:
    """Run PSO magnetic inversion on a single N-S profile.

    Identical workflow to :func:`run_pso_gravity_profile` but uses the
    normalised magnetic sensitivity matrix.  The observed anomaly is
    standardised before inversion and rescaled to nT in the returned
    result.

    Parameters
    ----------
    grid_column : np.ndarray, shape (ny,)
        RTP total-field anomaly values along the column (nT).
    lats, profile_lon, layer_tops_km, layer_dzs_km, dx_grid_km,
    n_subsample, params : see :func:`run_pso_gravity_profile`.

    Returns
    -------
    PSOResult
        ``data_type = 'magnetic'``; ``obs`` and ``pred`` are in nT;
        ``model`` values are normalised susceptibility contrasts (SI,
        bounded to ±params.param_max).
    """
    if params is None:
        params = PSOParams(param_min=-0.15, param_max=0.15)

    valid   = ~np.isnan(grid_column)
    lat_v   = lats[valid]
    mag_v   = grid_column[valid]
    m_mean  = mag_v.mean()
    m_anom  = mag_v - m_mean

    n_sub = min(len(lat_v), n_subsample)
    idx   = np.round(np.linspace(0, len(lat_v) - 1, n_sub)).astype(int)
    lat_s = lat_v[idx]
    obs_s = m_anom[idx]

    # Normalise observed anomaly and build normalised kernel
    a_std    = np.std(obs_s) + 1e-10
    obs_norm = obs_s / a_std
    K, _     = build_magnetic_kernel(lat_s, layer_tops_km, layer_dzs_km, dx_grid_km)

    gbest, pred_norm, history, rms_norm = _run_pso(K, obs_norm, params)

    pred_nT = pred_norm * a_std
    rms_nT  = rms_norm * a_std

    n_layers = len(layer_tops_km)
    model_2d = gbest.reshape(n_layers, n_sub)

    return PSOResult(
        profile_lon   = profile_lon,
        lat_obs       = lat_s,
        obs           = obs_s,
        pred          = pred_nT,
        model         = model_2d,
        rms           = rms_nT,
        convergence   = history,
        n_layers      = n_layers,
        layer_tops_km = layer_tops_km,
        layer_bots_km = [t + d for t, d in zip(layer_tops_km, layer_dzs_km)],
        data_type     = 'magnetic',
    )


def run_pso_multi_profile(grid: np.ndarray,
                           lons: np.ndarray,
                           lats: np.ndarray,
                           n_profiles: int,
                           layer_tops_km: list[float],
                           layer_dzs_km: list[float],
                           dx_grid_km: float,
                           data_type: str = 'gravity',
                           n_subsample: int = 60,
                           params: Optional[PSOParams] = None,
                           verbose: bool = True) -> list[PSOResult]:
    """Run PSO inversion on *n_profiles* evenly spaced N-S profiles.

    Parameters
    ----------
    grid : np.ndarray, shape (ny, nx)
        Bouguer gravity (mGal) or RTP magnetic (nT) grid.
    lons : np.ndarray, shape (nx,)
        Longitude values of grid columns.
    lats : np.ndarray, shape (ny,)
        Latitude values of grid rows.
    n_profiles : int
        Number of equally spaced N-S profiles to invert.
    data_type : {'gravity', 'magnetic'}
        Selects the forward kernel.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    list[PSOResult]
        One result per profile, ordered west to east.
    """
    ny, nx = grid.shape
    col_indices = np.round(np.linspace(20, nx - 21, n_profiles)).astype(int)

    runner = (run_pso_gravity_profile if data_type == 'gravity'
              else run_pso_magnetic_profile)

    results = []
    for ip, cidx in enumerate(col_indices):
        plon = float(lons[cidx])
        if verbose:
            print(f'  Profile {ip + 1}/{n_profiles}  lon={plon:.3f}°E', flush=True)
        result = runner(
            grid_column   = grid[:, cidx],
            lats          = lats,
            profile_lon   = plon,
            layer_tops_km = layer_tops_km,
            layer_dzs_km  = layer_dzs_km,
            dx_grid_km    = dx_grid_km,
            n_subsample   = n_subsample,
            params        = params,
        )
        if verbose:
            print(f'    RMS={result.rms:.4f}  '
                  f'model range {result.model.min():.4f}–{result.model.max():.4f}')
        results.append(result)

    return results
