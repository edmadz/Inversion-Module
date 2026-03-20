"""
geomagpro.grid_processing
=========================
Spectral processing and derivative operators for gridded gravity and
aeromagnetic data.

Includes:
- Radial Average Power Spectrum (RAPS) depth estimation
- Upward continuation
- Tilt Derivative (TDR)
- Total Horizontal Derivative (THD)
- Source depth estimation (TDR half-width method)

Academic references
-------------------
Spector A, Grant FS (1970) Statistical models for interpreting
  aeromagnetic data. Geophysics 35:293-302.
  https://doi.org/10.1190/1.1440092

Salem A, Williams S, Fairhead D, Smith R, Ravat D (2007)
  Interpretation of magnetic data using tilt-angle derivatives.
  Geophysics 73:L1-L10. https://doi.org/10.1190/1.2799992

Miller HG, Singh V (1994) Potential field tilt — a new concept for
  location of potential field sources. J Appl Geophys 32:213-217.
  https://doi.org/10.1016/0926-9851(94)90022-1

Blakely RJ (1995) Potential Theory in Gravity and Magnetic
  Applications. Cambridge University Press, Cambridge.

Author
------
Muhammet Ali Aygün
Istanbul University& Istanbul Tecnical University, Departmant of Marine Geology and Geophysics
maygun@ogr.iu.edu.tr

Version : 1.0.0
Date    : 2026-03
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def raps_depth(grid: np.ndarray,
               dx_km: float,
               n_segments: int = 4,
               smooth_sigma: float = 1.0,
               verbose: bool = True) -> dict:
    """Radial Average Power Spectrum (RAPS) depth estimation.

    Follows the method of Spector and Grant (1970): the logarithm of
    the radially averaged power spectrum is approximately linear in
    wavenumber, with slope = −4π × mean source depth.

    Parameters
    ----------
    grid : np.ndarray, shape (ny, nx)
        Gravity or magnetic anomaly grid.
    dx_km : float
        Isotropic cell spacing (km).
    n_segments : int
        Number of linear segments to fit (= number of depth horizons).
    smooth_sigma : float
        Gaussian smoothing σ applied to the power spectrum before
        segmentation to reduce spectral noise.
    verbose : bool

    Returns
    -------
    dict with keys:
        'wavenumbers' : np.ndarray
            Radial wavenumbers (cycles/km).
        'log_power' : np.ndarray
            log₁₀ of radially averaged power.
        'depths_km' : list[float]
            Estimated mean source depths from each linear segment.
        'slopes' : list[float]
            Fitted log-power slopes.
        'intercepts' : list[float]
            Fitted intercepts.
    """
    ny, nx = grid.shape
    g_clean = np.nan_to_num(grid, nan=float(np.nanmean(grid)))

    # Hanning window to suppress spectral leakage
    wy = np.hanning(ny)
    wx = np.hanning(nx)
    W  = np.outer(wy, wx)
    g_win = g_clean * W

    F  = np.fft.fft2(g_win)
    P  = np.abs(F) ** 2

    ky = np.fft.fftfreq(ny, d=dx_km)
    kx = np.fft.fftfreq(nx, d=dx_km)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)

    # Radial binning
    k_max  = 0.5 / dx_km   # Nyquist
    k_bins = np.linspace(0, k_max * 0.9, 80)
    k_mid  = (k_bins[:-1] + k_bins[1:]) / 2
    log_P  = []
    for k0, k1 in zip(k_bins[:-1], k_bins[1:]):
        mask = (K >= k0) & (K < k1)
        if mask.sum() > 0:
            log_P.append(np.log10(P[mask].mean()))
        else:
            log_P.append(np.nan)

    log_P  = np.array(log_P)
    valid  = ~np.isnan(log_P)
    k_mid  = k_mid[valid]
    log_P  = log_P[valid]

    if smooth_sigma > 0:
        log_P = gaussian_filter(log_P, sigma=smooth_sigma)

    # Divide wavenumber range into n_segments equal parts and fit lines
    n_pts   = len(k_mid)
    seg_len = n_pts // n_segments
    depths, slopes, intercepts = [], [], []

    for s in range(n_segments):
        i0 = s * seg_len
        i1 = i0 + seg_len
        k_seg = k_mid[i0:i1]; p_seg = log_P[i0:i1]
        coeffs = np.polyfit(k_seg, p_seg, 1)
        slope = coeffs[0]
        # depth = |slope| / (4π × log₁₀(e))
        #       = |slope| * ln(10) / (4π)
        depth_km = abs(slope) * np.log(10) / (4 * np.pi)
        depths.append(depth_km)
        slopes.append(slope)
        intercepts.append(coeffs[1])
        if verbose:
            print(f'  Segment {s+1}: slope={slope:.4f}  depth={depth_km:.3f} km')

    return {
        'wavenumbers' : k_mid,
        'log_power'   : log_P,
        'depths_km'   : depths,
        'slopes'      : slopes,
        'intercepts'  : intercepts,
    }


def upward_continue(grid: np.ndarray, dx_km: float,
                    height_km: float) -> np.ndarray:
    """Upward continuation of a potential field grid.

    Multiplies the 2D Fourier spectrum by the continuation operator
    exp(−2π |k| h), where h is the continuation height and k is the
    radial wavenumber.

    Parameters
    ----------
    grid : np.ndarray, shape (ny, nx)
        Input anomaly grid.
    dx_km : float
        Cell spacing (km).
    height_km : float
        Upward continuation height (km, positive upward).

    Returns
    -------
    np.ndarray
        Upward-continued grid (same shape).
    """
    ny, nx = grid.shape
    g_clean = np.nan_to_num(grid, nan=float(np.nanmean(grid)))

    ky = np.fft.fftfreq(ny, d=dx_km)
    kx = np.fft.fftfreq(nx, d=dx_km)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)

    F   = np.fft.fft2(g_clean)
    F_c = F * np.exp(-2 * np.pi * K * height_km)
    return np.real(np.fft.ifft2(F_c))


def tilt_derivative(grid: np.ndarray, dx_km: float) -> np.ndarray:
    """Tilt Derivative (TDR) of a potential field grid.

    TDR = arctan( VD / THD )

    where VD is the vertical derivative and THD is the total horizontal
    derivative.  The zero contour of TDR marks density or susceptibility
    contacts (Miller and Singh 1994; Salem et al. 2007).

    Parameters
    ----------
    grid : np.ndarray, shape (ny, nx)
        Input anomaly grid.
    dx_km : float
        Cell spacing (km).

    Returns
    -------
    np.ndarray
        TDR in degrees.
    """
    ny, nx = grid.shape
    g_clean = np.nan_to_num(grid, nan=float(np.nanmean(grid)))

    ky = np.fft.fftfreq(ny, d=dx_km)
    kx = np.fft.fftfreq(nx, d=dx_km)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)
    K[K == 0] = 1e-12

    F   = np.fft.fft2(g_clean)
    # x and y horizontal derivatives
    dx_g = np.real(np.fft.ifft2(1j * 2 * np.pi * KX * F))
    dy_g = np.real(np.fft.ifft2(1j * 2 * np.pi * KY * F))
    # Vertical derivative
    dz_g = np.real(np.fft.ifft2(2 * np.pi * K * F))

    thd = np.sqrt(dx_g ** 2 + dy_g ** 2) + 1e-12
    tdr = np.degrees(np.arctan2(np.abs(dz_g), thd))

    # Sign: TDR < 0 over source, > 0 outside (Salem et al. 2007)
    # Here we return the signed version so zero contour marks contacts
    tdr_signed = np.degrees(np.arctan2(dz_g, thd))
    return tdr_signed


def total_horizontal_derivative(grid: np.ndarray,
                                 dx_km: float) -> np.ndarray:
    """Total Horizontal Derivative (THD) of a potential field grid.

    THD = sqrt( (∂g/∂x)² + (∂g/∂y)² )

    Maxima of THD locate the edges of density or susceptibility bodies
    (Roest et al. 1992; Nabighian et al. 2005).

    Parameters
    ----------
    grid : np.ndarray, shape (ny, nx)
        Input anomaly grid.
    dx_km : float
        Cell spacing (km).

    Returns
    -------
    np.ndarray
        THD in units of [input units / km].
    """
    ny, nx = grid.shape
    g_clean = np.nan_to_num(grid, nan=float(np.nanmean(grid)))

    ky = np.fft.fftfreq(ny, d=dx_km)
    kx = np.fft.fftfreq(nx, d=dx_km)
    KX, KY = np.meshgrid(kx, ky)

    F   = np.fft.fft2(g_clean)
    dx_g = np.real(np.fft.ifft2(1j * 2 * np.pi * KX * F))
    dy_g = np.real(np.fft.ifft2(1j * 2 * np.pi * KY * F))
    return np.sqrt(dx_g ** 2 + dy_g ** 2)


def extract_tdr_lineaments(tdr_grid: np.ndarray,
                            thd_grid: np.ndarray,
                            lons: np.ndarray,
                            lats: np.ndarray,
                            thd_percentile: float = 75.0) -> np.ndarray:
    """Extract structural lineament points from TDR zero-crossings.

    Points where the TDR changes sign AND the THD exceeds a given
    percentile threshold are interpreted as potential-field structural
    contacts (density or susceptibility boundaries).

    Parameters
    ----------
    tdr_grid : np.ndarray, shape (ny, nx)
        Tilt derivative grid (degrees).
    thd_grid : np.ndarray, shape (ny, nx)
        Total horizontal derivative grid.
    lons : np.ndarray, shape (nx,)
        Longitude array.
    lats : np.ndarray, shape (ny,)
        Latitude array.
    thd_percentile : float
        THD threshold percentile (e.g. 75 keeps the most significant
        contacts).

    Returns
    -------
    np.ndarray, shape (N, 3)
        Columns: [longitude, latitude, thd_value] for each lineament
        point.
    """
    thd_thr = np.nanpercentile(thd_grid, thd_percentile)
    points  = []

    for row in range(tdr_grid.shape[0]):
        tdr_row = np.nan_to_num(tdr_grid[row])
        zc      = np.where(np.diff(np.sign(tdr_row)))[0]
        for z in zc:
            thd_val = float(thd_grid[row, z])
            if thd_val >= thd_thr:
                lo_c = float((lons[z] + lons[min(z + 1, len(lons) - 1)]) / 2)
                la_c = float(lats[row])
                points.append((lo_c, la_c, thd_val))

    return np.array(points) if points else np.empty((0, 3))
