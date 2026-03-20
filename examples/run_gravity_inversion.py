#!/usr/bin/env python3
"""
examples/run_gravity_inversion.py
==================================
Full gravity inversion example for the southern Marmara Sea
dataset.

This script demonstrates:
  1. Loading a pre-processed Bouguer gravity grid
  2. RAPS spectral depth estimation
  3. PSO 2D inversion on 10 N-S profiles
  4. ABIC 3D spectral inversion
  5. Li-Oldenburg IRLS inversion along a reference profile
  6. Saving all results to .npy files

Expected input
--------------
A .npy file (or any 2D array) with keys / structure matching the
output of the GeoMagPro digitisation and gridding pipeline
(Aygün and Demirel, in press-a).  The minimum required fields are:

    GS   : np.ndarray (ny, nx)   Bouguer anomaly (mGal)
    lons : np.ndarray (nx,)      Longitude array
    lats : np.ndarray (ny,)      Latitude array
    dx   : float                 Cell size in degrees

Usage
-----
    python examples/run_gravity_inversion.py --grid path/to/grids.npy

Academic references for this workflow
--------------------------------------
Kennedy & Eberhart (1995); Pallero et al. (2015) — PSO
Akaike (1980); Murata (1993) — ABIC
Li & Oldenburg (1996, 1998) — Li-Oldenburg IRLS
Spector & Grant (1970) — RAPS

Author: Muhammet Ali Aygün  |  Istanbul University & Istanbul Tecnical University  |  2025-09
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geomagpro import (
    PSOParams, run_pso_multi_profile,
    ABICParams, run_abic_gravity,
    LiOldenburgParams, run_li_oldenburg,
    raps_depth, tilt_derivative, total_horizontal_derivative,
    extract_tdr_lineaments,
)


def parse_args():
    p = argparse.ArgumentParser(description='Gravity inversion — southern Marmara Sea')
    p.add_argument('--grid', default='grids_v2.npy',
                   help='Path to preprocessed grid .npy file')
    p.add_argument('--out',  default='results_gravity',
                   help='Output directory for results')
    p.add_argument('--n-profiles', type=int, default=10,
                   help='Number of PSO N-S profiles')
    p.add_argument('--pso-iter', type=int, default=100,
                   help='PSO iterations per profile')
    p.add_argument('--skip-li', action='store_true',
                   help='Skip Li-Oldenburg (faster testing)')
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load grid ─────────────────────────────────────────────────────────
    print(f'Loading grid: {args.grid}')
    data  = np.load(args.grid, allow_pickle=True).item()
    GS    = data['GS']    # shape (ny, nx)
    lons  = data['lons']  # shape (nx,)
    lats  = data['lats']  # shape (ny,)
    dx_deg = float(data['dx'])
    dx_km  = dx_deg * 111.0

    print(f'  Grid: {GS.shape}  |  Bouguer range {np.nanmin(GS):.1f}–{np.nanmax(GS):.1f} mGal')

    # ── RAPS depth estimation ─────────────────────────────────────────────
    print('\n=== RAPS depth estimation ===')
    raps = raps_depth(GS, dx_km=dx_km, n_segments=4, verbose=True)
    np.save(out_dir / 'raps_gravity.npy', raps)
    print(f'  Depth horizons: {[f"{d:.2f} km" for d in raps["depths_km"]]}')

    # Use RAPS horizons for inversion layers
    # These match the published values for the southern Marmara Sea
    # (Aygün and Demirel, in press-b)
    layer_tops = [0.52, 3.2,  8.0, 15.0]
    layer_dzs  = [2.68, 4.8,  7.0, 10.0]

    # ── PSO inversion ─────────────────────────────────────────────────────
    print(f'\n=== PSO gravity inversion ({args.n_profiles} profiles, '
          f'{args.pso_iter} iterations) ===')
    pso_params = PSOParams(
        n_particles  = 25,
        n_iterations = args.pso_iter,
        inertia_w    = 0.72,
        cognitive_c1 = 1.50,
        social_c2    = 1.50,
        param_min    = -0.80,
        param_max    =  0.80,
    )
    pso_results = run_pso_multi_profile(
        grid          = GS,
        lons          = lons,
        lats          = lats,
        n_profiles    = args.n_profiles,
        layer_tops_km = layer_tops,
        layer_dzs_km  = layer_dzs,
        dx_grid_km    = dx_km,
        data_type     = 'gravity',
        params        = pso_params,
        verbose       = True,
    )
    np.save(out_dir / 'pso_gravity_results.npy', pso_results, allow_pickle=True)
    rms_vals = [r.rms for r in pso_results]
    print(f'  PSO RMS range: {min(rms_vals):.3f}–{max(rms_vals):.3f} mGal')

    # ── ABIC 3D inversion ─────────────────────────────────────────────────
    print('\n=== ABIC 3D gravity inversion ===')
    abic_params = ABICParams(
        log_omega_min = 1.0,
        log_omega_max = 8.0,
        n_grid_search = 30,
        param_clip    = 0.80,
    )
    abic_result = run_abic_gravity(
        grid           = GS,
        lons           = lons,
        lats           = lats,
        dx_deg         = dx_deg,
        depths_km      = layer_tops,
        dzs_km         = layer_dzs,
        subsample_step = 4,
        params         = abic_params,
        verbose        = True,
    )
    np.save(out_dir / 'abic_gravity_results.npy',
            {
                'kappa_layers'  : abic_result.model_layers,
                'pred_layers'   : abic_result.pred_layers,
                'rms_layers'    : abic_result.rms_layers,
                'lo_sub'        : abic_result.lo_sub,
                'la_sub'        : abic_result.la_sub,
                'opt_omega'     : abic_result.opt_omega,
                'opt_log_omega' : abic_result.opt_log_omega,
                'log_omegas'    : abic_result.abic_grid_log_omegas,
                'abic_scores'   : abic_result.abic_grid_scores,
                'depths'        : abic_result.depths_km,
                'dzs'           : abic_result.dzs_km,
            },
            allow_pickle=True)

    # ── Li-Oldenburg IRLS ─────────────────────────────────────────────────
    if not args.skip_li:
        print('\n=== Li-Oldenburg IRLS gravity inversion ===')
        # Extract profile A-A' at lat ≈ 40.35°N
        lat_target = 40.35
        row_idx    = int(np.argmin(np.abs(lats - lat_target)))
        profile    = GS[row_idx, :]
        valid      = ~np.isnan(profile)
        lon_v      = lons[valid]
        prof_v     = profile[valid]
        prof_mean  = prof_v.mean()
        prof_anom  = prof_v - prof_mean

        # Subsample to 150 points for tractable matrix
        n_sub = min(len(lon_v), 150)
        idx   = np.round(np.linspace(0, len(lon_v) - 1, n_sub)).astype(int)

        lo_params = LiOldenburgParams(
            n_iterations          = 100,
            regularisation_lambda = 0.50,
            depth_weight_beta     = 1.50,
            param_clip            = 0.80,
        )
        li_result = run_li_oldenburg(
            anomaly_profile = prof_anom[idx],
            lon_profile     = lon_v[idx],
            layer_tops_km   = [0.52, 3.2, 8.0, 15.0, 25.0],
            layer_dzs_km    = [2.68, 4.8, 7.0, 10.0, 12.0],
            dx_grid_km      = dx_km,
            data_type       = 'gravity',
            params          = lo_params,
            verbose         = True,
        )
        np.save(out_dir / 'li_oldenburg_gravity.npy', {
            'lon_arr'    : li_result.lon_profile,
            'obs'        : li_result.obs,
            'pred'       : li_result.pred,
            'model'      : li_result.model,
            'rms'        : li_result.rms,
            'convergence': li_result.convergence,
            'lat_prof'   : float(lats[row_idx]),
        }, allow_pickle=True)
        print(f'  Li-Oldenburg final RMS: {li_result.rms:.4f} mGal')

    # ── TDR / THD structural mapping ──────────────────────────────────────
    print('\n=== Structural derivative maps ===')
    tdr = tilt_derivative(GS, dx_km=dx_km)
    thd = total_horizontal_derivative(GS, dx_km=dx_km)
    lin = extract_tdr_lineaments(tdr, thd, lons, lats, thd_percentile=75)

    np.save(out_dir / 'tdr_gravity.npy', tdr)
    np.save(out_dir / 'thd_gravity.npy', thd)
    np.save(out_dir / 'lineaments_gravity.npy', lin)
    print(f'  TDR range: {tdr.min():.1f}–{tdr.max():.1f}°')
    print(f'  Lineament points: {len(lin)}')

    print(f'\nAll results saved to: {out_dir.resolve()}')


if __name__ == '__main__':
    main()
