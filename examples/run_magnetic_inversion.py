#!/usr/bin/env python3
"""
examples/run_magnetic_inversion.py
===================================
Ful magnetic inversion example for the southern Marmara Sea
dataset (companion paper to gravity analysis).

Demonstrates:
  1. Loading a pre-processed RTP aeromagnetic grid
  2. PSO 2D inversion on 10 N-S profiles
  3. ABIC 3D spectral inversion
  4. Li-Oldenburg IRLS along profile A-A'
  5. Magnetic lineament extraction

Usage
-----
    python examples/run_magnetic_inversion.py --grid path/to/grids.npy

Author: Muhammet Ali Aygün  |  Istanbul University  |  2026-03
"""

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geomagpro import (
    PSOParams, run_pso_multi_profile,
    ABICParams, run_abic_magnetic,
    LiOldenburgParams, run_li_oldenburg,
    raps_depth, tilt_derivative, total_horizontal_derivative,
    extract_tdr_lineaments,
)


def parse_args():
    p = argparse.ArgumentParser(description='Magnetic inversion — southern Marmara Sea')
    p.add_argument('--grid',       default='grids_v2.npy')
    p.add_argument('--out',        default='results_magnetic')
    p.add_argument('--n-profiles', type=int, default=10)
    p.add_argument('--pso-iter',   type=int, default=100)
    p.add_argument('--skip-li',    action='store_true')
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading grid: {args.grid}')
    data   = np.load(args.grid, allow_pickle=True).item()
    MS     = data['MS']    # RTP magnetic anomaly (nT)
    lons   = data['lons']
    lats   = data['lats']
    dx_deg = float(data['dx'])
    dx_km  = dx_deg * 111.0

    print(f'  RTP range: {np.nanmin(MS):.1f}–{np.nanmax(MS):.1f} nT')

    # RAPS-guided layer geometry for the southern Marmara Sea
    # (verified against companion gravity RAPS — Aygün & Demirel in press-b)
    layer_tops = [0.52,  3.2, 10.0, 15.0]
    layer_dzs  = [2.68,  6.8,  5.0, 10.0]

    # ── RAPS ──────────────────────────────────────────────────────────────
    print('\n=== Magnetic RAPS depth estimation ===')
    raps = raps_depth(MS, dx_km=dx_km, n_segments=3, verbose=True)
    np.save(out_dir / 'raps_magnetic.npy', raps)

    # ── PSO ───────────────────────────────────────────────────────────────
    print(f'\n=== PSO magnetic inversion ({args.n_profiles} profiles) ===')
    pso_params = PSOParams(
        n_particles  = 30,
        n_iterations = args.pso_iter,
        inertia_w    = 0.72,
        cognitive_c1 = 1.50,
        social_c2    = 1.50,
        param_min    = -0.15,   # SI susceptibility bounds
        param_max    =  0.15,
    )
    pso_results = run_pso_multi_profile(
        grid          = MS,
        lons          = lons,
        lats          = lats,
        n_profiles    = args.n_profiles,
        layer_tops_km = layer_tops,
        layer_dzs_km  = layer_dzs,
        dx_grid_km    = dx_km,
        data_type     = 'magnetic',
        params        = pso_params,
        verbose       = True,
    )
    np.save(out_dir / 'pso_magnetic_results.npy', pso_results, allow_pickle=True)
    rms_vals = [r.rms for r in pso_results]
    print(f'  RMS range: {min(rms_vals):.1f}–{max(rms_vals):.1f} nT')

    # ── ABIC ──────────────────────────────────────────────────────────────
    print('\n=== ABIC 3D magnetic inversion ===')
    abic_result = run_abic_magnetic(
        grid           = MS,
        lons           = lons,
        lats           = lats,
        dx_deg         = dx_deg,
        depths_km      = layer_tops,
        dzs_km         = layer_dzs,
        subsample_step = 4,
        params         = ABICParams(log_omega_min=1.0,
                                    log_omega_max=8.0,
                                    param_clip=0.15),
        verbose        = True,
    )
    np.save(out_dir / 'abic_magnetic_results.npy', {
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
    }, allow_pickle=True)

    # ── Li-Oldenburg ──────────────────────────────────────────────────────
    if not args.skip_li:
        print('\n=== Li-Oldenburg IRLS magnetic inversion ===')
        lat_target = 40.35
        row_idx    = int(np.argmin(np.abs(lats - lat_target)))
        profile    = MS[row_idx, :]
        valid      = ~np.isnan(profile)
        lon_v      = lons[valid]
        prof_anom  = profile[valid] - profile[valid].mean()

        n_sub = min(len(lon_v), 150)
        idx   = np.round(np.linspace(0, len(lon_v) - 1, n_sub)).astype(int)

        li_result = run_li_oldenburg(
            anomaly_profile = prof_anom[idx],
            lon_profile     = lon_v[idx],
            layer_tops_km   = [0.52, 3.2, 10.0, 15.0, 25.0],
            layer_dzs_km    = [2.68, 6.8,  5.0, 10.0, 12.0],
            dx_grid_km      = dx_km,
            data_type       = 'magnetic',
            params          = LiOldenburgParams(
                n_iterations=100,
                regularisation_lambda=0.30,
                depth_weight_beta=3.0,   # z^(-3) for magnetic
            ),
            verbose=True,
        )
        np.save(out_dir / 'li_oldenburg_magnetic.npy', {
            'lon_arr'    : li_result.lon_profile,
            'obs'        : li_result.obs,
            'pred'       : li_result.pred,
            'model'      : li_result.model,
            'rms'        : li_result.rms,
            'convergence': li_result.convergence,
            'lat_prof'   : float(lats[row_idx]),
        }, allow_pickle=True)
        print(f'  Final RMS: {li_result.rms:.4f} nT')

    # ── TDR / lineaments ──────────────────────────────────────────────────
    print('\n=== Magnetic structural maps ===')
    tdr = tilt_derivative(MS, dx_km=dx_km)
    thd = total_horizontal_derivative(MS, dx_km=dx_km)
    lin = extract_tdr_lineaments(tdr, thd, lons, lats, thd_percentile=75)
    np.save(out_dir / 'tdr_magnetic.npy', tdr)
    np.save(out_dir / 'thd_magnetic.npy', thd)
    np.save(out_dir / 'lineaments_magnetic.npy', lin)
    print(f'  Magnetic lineament points: {len(lin)}')

    print(f'\nAll results saved to: {out_dir.resolve()}')


if __name__ == '__main__':
    main()
