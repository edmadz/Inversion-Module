"""
tests/test_inversions.py
========================
Unit tests for GeoMagPro inversion modules.

Run with:
    pytest tests/test_inversions.py -v

All tests use synthetic data so no real dataset files are required.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geomagpro.pso_inversion import (
    PSOParams, run_pso_gravity_profile, run_pso_magnetic_profile,
    build_gravity_kernel, build_magnetic_kernel)
from geomagpro.abic_inversion import (
    ABICParams, run_abic_gravity, run_abic_magnetic,
    _gravity_spectral_kernel, _magnetic_spectral_kernel)
from geomagpro.li_oldenburg import (
    LiOldenburgParams, run_li_oldenburg)
from geomagpro.grid_processing import (
    raps_depth, upward_continue,
    tilt_derivative, total_horizontal_derivative,
    extract_tdr_lineaments)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_grid(ny=30, nx=60, seed=42):
    """Generate a smooth synthetic Bouguer / RTP anomaly grid."""
    rng = np.random.default_rng(seed)
    x   = np.linspace(0, 2 * np.pi, nx)
    y   = np.linspace(0, np.pi, ny)
    X, Y = np.meshgrid(x, y)
    grid = 20 * np.sin(X) * np.cos(Y) + 5 * rng.standard_normal((ny, nx))
    lons = np.linspace(27.0, 29.4, nx)
    lats = np.linspace(40.17, 40.76, ny)
    return grid, lons, lats, 0.04   # grid, lons, lats, dx_deg


def make_profile(ny=30):
    """Extract a synthetic N-S profile."""
    grid, _, lats, dx_deg = make_synthetic_grid(ny=ny)
    col = grid[:, 15]
    return col, lats, dx_deg


# ─────────────────────────────────────────────────────────────────────────────
# Grid processing
# ─────────────────────────────────────────────────────────────────────────────

class TestGridProcessing:
    def setup_method(self):
        self.grid, self.lons, self.lats, self.dx_deg = make_synthetic_grid()
        self.dx_km = self.dx_deg * 111.0

    def test_upward_continue_shape(self):
        uc = upward_continue(self.grid, self.dx_km, height_km=2.0)
        assert uc.shape == self.grid.shape

    def test_upward_continue_attenuates(self):
        uc = upward_continue(self.grid, self.dx_km, height_km=5.0)
        assert np.std(uc) < np.std(self.grid)

    def test_tdr_shape(self):
        tdr = tilt_derivative(self.grid, self.dx_km)
        assert tdr.shape == self.grid.shape

    def test_tdr_range(self):
        tdr = tilt_derivative(self.grid, self.dx_km)
        assert tdr.min() >= -90.0
        assert tdr.max() <=  90.0

    def test_thd_nonnegative(self):
        thd = total_horizontal_derivative(self.grid, self.dx_km)
        assert np.all(thd >= 0)

    def test_raps_returns_depths(self):
        result = raps_depth(self.grid, self.dx_km, n_segments=3, verbose=False)
        assert 'depths_km' in result
        assert len(result['depths_km']) == 3
        assert all(d > 0 for d in result['depths_km'])

    def test_lineament_extraction(self):
        tdr = tilt_derivative(self.grid, self.dx_km)
        thd = total_horizontal_derivative(self.grid, self.dx_km)
        lin = extract_tdr_lineaments(tdr, thd, self.lons, self.lats)
        assert lin.ndim == 2
        assert lin.shape[1] == 3


# ─────────────────────────────────────────────────────────────────────────────
# PSO inversion
# ─────────────────────────────────────────────────────────────────────────────

class TestPSOInversion:
    def setup_method(self):
        self.col, self.lats, dx_deg = make_profile(ny=30)
        self.dx_km = dx_deg * 111.0
        self.layer_tops = [0.52, 3.2]
        self.layer_dzs  = [2.68, 4.8]
        self.quick_params = PSOParams(n_particles=5, n_iterations=5, random_seed=0)

    def test_gravity_kernel_shape(self):
        lat_s = self.lats[:10]
        K = build_gravity_kernel(lat_s, self.layer_tops, self.layer_dzs, self.dx_km)
        n_obs = len(lat_s); n_l = len(self.layer_tops)
        assert K.shape == (n_obs, n_obs * n_l)

    def test_magnetic_kernel_shape(self):
        lat_s = self.lats[:10]
        K, Kmax = build_magnetic_kernel(lat_s, self.layer_tops, self.layer_dzs, self.dx_km)
        n_obs = len(lat_s); n_l = len(self.layer_tops)
        assert K.shape == (n_obs, n_obs * n_l)
        assert Kmax > 0

    def test_pso_gravity_result_type(self):
        result = run_pso_gravity_profile(
            grid_column=self.col, lats=self.lats,
            profile_lon=28.0,
            layer_tops_km=self.layer_tops, layer_dzs_km=self.layer_dzs,
            dx_grid_km=self.dx_km, n_subsample=10,
            params=self.quick_params)
        assert result.data_type == 'gravity'
        assert result.rms >= 0
        assert result.model.shape[0] == len(self.layer_tops)

    def test_pso_magnetic_result_type(self):
        result = run_pso_magnetic_profile(
            grid_column=self.col, lats=self.lats,
            profile_lon=28.0,
            layer_tops_km=self.layer_tops, layer_dzs_km=self.layer_dzs,
            dx_grid_km=self.dx_km, n_subsample=10,
            params=PSOParams(n_particles=5, n_iterations=5,
                             param_min=-0.15, param_max=0.15, random_seed=0))
        assert result.data_type == 'magnetic'
        assert np.all(result.model <= 0.15 + 1e-6)

    def test_pso_convergence_monotone(self):
        result = run_pso_gravity_profile(
            grid_column=self.col, lats=self.lats, profile_lon=28.0,
            layer_tops_km=self.layer_tops, layer_dzs_km=self.layer_dzs,
            dx_grid_km=self.dx_km, n_subsample=10,
            params=PSOParams(n_particles=8, n_iterations=20, random_seed=1))
        # Global best should never increase
        conv = result.convergence
        for i in range(1, len(conv)):
            assert conv[i] <= conv[i - 1] + 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# ABIC inversion
# ─────────────────────────────────────────────────────────────────────────────

class TestABICInversion:
    def setup_method(self):
        self.grid, self.lons, self.lats, self.dx_deg = make_synthetic_grid(ny=20, nx=40)
        self.depths = [0.52, 3.2]
        self.dzs    = [2.68, 4.8]
        self.params = ABICParams(log_omega_min=2.0, log_omega_max=5.0,
                                 n_grid_search=5, param_clip=0.80)

    def test_spectral_kernel_gravity(self):
        H = _gravity_spectral_kernel(20, 40, dx_km=0.5, z_km=1.0, dz_km=2.0)
        assert H.shape == (20, 40)
        assert np.all(np.isfinite(H))

    def test_spectral_kernel_magnetic(self):
        H = _magnetic_spectral_kernel(20, 40, dx_km=0.5, z_km=1.0, dz_km=2.0)
        assert H.shape == (20, 40)

    def test_abic_gravity_runs(self):
        result = run_abic_gravity(
            self.grid, self.lons, self.lats, self.dx_deg,
            depths_km=self.depths, dzs_km=self.dzs,
            subsample_step=2, params=self.params, verbose=False)
        assert len(result.model_layers) == 2
        assert result.opt_omega > 0
        assert all(r >= 0 for r in result.rms_layers)

    def test_abic_magnetic_runs(self):
        result = run_abic_magnetic(
            self.grid, self.lons, self.lats, self.dx_deg,
            depths_km=self.depths, dzs_km=self.dzs,
            subsample_step=2,
            params=ABICParams(log_omega_min=2.0, log_omega_max=5.0,
                              n_grid_search=5, param_clip=0.15),
            verbose=False)
        assert result.data_type == 'magnetic'
        for layer in result.model_layers:
            assert np.all(np.abs(layer) <= 0.15 + 1e-6)

    def test_abic_clip_respected(self):
        params = ABICParams(log_omega_min=1.0, log_omega_max=3.0,
                            n_grid_search=3, param_clip=0.05)
        result = run_abic_gravity(
            self.grid, self.lons, self.lats, self.dx_deg,
            depths_km=self.depths, dzs_km=self.dzs,
            subsample_step=4, params=params, verbose=False)
        for layer in result.model_layers:
            assert np.all(np.abs(layer) <= 0.05 + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Li-Oldenburg inversion (small profile for speed)
# ─────────────────────────────────────────────────────────────────────────────

class TestLiOldenburg:
    def setup_method(self):
        n = 12  # tiny profile for fast testing
        self.lon  = np.linspace(27.5, 29.0, n)
        rng       = np.random.default_rng(7)
        self.anom = 10 * np.sin(np.linspace(0, 2 * np.pi, n)) + rng.standard_normal(n)
        self.tops = [0.52, 3.2]
        self.dzs  = [2.68, 4.8]
        self.params = LiOldenburgParams(n_iterations=5, param_clip=0.8)

    def test_gravity_output_shape(self):
        res = run_li_oldenburg(
            self.anom, self.lon, self.tops, self.dzs,
            dx_grid_km=0.5, data_type='gravity',
            params=self.params, verbose=False)
        assert res.model.shape == (2, 12)

    def test_magnetic_depth_weight(self):
        params_mag = LiOldenburgParams(
            n_iterations=5, depth_weight_beta=3.0, param_clip=0.15)
        res = run_li_oldenburg(
            self.anom, self.lon, self.tops, self.dzs,
            dx_grid_km=0.5, data_type='magnetic',
            params=params_mag, verbose=False)
        assert res.data_type == 'magnetic'
        assert np.all(np.abs(res.model) <= 0.15 + 1e-6)

    def test_convergence_list_length(self):
        res = run_li_oldenburg(
            self.anom, self.lon, self.tops, self.dzs,
            dx_grid_km=0.5, params=self.params, verbose=False)
        assert len(res.convergence) <= self.params.n_iterations
        assert len(res.convergence) >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
